"""
classifier_agent.py — Classifier & RAG Agent powered by Qwen 2.5 3B Instruct

Supports 3 inference backends:
  1. LMStudio (Local) — OpenAI-compatible API at localhost:1234
  2. ZeroGPU (HF Spaces) — transformers + @spaces.GPU decorator
  3. Transformers (CPU) — direct transformers loading (slow fallback)

Prompt Strategy:
  - Few-shot examples with real Indian banking email patterns
  - Chain-of-thought reasoning for ambiguous classification
  - Structured JSON output schemas for reliable extraction
  - Context-aware prompts with Indian banking terminology

Handles email classification, transaction extraction, offer extraction,
RAG-based summarization, and Plotly visualization generation.
All analytics operations are 100% offline using the local SQLite database.
"""

import gc
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import plotly.graph_objects as go

from database import BankingDatabase
from config import (
    QWEN_MAX_TOKENS, QWEN_TEMPERATURE,
    QWEN_CONTEXT_LENGTH, QWEN_THREADS,
    QWEN_HF_MODEL_ID, QWEN_MODEL_PATH,
    LMSTUDIO_BASE_URL, LMSTUDIO_MODEL,
    BACKEND_LMSTUDIO, BACKEND_ZEROGPU, BACKEND_GGUF,
    DEFAULT_BACKEND,
    TRANSACTION_KEYWORDS, OFFER_KEYWORDS, STATEMENT_KEYWORDS, ALERT_KEYWORDS,
    PLOT_DARK_BG, PLOT_SURFACE, PLOT_TEXT_COLOR, PLOT_GRID_COLOR, PLOT_ACCENT_COLORS,
    format_inr,
)

logger = logging.getLogger(__name__)

# ─── Try importing optional dependencies ─────────────────────────────────────
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    class spaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is None:
                return lambda f: f
            return func

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES — Few-Shot, Chain-of-Thought, Structured Output
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an elite AI system specialized in analyzing Indian banking and financial emails. You excel at:
- Accurately classifying emails from major Indian banks (e.g., HDFC, ICICI, SBI, Axis, Kotak, IDFC First).
- Extracting exact transaction details, including UPI, NEFT, IMPS, RTGS, debit/credit cards, and EMI deductions.
- Identifying promotional offers, cashback deals, and reward programs with their constraints.
- Understanding uniquely Indian financial terminology (₹, Rs., lakhs, crores, EMI, SIP, FD).
- Parsing Indian date formats (DD/MM/YYYY) and the Indian numbering system (e.g., 1,50,000.00).

Your reasoning must be logical and your extractions flawless. Always adhere strictly to the requested JSON schemas. Be concise, precise, and objective."""

CLASSIFY_FEW_SHOT_PROMPT = """You are an expert financial data extraction agent specializing in the Indian banking, credit cards, and digital payments ecosystem. Your task is to analyze banking/credit emails and classify them into a precise taxonomy.

DEFINITIONS:
1. FUNDS_CREDITED_ALERT: Notifications of incoming money where an explicit amount has been credited to a specific account number, card number, or wallet. Includes salary NEFT/RTGS, UPI receives, merchant refunds, cashback credited, or interest payouts. MUST contain phrases like "credited to A/c", "credited to your card", "added to wallet", "received in your account", "refunded to your account" along with a specific account/card identifier. Refund emails with "x/- has been refunded into your account" are FUNDS_CREDITED_ALERT.
2. FUNDS_DEBITED_ALERT: Notifications of outgoing money where an explicit amount has been debited from a specific account number or card number. Includes UPI sends, POS card swipes, ATM withdrawals, auto-debits (NACH/EMI), or platform fees. MUST contain phrases like "debited from A/c", "debited from your card", "spent on" with a specific card/account identifier.
3. OTP_SECURITY_ALERT: RBI-mandated 2-Factor Authentication OTPs, internet banking login alerts, or password change requests.
4. CREDIT_LOAN_PROMOTION: Marketing for pre-approved personal loans, car/home loans, No-Cost EMI options, or lifestyle partner discounts. Emails promoting a new credit card (e.g. "Limit: 1,00,000/- with your pre-approved XYZ Credit Card") are PROMOTIONS even if they mention a limit amount, because no card number is specified — they are selling you a new product.
5. REGULATORY_KYC_NOTICE: Compliance notifications regarding RBI guidelines, Re-KYC deadlines, PAN-Aadhaar linking, or positive pay system alerts.
6. ACCOUNT_STATEMENT_BILL: Monthly savings account e-statements, Credit Card Total/Minimum Amount Due bills, or AMB (Average Monthly Balance) non-maintenance warnings. Subjects often contain "Statement" or "e-Statement".
7. CREDIT_SCORE_BUREAU_ALERT: Monthly CIBIL, Experian, or CRIF score updates, credit health reports, or new credit inquiry alerts sent by banks or financial platforms.
8. CREDIT_LIMIT_CARD_MANAGEMENT: Official bank communications regarding Credit Limit Increases (CLI), card upgrades, activation/blocking requests, or card application status tracking.

INSTRUCTIONS:
- Step 1: Read the email SUBJECT first. The subject is the STRONGEST signal and should be checked BEFORE analyzing the body.
  * Transaction subjects: "Debit Alert", "Transaction Alert", "Credit Alert", "Debit Card Transaction", "Fund Transfer".
  * Promotion subjects: "Pre Approved", "Upgrade", "Rewards Points", "Membership", "Cashback Offer", "Exclusive Offer", "Special Offer", "Congratulations", "Limited Period", "wish list", "personal loan", "loan". If the subject contains any of these promotion keywords, classify as CREDIT_LOAN_PROMOTION or CREDIT_LIMIT_CARD_MANAGEMENT — do NOT classify as a transaction even if the body mentions amounts.
- Step 2: Only if the subject is ambiguous, read the body. Note Indian financial terminology (INR, Rs., amount/-, Lakh, Cr).
- Step 3: Determine the primary trigger. Pay strict attention to the direction of cash flow (Credited vs. Debited).
- Step 4: CRITICAL: For FUNDS_CREDITED_ALERT or FUNDS_DEBITED_ALERT, the email MUST contain a specific account number (A/c XX1234) or card number (ending 5678) AND use explicit debit/credit language ("debited from", "credited to"). If the email mentions amounts alongside promotional language (reward points, upgrade, membership, cashback offer, discount, limit) without a specific card/account number being debited/credited, it is a PROMOTION, not a transaction.
- Step 5: Refund emails ("refunded to", "reversal credited") with a specific account/card are FUNDS_CREDITED_ALERT.
- Step 6: Extract the Bank/Fintech Name and the numeric value involved.
- Step 7: Output your response as a valid JSON object. Do not output any markdown code blocks, introductory text, or conversational filler.

OUTPUT SCHEMA:
{{
  "thought_process": "<1-2 sentences in English analyzing the language, direction of money flow, or compliance keywords before choosing the category>",
  "category": "<Must be exactly one of the 8 categories listed above>",
  "bank_or_platform": "<Name of the Indian bank or fintech platform>",
  "amount_inr": <Float value of the transaction, bill, or credit limit if applicable; otherwise null>
}}

EXAMPLES:

Subject: Alert: Money Received via UPI
Body: Dear Customer, Rs. 50 /- has been credited into your account XX8932 linked to UPI ID aditya@ybl.
Output:
{{
  "thought_process": "The email uses the phrase 'has been credited', indicating an incoming flow of funds into the user's account via the UPI network.",
  "category": "FUNDS_CREDITED_ALERT",
  "bank_or_platform": "Yes Bank / UPI",
  "amount_inr": 50.00
}}

Subject: Transaction Alert for your HDFC Bank Debit Card
Body: INR 1,250.00 debited from A/c **4421 on 14-Jun-26. Info: SWIGGY*MUMBAI. Available Bal: INR 14,020.50.
Output:
{{
  "thought_process": "The keyword 'debited' explicitly marks this as an outgoing transaction where money left the account for a merchant purchase.",
  "category": "FUNDS_DEBITED_ALERT",
  "bank_or_platform": "HDFC Bank",
  "amount_inr": 1250.00
}}

Subject: Refund Processed for Order #98432
Body: Hi, your refund of ₹899 for your recent Myntra return has been processed and credited to your ICICI Bank Credit Card ending 0021.
Output:
{{
  "thought_process": "Although this is related to a purchase return, the primary trigger is money being returned ('credited') to the credit card.",
  "category": "FUNDS_CREDITED_ALERT",
  "bank_or_platform": "ICICI Bank",
  "amount_inr": 899.00
}}

Subject: Pre-Approved Kotak League Credit Card
Body: Congratulations! You are eligible for a pre-approved Kotak League Platinum Credit Card with a limit of Rs. 1,00,000/-. Apply now with zero documentation.
Output:
{{
  "thought_process": "This email promotes a NEW credit card product with a limit amount. No existing card number is referenced, and no money was actually credited. The word 'limit' indicates a credit line offer, not a transaction.",
  "category": "CREDIT_LOAN_PROMOTION",
  "bank_or_platform": "Kotak Mahindra Bank",
  "amount_inr": 100000.00
}}

Subject: Credit Limit Enhancement on your HDFC Bank Credit Card
Body: Dear Customer, we are pleased to inform you that the credit limit on your HDFC Bank Credit Card ending 8876 has been enhanced to Rs. 3,50,000/-. 
Output:
{{
  "thought_process": "This email is about a credit limit increase on an existing card. While it mentions a card number and uses the word 'credit', no actual money was credited to the account. This is a card management notification.",
  "category": "CREDIT_LIMIT_CARD_MANAGEMENT",
  "bank_or_platform": "HDFC Bank",
  "amount_inr": 350000.00
}}

Subject: Your Pre Approved Upgrade comes with 15000/- Membership rewards points
Body: Dear Cardholder, Upgrade to Axis Bank Vistara Credit Card and get 15000 Membership reward points. Enjoy complimentary lounge access, 12000/- off on flight bookings, and more. Apply now.
Output:
{{
  "thought_process": "The subject says 'Pre Approved Upgrade' with 'Membership rewards points' — this is a marketing promotion for a card upgrade, not a transaction. The amounts 15000/- and 12000/- refer to reward points and a discount offer, not actual funds being credited or debited. No account or card number is being debited/credited.",
  "category": "CREDIT_LOAN_PROMOTION",
  "bank_or_platform": "Axis Bank",
  "amount_inr": null
}}

Subject: Your wish list calling? ₹50,000 will be credited to your account!
Body: Dear Customer, get a Kotak Personal Loan of ₹50,000 instantly. The amount will be credited to your account in 3 seconds. Apply now!
Output:
{{
  "thought_process": "Although the email mentions 'will be credited' and an amount, it is an offer for a personal loan ('wish list calling', 'Personal Loan'). It is a future marketing offer, not an actual completed transaction.",
  "category": "CREDIT_LOAN_PROMOTION",
  "bank_or_platform": "Kotak Mahindra Bank",
  "amount_inr": 50000.00
}}

Subject: {subject}
Body: {body}
Output:"""


EXTRACT_TRANSACTION_PROMPT = """Carefully extract precise transaction details from this Indian banking email. 

### Indian Banking Context & Guidelines:
- **Currency & Formatting**: Amounts may be prefixed/suffixed with ₹, Rs., Rs, or INR. Use standard decimal format (e.g., 1500.50). Ignore commas from the Indian numbering system (e.g., 1,23,456.78 -> 123456.78).
- **Transaction Types**: UPI, NEFT, IMPS, RTGS, card swipe, EMI, autopay, mandate, ATM withdrawal.
- **Card References**: Extract the last 4 digits if present (e.g., "XX1234", "ending 5678", "card no. ****1234").
- **Payment Mode**: Infer correctly based on keywords (e.g., "UPI Ref" -> UPI, "NEFT/" -> NEFT).

### Few-Shot Examples:

**Example 1 (UPI Debit):**
Email: "Rs.450.00 debited from A/c XX3421 on 12-Jun-25. UPI Ref: 412345678901. Txn to ZOMATO MEDIA PVT. Avl Bal: Rs.23,100.50"
→ {{"amount": 450.00, "transaction_type": "debit", "merchant": "Zomato", "card_last4": "3421", "category": "Food & Dining", "transaction_date": "12-Jun-25", "payment_mode": "UPI"}}

**Example 2 (Credit Card):**
Email: "Alert: HDFC Bank Credit Card XX9876 has been used for a transaction of INR 15,999 at AMAZON SELLER SERVICES on 10/06/2025"
→ {{"amount": 15999.00, "transaction_type": "debit", "merchant": "Amazon", "card_last4": "9876", "category": "Shopping", "transaction_date": "10/06/2025", "payment_mode": "Card"}}

**Example 3 (Credit/Refund):**
Email: "INR 2,100.00 credited to your SBI account XX4567 via NEFT from FLIPKART INTERNET. Ref No: NEFT/2025/123456"
→ {{"amount": 2100.00, "transaction_type": "credit", "merchant": "Flipkart", "card_last4": "4567", "category": "Shopping", "transaction_date": null, "payment_mode": "NEFT"}}

### Now extract from this email:

Bank: {bank_name}
Email: {body_text}

Respond with ONLY valid JSON matching this schema:
{{"amount": <number>, "transaction_type": "debit|credit", "merchant": "<name or null>", "card_last4": "<4 digits or null>", "category": "<category>", "transaction_date": "<date or null>", "payment_mode": "<UPI|Card|NEFT|IMPS|RTGS|EMI|Other>"}}

Categories: Food & Dining, Shopping, Travel, Entertainment, Utilities, Fuel, Health, Education, Insurance, Investment, EMI, Grocery, Transfer, Other"""


EXTRACT_OFFER_PROMPT = """Extract the promotional banking offer from this email. Focus on actionable deal details.

### What to extract:
1. **Offer headline**: The main deal (e.g., "20% cashback on Swiggy")
2. **Bank & card**: Which bank/card is offering this
3. **Discount details**: Percentage, flat amount, max cap
4. **Eligible platforms**: Where can this be used (merchants, categories)
5. **Validity**: Start date, end date, or "limited time"
6. **Terms**: Minimum spend, max cashback, frequency limits

### Few-Shot Examples:

**Example 1:**
Email: "Exclusive for HDFC Millennia Card holders! Get flat 5% cashback on Amazon purchases. Max cashback ₹750. Min transaction ₹2,000. Valid till 30 June 2025. Use code HDFC5."
→ "HDFC Millennia Card: 5% cashback on Amazon (max ₹750, min spend ₹2,000). Use code HDFC5. Valid till 30 June 2025."

**Example 2:**
Email: "Your Axis Bank Ace Credit Card gives you 5% back on bill payments via Google Pay and 2% on all other spends, every day, no cap! Plus earn 5X Edge Reward Points on dining."
→ "Axis Ace Card: 5% back on bill payments via GPay (no cap) + 2% on all other spends + 5X reward points on dining. Ongoing offer."

### Now extract from this email:

Bank: {bank_name}
Email: {body_text}

Respond with a concise 1-3 sentence summary highlighting the deal, eligibility, cap, and validity. If there's no clear offer, respond with "NO_OFFER"."""


ANALYZE_SPENDING_PROMPT = """You are a personal finance analyst for Indian consumers. Analyze this spending data and provide clear insights.

### Spending Data:
{context}

### Your Analysis Must Include:

**1. Key Insights** (3-4 bullet points)
- Identify the biggest spending category and its percentage of total
- Flag any unusual or concerning patterns
- Compare debits vs credits and note the net cash flow direction
- Highlight any recurring or notable transactions

**2. Spending Patterns**
- Which categories dominate spending
- Bank-wise distribution observations
- Monthly trend analysis (increasing/decreasing/stable)

**3. Alerts** (if applicable)
- Spending spikes or sudden jumps
- Categories exceeding typical budgets
- Unusually large single transactions
- Opportunities to optimize credit card or bank usage

Format as clean Markdown with the section headers above. Use Rs. for all amounts in Indian numbering format. Do NOT include savings tips or suggestions — focus purely on data-driven insights and observations."""


SUMMARIZE_OFFERS_PROMPT = """You are a banking rewards expert. Summarize these offers for an Indian consumer, making it easy to find the best deals.

### Offers from database:
{context}

### Format your response as:

## Top Picks (Best 3 Deals)
- Highlight the highest-value offers with clear action items

## All Offers by Bank

### [Bank Name]
- **Offer**: [concise summary]
- **How to avail**: [steps if available]
- **Valid till**: [date if known]

Be concise but actionable."""


class ClassifierAgent:
    """
    Local LLM-powered agent for email classification, data extraction,
    and offline RAG summarization.

    Supports LMStudio (OpenAI API), ZeroGPU, and CPU transformers backends.
    
    Prompt Strategy:
    - Few-shot examples with real Indian banking patterns
    - Chain-of-thought reasoning for classification
    - Structured JSON output for extraction
    - Context-aware RAG prompts for summarization
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._backend = DEFAULT_BACKEND
        self._openai_client = None

    # ─── Backend Management ──────────────────────────────────────────────────

    def set_backend(self, backend_name: str):
        """Switch inference backend. Unloads any previously loaded model."""
        if backend_name == self._backend and self._loaded:
            return
        self.unload_model()
        self._backend = backend_name
        logger.info(f"Classifier backend set to: {backend_name}")

    def get_backend(self) -> str:
        return self._backend

    def check_backend_status(self) -> str:
        """Check if the current backend is available. Returns status string."""
        if self._backend == BACKEND_LMSTUDIO:
            return self._check_lmstudio()
        elif self._backend == BACKEND_ZEROGPU:
            if HAS_SPACES:
                return "[OK] ZeroGPU available"
            return "[WARN] `spaces` package not found -- not running on HF Spaces"
        else:
            return "[OK] CPU backend always available (slow)"

    def _check_lmstudio(self) -> str:
        """Ping LMStudio API to check connectivity."""
        try:
            if not HAS_OPENAI:
                return "[ERROR] `openai` package not installed"
            client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")
            models = client.models.list()
            model_ids = [m.id for m in models.data]
            if model_ids:
                return f"[OK] LMStudio connected -- Models: {', '.join(model_ids[:3])}"
            return "[WARN] LMStudio running but no models loaded"
        except Exception as e:
            return f"[ERROR] LMStudio unreachable: {e}"

    # ─── Model Loading ───────────────────────────────────────────────────────

    def load_model(self, progress_callback=None):
        """Load model based on current backend."""
        if self._loaded:
            return

        def log(msg):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        if self._backend == BACKEND_LMSTUDIO:
            self._load_lmstudio(log)
        elif self._backend == BACKEND_ZEROGPU:
            self._load_zerogpu(log)
        else:
            self._load_transformers_cpu(log)

    def _load_lmstudio(self, log):
        """Initialize OpenAI client for LMStudio."""
        if not HAS_OPENAI:
            log("[ERROR] `openai` package not installed. Run: pip install openai")
            return
        try:
            self._openai_client = OpenAI(
                base_url=LMSTUDIO_BASE_URL,
                api_key="lm-studio",
            )
            # Quick connectivity check
            self._openai_client.models.list()
            self._loaded = True
            log(f"[OK] Connected to LMStudio at {LMSTUDIO_BASE_URL}")
            log(f"  Model: {LMSTUDIO_MODEL}")
        except Exception as e:
            log(f"[ERROR] LMStudio connection failed: {e}")
            log("[WARN] Falling back to rule-based classification")
            self._loaded = False

    def _load_zerogpu(self, log):
        """Load model for ZeroGPU (transformers on HF Spaces GPU)."""
        log("ZeroGPU backend -- loading model with GPU acceleration")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            log(f"  Model: {QWEN_HF_MODEL_ID}")
            log(f"  Loading model (ZeroGPU)...")

            self.tokenizer = AutoTokenizer.from_pretrained(QWEN_HF_MODEL_ID)
            self.model = AutoModelForCausalLM.from_pretrained(
                QWEN_HF_MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model.eval()
            self._loaded = True
            log("[OK] Qwen model loaded on ZeroGPU")
        except Exception as e:
            log(f"[ERROR] ZeroGPU model loading failed: {e}")
            log("[WARN] Falling back to rule-based classification")
            self._loaded = False

    def _load_transformers_cpu(self, log):
        """Load Qwen model via transformers on CPU (slow fallback)."""
        log("Loading Qwen 2.5 3B model via transformers (CPU)...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            log(f"  Model: {QWEN_HF_MODEL_ID}")
            log(f"  Loading to CPU (no GPU)...")

            self.tokenizer = AutoTokenizer.from_pretrained(QWEN_HF_MODEL_ID)
            self.model = AutoModelForCausalLM.from_pretrained(
                QWEN_HF_MODEL_ID,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            self.model.eval()
            self._loaded = True
            log("[OK] Qwen model loaded on CPU")
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            log(f"[ERROR] Model loading failed: {e}")
            log("[WARN] Falling back to rule-based classification")
            self._loaded = False

    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._openai_client = None
        self._loaded = False
        gc.collect()
        logger.info("Classifier model unloaded")

    # ─── Text Generation ─────────────────────────────────────────────────────

    def _generate(self, prompt: str, max_tokens: int = None, system_prompt: str = None) -> str:
        """Generate text using the current backend."""
        if not self._loaded:
            return ""

        max_tokens = max_tokens or QWEN_MAX_TOKENS
        system_prompt = system_prompt or SYSTEM_PROMPT

        if self._backend == BACKEND_LMSTUDIO:
            return self._generate_lmstudio(prompt, max_tokens, system_prompt)
        else:
            return self._generate_transformers(prompt, max_tokens, system_prompt)

    def _generate_lmstudio(self, prompt: str, max_tokens: int, system_prompt: str) -> str:
        """Generate text via LMStudio's OpenAI-compatible API."""
        if not self._openai_client:
            return ""
        try:
            response = self._openai_client.chat.completions.create(
                model=LMSTUDIO_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=QWEN_TEMPERATURE,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LMStudio generation failed: {e}")
            return ""

    @spaces.GPU
    def _generate_transformers(self, prompt: str, max_tokens: int, system_prompt: str) -> str:
        """Generate text using loaded transformers model (ZeroGPU or CPU)."""
        if not self._loaded or self.model is None:
            return ""

        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        # Move inputs to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=QWEN_TEMPERATURE,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    # ─── Rule-Based Classification (Fast, No LLM) ────────────────────────────

    def classify_email_rules(self, subject: str, body_text: str) -> str:
        """Fast rule-based classification using keyword matching to the 8 categories."""
        text = f"{subject} {body_text}".lower()

        scores = {
            "FUNDS_CREDITED_ALERT": sum(1 for kw in ["credited", "received", "added", "refund"] if kw in text),
            "FUNDS_DEBITED_ALERT": sum(1 for kw in ["debited", "spent", "paid", "withdrawn"] if kw in text),
            "OTP_SECURITY_ALERT": sum(1 for kw in ["otp", "login", "password", "security"] if kw in text),
            "CREDIT_LOAN_PROMOTION": sum(1 for kw in ["loan", "pre-approved", "pre approved", "cashback", "offer", "discount", "upgrade", "reward", "points", "membership", "wish list", "will be credited", "personal loan"] if kw in text),
            "REGULATORY_KYC_NOTICE": sum(1 for kw in ["kyc", "rbi", "pan", "aadhaar"] if kw in text),
            "ACCOUNT_STATEMENT_BILL": sum(1 for kw in ["statement", "due", "bill", "amb"] if kw in text),
            "CREDIT_SCORE_BUREAU_ALERT": sum(1 for kw in ["cibil", "experian", "score", "inquiry"] if kw in text),
            "CREDIT_LIMIT_CARD_MANAGEMENT": sum(1 for kw in ["limit", "upgrade", "blocked", "activation"] if kw in text),
        }

        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return "OTHER"
        return best

    # ─── LLM-Powered Classification (Few-Shot + Chain-of-Thought) ─────────────

    def classify_email(self, subject: str, body_text: str) -> Dict:
        """
        Classify an email and extract basic amount/platform data.
        Returns a dictionary matching the new 8-category JSON schema.
        """
        # Always start with rules for speed (fallback)
        rule_result = {"category": self.classify_email_rules(subject, body_text)}

        if self._loaded:
            prompt = CLASSIFY_FEW_SHOT_PROMPT.format(
                subject=subject[:200],
                body=body_text[:600],
            )
            try:
                result = self._generate(prompt, max_tokens=250)
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                        if "category" in parsed:
                            logger.info(f"LLM classification: {parsed['category']} (reasoning: {parsed.get('thought_process', 'N/A')})")
                            return parsed
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")

        return rule_result

    # ─── Transaction Extraction (Rules + LLM Few-Shot) ───────────────────────

    def extract_transaction_rules(self, body_text: str, bank_name: str) -> Optional[Dict]:
        """Extract transaction details using regex patterns."""
        if not body_text:
            return None

        result = {
            "bank_name": bank_name,
            "amount": None,
            "transaction_type": None,
            "merchant": None,
            "card_last4": None,
            "category": "Other",
            "transaction_date": None,
            "payment_mode": "Other",
            "raw_text": body_text[:500],
        }

        text = body_text.lower()

        # Determine debit or credit
        if any(kw in text for kw in ["debited", "spent", "purchase", "withdrawn", "paid"]):
            result["transaction_type"] = "debit"
        elif any(kw in text for kw in ["credited", "received", "refund", "cashback"]):
            result["transaction_type"] = "credit"
        else:
            return None  # Can't determine type

        # Determine payment mode
        if "upi" in text:
            result["payment_mode"] = "UPI"
        elif "neft" in text:
            result["payment_mode"] = "NEFT"
        elif "imps" in text:
            result["payment_mode"] = "IMPS"
        elif "rtgs" in text:
            result["payment_mode"] = "RTGS"
        elif any(kw in text for kw in ["card", "credit card", "debit card"]):
            result["payment_mode"] = "Card"
        elif "emi" in text:
            result["payment_mode"] = "EMI"

        # Extract amount — match ₹, Rs., Rs, INR patterns
        amount_patterns = [
            r'(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d{1,2})?)',
            r'([\d,]+(?:\.\d{1,2})?)\s*(?:₹|rs\.?|inr)',
            r'amount[:\s]*(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d{1,2})?)',
        ]
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(",", "")
                try:
                    result["amount"] = float(amount_str)
                    break
                except ValueError:
                    continue

        if not result["amount"]:
            return None  # Can't extract amount

        # Extract card last 4 digits
        card_match = re.search(r'(?:card|xx|ending)\s*(?:with|in|no\.?)?\s*(\d{4})', text)
        if card_match:
            result["card_last4"] = card_match.group(1)

        # Extract merchant
        merchant_patterns = [
            r'(?:at|to|from|merchant|towards)\s+([A-Za-z0-9\s&\'-]+?)(?:\s+on|\s+via|\s+for|\.|\n)',
            r'(?:info|details?):\s*([A-Za-z0-9\s&\'-]+?)(?:\s*$|\n)',
        ]
        for pattern in merchant_patterns:
            match = re.search(pattern, body_text, re.IGNORECASE)
            if match:
                merchant = match.group(1).strip()
                if len(merchant) > 3 and len(merchant) < 50:
                    result["merchant"] = merchant
                    break

        # Categorize based on merchant/keywords
        result["category"] = self._categorize_transaction(text, result.get("merchant", ""))

        # Extract date
        date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{2,4})',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["transaction_date"] = self._normalize_date(match.group(1))
                break

        return result

    @staticmethod
    def _normalize_date(date_str: str) -> Optional[str]:
        """Normalize various Indian date formats to ISO YYYY-MM-DD for SQLite."""
        from datetime import datetime as dt
        formats = [
            "%d-%b-%y", "%d-%b-%Y",       # 12-Jun-25, 12-Jun-2025
            "%d/%m/%Y", "%d/%m/%y",       # 10/06/2025, 10/06/25
            "%d-%m-%Y", "%d-%m-%y",       # 10-06-2025, 10-06-25
            "%d %b %Y", "%d %b %y",       # 10 Jun 2025
            "%d %B %Y", "%d %B %y",       # 10 June 2025
        ]
        date_str = date_str.strip()
        for fmt in formats:
            try:
                return dt.strptime(date_str, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return date_str  # Return as-is if no format matches

    def _categorize_transaction(self, text: str, merchant: str) -> str:
        """Categorize transaction based on keywords and merchant name."""
        combined = f"{text} {merchant}".lower()

        categories = {
            "Food & Dining": ["swiggy", "zomato", "restaurant", "food", "cafe", "dining", "pizza", "burger"],
            "Shopping": ["amazon", "flipkart", "myntra", "ajio", "shopping", "store", "mall", "mart"],
            "Travel": ["uber", "ola", "makemytrip", "irctc", "flight", "hotel", "travel", "cab", "train"],
            "Entertainment": ["netflix", "hotstar", "spotify", "movie", "theatre", "gaming"],
            "Utilities": ["electricity", "water", "gas", "broadband", "jio", "airtel", "recharge", "bill"],
            "Fuel": ["petrol", "diesel", "fuel", "hp", "indian oil", "bharat petroleum"],
            "Health": ["pharmacy", "hospital", "doctor", "medical", "health", "apollo", "practo"],
            "Education": ["school", "college", "course", "udemy", "education", "tuition"],
            "Insurance": ["insurance", "premium", "lic", "policy"],
            "Investment": ["mutual fund", "sip", "stock", "zerodha", "groww", "investment"],
            "EMI": ["emi", "loan", "installment"],
            "Grocery": ["grocery", "bigbasket", "blinkit", "zepto", "dmart", "supermarket"],
            "Transfer": ["transfer", "upi", "neft", "imps", "rtgs", "sent to"],
        }

        for category, keywords in categories.items():
            if any(kw in combined for kw in keywords):
                return category

        return "Other"

    def extract_transaction(self, body_text: str, bank_name: str) -> Optional[Dict]:
        """
        Extract transaction using multi-strategy approach:
        1. Rule-based regex extraction (fast, handles 80% of emails)
        2. LLM few-shot extraction (for complex/unusual formats)
        """
        # Try rule-based extraction first (fast)
        result = self.extract_transaction_rules(body_text, bank_name)
        if result and result.get("amount"):
            return result

        # LLM fallback with few-shot structured prompt
        if self._loaded:
            try:
                prompt = EXTRACT_TRANSACTION_PROMPT.format(
                    bank_name=bank_name,
                    body_text=body_text[:800],
                )
                response = self._generate(prompt, max_tokens=250)
                # Try to parse JSON from response
                json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    if parsed.get("amount"):
                        parsed["bank_name"] = bank_name
                        parsed["raw_text"] = body_text[:500]
                        return parsed
            except Exception as e:
                logger.warning(f"LLM transaction extraction failed: {e}")

        return result

    # ─── Offer Extraction (LLM Few-Shot) ─────────────────────────────────────

    def extract_offer(self, email_data: Dict) -> Optional[Dict]:
        """Extract promotional offer details using few-shot LLM prompt."""
        body_text = email_data.get("body_text", "")
        body_html = email_data.get("body_html", "")
        bank_name = email_data.get("bank_name", "Unknown")
        email_date = email_data.get("received_date", "Unknown Date")
        
        if not body_text or len(body_text) < 20:
            return None

        # Extract links from HTML
        links = []
        if body_html:
            import bs4
            soup = bs4.BeautifulSoup(body_html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http") and "unsubscribe" not in href.lower() and len(href) < 150:
                    links.append(href)
        
        # Deduplicate and limit links
        unique_links = list(dict.fromkeys(links))[:3]
        links_str = "\n".join(unique_links)

        date_and_links = f"\n\nValid From/Received: {email_date}"
        if links_str:
            date_and_links += f"\nLinks:\n{links_str}"

        # If LLM is loaded, use few-shot extraction prompt
        if self._loaded:
            try:
                prompt = EXTRACT_OFFER_PROMPT.format(
                    bank_name=bank_name,
                    body_text=body_text[:700],
                )
                response = self._generate(prompt, max_tokens=250)
                if response and "NO_OFFER" not in response.upper():
                    return {
                        "bank_name": bank_name,
                        "offer_text": response.strip() + date_and_links,
                        "source": "email_body",
                        "category": self._categorize_offer(response),
                    }
            except Exception as e:
                logger.warning(f"LLM offer extraction failed: {e}")

        # Fallback: use raw text as offer
        offer_text = body_text[:1000]
        return {
            "bank_name": bank_name,
            "offer_text": offer_text + date_and_links,
            "source": "email_body",
            "category": self._categorize_offer(offer_text),
        }

    def _categorize_offer(self, text: str) -> str:
        """Categorize an offer based on keywords."""
        text_lower = text.lower()
        categories = {
            "Cashback": ["cashback", "cash back"],
            "Discount": ["discount", "% off", "flat off"],
            "Rewards": ["reward", "points", "bonus"],
            "Travel": ["travel", "flight", "hotel", "booking"],
            "Shopping": ["shopping", "amazon", "flipkart"],
            "Dining": ["dining", "food", "restaurant"],
            "Fuel": ["fuel", "petrol"],
            "EMI": ["emi", "no cost"],
        }
        for cat, keywords in categories.items():
            if any(kw in text_lower for kw in keywords):
                return cat
        return "General"

    # ─── RAG Summarization (Context-Aware Prompts) ───────────────────────────

    def summarize_offers(self, db: BankingDatabase) -> str:
        """
        RAG: Query database for offers and generate summary with context-aware prompt.
        100% offline — only uses SQLite database and local model.
        """
        offers = db.get_offers(limit=30)

        if not offers:
            return "No offers found in the database. Run a Network Sync first to fetch banking emails."

        # Build context from database records
        context_parts = []
        for offer in offers:
            bank = offer.get("bank_name", "Unknown")
            text = offer.get("offer_text", "")[:200]
            source = offer.get("source", "email_body")
            cat = offer.get("category", "General")
            context_parts.append(f"[{bank}] [{cat}] ({source}): {text}")

        context = "\n".join(context_parts)

        if self._loaded:
            try:
                prompt = SUMMARIZE_OFFERS_PROMPT.format(context=context[:2500])
                summary = self._generate(prompt, max_tokens=800)
                if summary:
                    return summary
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}")

        # Fallback: structured text summary
        by_bank = {}
        for offer in offers:
            bank = offer.get("bank_name", "Unknown")
            if bank not in by_bank:
                by_bank[bank] = []
            by_bank[bank].append(offer.get("offer_text", "")[:150])

        lines = ["# Banking Offers Summary\n"]
        for bank, offer_texts in by_bank.items():
            lines.append(f"## {bank}")
            for text in offer_texts[:5]:
                lines.append(f"- {text}")
            lines.append("")

        return "\n".join(lines)

    def analyze_spending(self, db: BankingDatabase) -> str:
        """
        RAG: Analyze spending patterns with context-aware prompt.
        100% offline — only uses SQLite database and local model.
        """
        summary = db.get_spending_summary()

        if not summary or summary["total_count"] == 0:
            return "No transaction data found. Run a Network Sync to fetch banking emails, then transactions will be extracted automatically."

        # Build analytics context
        context = f"""Total Debits: {format_inr(summary['total_debits'])}
Total Credits: {format_inr(summary['total_credits'])}
Total Transactions: {summary['total_count']}
Net Outflow: {format_inr(summary['total_debits'] - summary['total_credits'])}

By Category (sorted by spend):
"""
        for cat in summary["by_category"]:
            pct = (cat['total'] / summary['total_debits'] * 100) if summary['total_debits'] else 0
            context += f"  {cat['category']}: {format_inr(cat['total'])} ({cat['count']} txns, {pct:.1f}%)\n"

        context += "\nBy Bank:\n"
        for bank in summary["by_bank"]:
            context += f"  {bank['bank_name']}: {format_inr(bank['total'])} ({bank['count']} txns)\n"

        context += "\nMonthly Trend:\n"
        for m in summary.get("monthly", []):
            context += f"  {m['month']}: Debits {format_inr(m['debits'])}, Credits {format_inr(m['credits'])}\n"

        if self._loaded:
            try:
                prompt = ANALYZE_SPENDING_PROMPT.format(context=context)
                analysis = self._generate(prompt, max_tokens=800)
                if analysis:
                    return analysis
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")

        # Fallback: return the raw summary as formatted text
        return f"# Spending Analysis\n\n{context}"

    # ─── Plotly Visualization Pipeline ────────────────────────────────────────

    def _plotly_layout(self, title: str, **kwargs) -> dict:
        """Standard dark-mode Plotly layout matching the app theme."""
        layout = dict(
            title=dict(text=title, font=dict(size=18, color=PLOT_TEXT_COLOR, family="Inter, sans-serif"), x=0.5),
            paper_bgcolor=PLOT_DARK_BG,
            plot_bgcolor=PLOT_SURFACE,
            font=dict(color=PLOT_TEXT_COLOR, family="Inter, sans-serif", size=12),
            margin=dict(l=60, r=30, t=60, b=50),
            hoverlabel=dict(
                bgcolor=PLOT_SURFACE,
                font_size=12,
                font_family="Inter, sans-serif",
                font_color=PLOT_TEXT_COLOR,
                bordercolor=PLOT_GRID_COLOR,
            ),
        )
        layout.update(kwargs)
        return layout

    def plot_spending_by_category(self, db: BankingDatabase) -> Optional[go.Figure]:
        """Generate an interactive Plotly donut chart of spending by category."""
        summary = db.get_spending_summary()
        categories = summary.get("by_category", [])

        if not categories:
            return None

        labels = [c["category"] for c in categories[:8]]
        values = [c["total"] for c in categories[:8]]
        colors = PLOT_ACCENT_COLORS[:len(labels)]

        # Format hover text with Indian rupee notation
        hover_text = [f"{label}<br>{format_inr(val)}" for label, val in zip(labels, values)]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker=dict(colors=colors, line=dict(color=PLOT_DARK_BG, width=2)),
            textinfo="percent",
            textfont=dict(size=11, color=PLOT_TEXT_COLOR),
            hovertext=hover_text,
            hoverinfo="text",
            direction="clockwise",
            sort=True,
        )])

        fig.update_layout(**self._plotly_layout(
            "Spending by Category",
            showlegend=True,
            legend=dict(
                font=dict(size=11, color=PLOT_TEXT_COLOR),
                bgcolor="rgba(26,26,46,0.8)",
                bordercolor=PLOT_GRID_COLOR,
                borderwidth=1,
            ),
            height=400,
        ))

        # Add center annotation
        total = sum(values)
        fig.add_annotation(
            text=f"<b>{format_inr(total)}</b><br><span style='font-size:11px;color:{PLOT_GRID_COLOR}'>Total Spent</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=PLOT_ACCENT_COLORS[0]),
        )

        return fig

    def plot_spending_by_bank(self, db: BankingDatabase) -> Optional[go.Figure]:
        """Generate an interactive Plotly horizontal bar chart of spending by bank."""
        summary = db.get_spending_summary()
        by_bank = summary.get("by_bank", [])

        if not by_bank:
            return None

        banks = [b["bank_name"] for b in by_bank]
        values = [b["total"] for b in by_bank]
        colors = PLOT_ACCENT_COLORS[:len(banks)]
        hover_text = [f"{bank}: {format_inr(val)}" for bank, val in zip(banks, values)]

        # Reverse for horizontal bars (top-down order)
        banks.reverse()
        values.reverse()
        colors_rev = list(reversed(colors))
        hover_text.reverse()

        fig = go.Figure(data=[go.Bar(
            y=banks,
            x=values,
            orientation="h",
            marker=dict(
                color=colors_rev,
                line=dict(color=PLOT_DARK_BG, width=1.5),
            ),
            hovertext=hover_text,
            hoverinfo="text",
            text=[format_inr(v) for v in values],
            textposition="outside",
            textfont=dict(size=10, color=PLOT_TEXT_COLOR),
        )])

        fig.update_layout(**self._plotly_layout(
            "Spending by Bank",
            xaxis=dict(
                gridcolor=PLOT_GRID_COLOR,
                gridwidth=0.5,
                zeroline=False,
                showticklabels=False,
            ),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
            height=max(300, len(banks) * 60 + 100),
        ))

        return fig

    def plot_daywise_trend(self, db: BankingDatabase) -> Optional[go.Figure]:
        """Generate an interactive Plotly line chart of daywise spending with drill down."""
        summary = db.get_spending_summary()
        daywise = summary.get("daywise", [])

        if not daywise:
            return None

        days = [d["day"] for d in daywise]
        debits = [d["debits"] for d in daywise]
        credits_ = [d["credits"] for d in daywise]

        fig = go.Figure()

        # Debits trace
        fig.add_trace(go.Scatter(
            x=days, y=debits,
            mode="lines+markers",
            name="Debits",
            line=dict(color=PLOT_ACCENT_COLORS[1], width=2),
            marker=dict(size=4, color=PLOT_ACCENT_COLORS[1]),
            fill="tozeroy",
            fillcolor="rgba(255,101,132,0.1)",
            hovertext=[f"Debits: {format_inr(d)}" for d in debits],
            hoverinfo="text+x",
        ))

        # Credits trace
        fig.add_trace(go.Scatter(
            x=days, y=credits_,
            mode="lines+markers",
            name="Credits",
            line=dict(color=PLOT_ACCENT_COLORS[2], width=2),
            marker=dict(size=4, symbol="square", color=PLOT_ACCENT_COLORS[2]),
            fill="tozeroy",
            fillcolor="rgba(67,232,216,0.1)",
            hovertext=[f"Credits: {format_inr(c)}" for c in credits_],
            hoverinfo="text+x",
        ))

        fig.update_layout(**self._plotly_layout(
            "Transaction Trend (Daywise)",
            xaxis=dict(
                gridcolor=PLOT_GRID_COLOR, 
                gridwidth=0.5, 
                title="Date", 
                type="date",
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]),
                    bgcolor="rgba(26,26,46,0.8)",
                    activecolor=PLOT_ACCENT_COLORS[0],
                    font=dict(color=PLOT_TEXT_COLOR)
                ),
                rangeslider=dict(visible=True, bgcolor="rgba(26,26,46,0.5)")
            ),
            yaxis=dict(gridcolor=PLOT_GRID_COLOR, gridwidth=0.5, title="Amount (₹)", zeroline=False),
            legend=dict(
                font=dict(size=11, color=PLOT_TEXT_COLOR),
                bgcolor="rgba(26,26,46,0.8)",
                bordercolor=PLOT_GRID_COLOR,
                borderwidth=1,
            ),
            hovermode="x unified",
            height=450,
        ))

        return fig

    # ─── Process New Emails ──────────────────────────────────────────────────

    def process_new_emails(self, db: BankingDatabase, email_ids: List[int], progress_callback=None, progress_bar=None) -> Dict:
        """
        Classify new emails and extract transactions/offers in parallel.
        Uses few-shot CoT classification and structured extraction.
        """
        import concurrent.futures
        import threading
        
        results = {"classified": 0, "transactions_extracted": 0, "offers_extracted": 0}
        total_emails = len(email_ids)
        if total_emails == 0:
            return results
            
        db_lock = threading.Lock()

        def log(msg):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        def process_single_email(eid):
            email_data = db.get_email_by_id(eid)
            if not email_data:
                return
                
            subject = email_data.get("subject", "")
            body_text = email_data.get("body_text", "")
            bank_name = email_data.get("bank_name", "Unknown")

            # Classify with new JSON schema (LLM call - can be parallel)
            email_result = self.classify_email(subject, body_text)
            email_type = email_result.get("category", "OTHER").upper()
            
            # Secondary extraction if needed (LLM call - can be parallel)
            txn = None
            offer = None
            if email_type in ("FUNDS_CREDITED_ALERT", "FUNDS_DEBITED_ALERT"):
                txn = self.extract_transaction(body_text, bank_name)
                if not txn:
                    txn = {
                        "bank_name": email_result.get("bank_or_platform", bank_name),
                        "category": "Other",
                        "payment_mode": "Other",
                        "merchant": "Unknown"
                    }
                amount_inr = email_result.get("amount_inr")
                if amount_inr is not None:
                    txn["amount"] = amount_inr
                txn["transaction_type"] = "credit" if email_type == "FUNDS_CREDITED_ALERT" else "debit"
                txn["email_id"] = eid
                if not txn.get("transaction_date"):
                    txn["transaction_date"] = email_data.get("received_date")
            elif email_type == "CREDIT_LOAN_PROMOTION":
                offer = self.extract_offer(email_data)
                if offer and offer.get("offer_text"):
                    offer["email_id"] = eid
            
            # DB writes and results update (Synchronized)
            with db_lock:
                db.update_email_type(eid, email_type)
                results["classified"] += 1
                
                # Progress logging per email
                desc = f"Classified {results['classified']}/{total_emails} emails... (Last: {email_type})"
                log(desc)
                if progress_bar:
                    progress_bar(results['classified'] / total_emails, desc=desc)
                
                if txn and txn.get("amount"):
                    db.insert_transaction(txn)
                    results["transactions_extracted"] += 1
                    log(f"  [{bank_name}] {txn['transaction_type'].upper()}: {format_inr(txn['amount'])}")
                
                if offer and offer.get("offer_text"):
                    db.insert_offer(offer)
                    results["offers_extracted"] += 1
                    log(f"  [{bank_name}] Promotion extracted")

        # Use 5 threads for parallel LLM inference
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_single_email, eid) for eid in email_ids]
            concurrent.futures.wait(futures)

        log(f"\nProcessing complete: {results['classified']} classified, "
            f"{results['transactions_extracted']} transactions, "
            f"{results['offers_extracted']} offers")

        return results
