"""
config.py — Centralized Configuration for Indian Banking Email Intelligence
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
# Auto-detect HF Spaces environment
IS_HF_SPACE = os.environ.get("SPACE_ID") is not None

# Use persistent storage if on HF Spaces, otherwise local 'data' folder
if IS_HF_SPACE:
    DATA_DIR = "/data"
else:
    DATA_DIR = os.path.join(BASE_DIR, "data")

BANNERS_DIR = os.path.join(DATA_DIR, "banners")
DB_PATH = os.path.join(DATA_DIR, "banking_vault.db")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BANNERS_DIR, exist_ok=True)

# ─── Inference Backend ───────────────────────────────────────────────────────
BACKEND_LMSTUDIO = "LMStudio (Local)"
BACKEND_ZEROGPU = "ZeroGPU (HF Spaces)"
BACKEND_GGUF = "Local GGUF (llama.cpp)"
INFERENCE_BACKENDS = [BACKEND_LMSTUDIO, BACKEND_ZEROGPU, BACKEND_GGUF]

DEFAULT_BACKEND = BACKEND_ZEROGPU if IS_HF_SPACE else BACKEND_LMSTUDIO

# ─── LMStudio Configuration ─────────────────────────────────────────────────
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen2.5-3b-instruct")
LMSTUDIO_VISION_MODEL = os.environ.get("LMSTUDIO_VISION_MODEL", "moondream")

# ─── Model Identifiers (HuggingFace Hub) ─────────────────────────────────────
QWEN_HF_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MOONDREAM_HF_MODEL_ID = "vikhyatk/moondream2"
MOONDREAM_HF_REVISION = "2025-01-09"

# ─── Legacy: Local GGUF model path (for CPU fallback) ───────────────────────
QWEN_MODEL_PATH = os.path.join(MODELS_DIR, "qwen2.5-coder-3b-instruct-q4_k_m.gguf")

# ─── LLM Settings ────────────────────────────────────────────────────────────
QWEN_MAX_TOKENS = 1024
QWEN_TEMPERATURE = 0.3
QWEN_CONTEXT_LENGTH = 4096
QWEN_THREADS = 8  # Use 8 of 12 available threads

# ─── IMAP Defaults ───────────────────────────────────────────────────────────
IMAP_SERVERS = {
    "Gmail": "imap.gmail.com",
    "Outlook": "outlook.office365.com",
    "Yahoo": "imap.mail.yahoo.com",
}
DEFAULT_IMAP_SERVER = "Gmail"
IMAP_PORT = 993

# Folders to scan (Gmail IMAP names)
IMAP_FOLDERS_GMAIL = ["INBOX", "[Gmail]/Promotions", "[Gmail]/Updates"]
IMAP_FOLDERS_OUTLOOK = ["INBOX", "Promotions", "Updates"]
IMAP_FOLDERS_YAHOO = ["INBOX", "Bulk Mail"]

# ─── Delta Sync ──────────────────────────────────────────────────────────────
INITIAL_SYNC_DAYS = 7  # How far back on first sync (demo: 7 days)

# ─── Indian Bank Domain Mapping ──────────────────────────────────────────────
INDIAN_BANK_DOMAINS = {
    # HDFC
    "hdfcbank.net": "HDFC",
    "hdfcbank.com": "HDFC",
    "hdfcbankmail.com": "HDFC",
    # ICICI
    "icicibank.com": "ICICI",
    "icicibank.co.in": "ICICI",
    "iciciprulife.com": "ICICI",
    # Axis
    "axisbank.com": "Axis",
    "axisdirect.in": "Axis",
    # Kotak
    "kotak.com": "Kotak",
    "kotakbank.com": "Kotak",
    # SBI
    "sbi.co.in": "SBI",
    "onlinesbi.com": "SBI",
    "sbicard.com": "SBI",
    # PNB
    "pnb.co.in": "PNB",
    # Bank of Baroda
    "bankofbaroda.co.in": "BOB",
    "bobfinancial.com": "BOB",
    # Bajaj Finserv
    "bajajfinserv.in": "Bajaj Finserv",
    "bajajfinance.in": "Bajaj Finserv",
    # AMEX
    "aexp.com": "AMEX",
    "americanexpress.com": "AMEX",
    "americanexpress.co.in": "AMEX",
    # CRED
    "cred.club": "CRED",
    # IndusInd
    "indusind.com": "IndusInd",
    # Yes Bank
    "yesbank.in": "Yes Bank",
    # IDFC First
    "idfcfirstbank.com": "IDFC First",
    # RBL
    "rbl.bank": "RBL",
    # AU Small Finance
    "aubank.in": "AU Bank",
    # Federal Bank
    "federalbank.co.in": "Federal",
}

# ─── Email Classification Keywords ───────────────────────────────────────────
TRANSACTION_KEYWORDS = [
    "debited", "credited", "spent", "payment", "transaction",
    "₹", "inr", "rs.", "rs ", "purchase", "withdrawn", "transferred",
    "upi", "neft", "imps", "rtgs", "emi", "autopay",
]

OFFER_KEYWORDS = [
    "offer", "cashback", "reward", "discount", "deal", "voucher",
    "coupon", "bonus", "eligible", "exclusive", "limited time",
    "earn", "save", "flat", "% off", "extra",
]

STATEMENT_KEYWORDS = [
    "statement", "e-statement", "account summary", "monthly summary",
    "billing", "due date", "minimum due",
]

ALERT_KEYWORDS = [
    "alert", "otp", "login", "suspicious", "blocked", "security",
    "verify", "unauthorized", "fraud",
]

# ─── Visualization (Plotly Dark Warm Theme) ───────────────────────────────────
PLOT_DARK_BG = "#12100e"
PLOT_SURFACE = "#1e1a16"
PLOT_TEXT_COLOR = "#f0ebe4"
PLOT_GRID_COLOR = "#3a3228"
PLOT_ACCENT_COLORS = [
    "#F59E0B",  # Amber
    "#EF4444",  # Warm Red
    "#10B981",  # Emerald
    "#F97316",  # Orange
    "#A78BFA",  # Lavender
    "#EC4899",  # Pink
    "#FBBF24",  # Gold
    "#34D399",  # Mint
    "#FB923C",  # Light Orange
    "#F472B6",  # Rose
]

# ─── Vision Agent Settings ────────────────────────────────────────────────────
VISION_FALLBACK_MIN_BODY_LENGTH = 50  # Chars below which vision is triggered
BANNER_DOWNLOAD_TIMEOUT = 10  # Seconds

# ─── Gradio Theme Colors ─────────────────────────────────────────────────────
GRADIO_PRIMARY = "#F59E0B"
GRADIO_BG_DARK = "#12100e"
GRADIO_SURFACE_DARK = "#1e1a16"
GRADIO_TEXT = "#f0ebe4"


def format_inr(amount: float) -> str:
    """Format amount in Indian Rupee notation (e.g., ₹1,23,456.78)."""
    if amount < 0:
        return f"-₹{format_inr_abs(abs(amount))}"
    return f"₹{format_inr_abs(amount)}"


def format_inr_abs(amount: float) -> str:
    """Format absolute amount in Indian numbering system."""
    s = f"{amount:,.2f}"
    # Convert international format to Indian format
    parts = s.split(".")
    integer_part = parts[0].replace(",", "")
    decimal_part = parts[1] if len(parts) > 1 else "00"

    if len(integer_part) <= 3:
        return f"{integer_part}.{decimal_part}"

    # Last 3 digits
    last3 = integer_part[-3:]
    remaining = integer_part[:-3]

    # Group remaining in pairs from right
    groups = []
    while remaining:
        groups.append(remaining[-2:])
        remaining = remaining[:-2]
    groups.reverse()

    formatted = ",".join(groups) + "," + last3
    return f"{formatted}.{decimal_part}"
