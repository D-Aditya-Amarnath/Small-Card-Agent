"""
vision_agent.py — Vision Agent (Fallback Only) — OCR Text Extraction Only

Supports 3 backends:
  1. LMStudio (Local) — OpenAI Vision API with base64-encoded images
  2. ZeroGPU (HF Spaces) — Moondream2 via transformers + GPU
  3. Transformers (CPU) — direct Moondream2 loading (slow fallback)

Prompt Strategy:
  Pure OCR — Extract ALL visible text from the banner image.
  No classification or offer parsing is done here.
  The extracted text is written back to the email's body_text field
  so the Classifier Agent (Qwen) can handle classification in Phase 3.

Only invoked when an email's body text is insufficient (< 50 chars)
and the HTML contains banner image URLs. Downloads images from URLs
and uses vision model to extract text from promotional banners.
"""

import gc
import base64
import logging
import requests
import io
from typing import Dict, List, Optional
from PIL import Image

from database import BankingDatabase
from config import (
    BANNER_DOWNLOAD_TIMEOUT, BANNERS_DIR,
    MOONDREAM_HF_MODEL_ID, MOONDREAM_HF_REVISION,
    LMSTUDIO_BASE_URL, LMSTUDIO_VISION_MODEL,
    BACKEND_LMSTUDIO, BACKEND_ZEROGPU, BACKEND_GGUF,
    DEFAULT_BACKEND, MODELS_DIR
)

logger = logging.getLogger(__name__)

# ─── Try importing optional dependencies ─────────────────────────────────────
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ══════════════════════════════════════════════════════════════════════════════
# VISION PROMPT TEMPLATES — Multi-Pass Strategy
# ══════════════════════════════════════════════════════════════════════════════

# OCR Extraction — get ALL text from the image using RECAT Strategy
VISION_OCR_PROMPT = """**Role**: You are a highly accurate Optical Character Recognition (OCR) assistant.
**Expertise**: You specialize in analyzing Indian banking and financial images, including transaction alerts, e-statements, and promotional banners.
**Context**: You are analyzing an image extracted from a banking email. The image may contain critical transaction details (money sent/received) OR promotional offers. 
**Action**: Extract ALL text you can see in the image exactly as written. Pay extremely close attention to:
1. Transaction details: Amounts (Rs./₹/INR), "debited from" / "credited to", merchant names, and transaction IDs/references.
2. Account details: Last 4 digits of cards or accounts.
3. Promotional details: Percentages (%), flat discounts, promo codes.
4. Dates and timestamps.
**Tone**: Be purely objective, accurate, and precise. Do not summarize or interpret; simply transcribe the text clearly and preserve the visual hierarchy."""

# LMStudio vision system prompt (OCR-focused)
VISION_SYSTEM_PROMPT = """You are a highly accurate OCR assistant specializing in Indian banking.
Extract all visible text accurately without adding commentary. Pay extremely close attention to transaction amounts, currency symbols (Rs, INR, ₹), merchant names, card numbers, percentages, and dates."""


class VisionAgent:
    """
    Vision agent for extracting text from promotional banners (OCR only).
    
    FALLBACK ONLY: Only used when email body text lacks useful information
    and the email HTML contains image URLs to promotional banners.
    
    This agent ONLY extracts raw text from images. It does NOT classify
    or categorize the content. The extracted text is appended to the
    email's body_text field so the Classifier Agent (Qwen) can process it.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._backend = DEFAULT_BACKEND
        self._openai_client = None
        self._resolved_vision_model = LMSTUDIO_VISION_MODEL  # Will be updated to actual LMStudio model ID

    # ─── Backend Management ──────────────────────────────────────────────────

    def set_backend(self, backend_name: str):
        """Switch inference backend. Unloads any previously loaded model."""
        if backend_name == self._backend and self._loaded:
            return
        self.unload_model()
        self._backend = backend_name
        logger.info(f"Vision backend set to: {backend_name}")

    def get_backend(self) -> str:
        return self._backend

    # ─── Model Loading ───────────────────────────────────────────────────────

    def load_model(self, progress_callback=None):
        """Load vision model based on current backend."""
        if self._loaded:
            return True

        def log(msg):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        if self._backend == BACKEND_LMSTUDIO:
            return self._load_lmstudio(log)
        elif self._backend == BACKEND_ZEROGPU:
            return self._load_zerogpu(log)
        else:
            return self._load_llama_cpp(log)

    def _load_lmstudio(self, log) -> bool:
        """Initialize OpenAI client for LMStudio vision model."""
        if not HAS_OPENAI:
            log("[ERROR] `openai` package not installed for LMStudio vision")
            return False
        try:
            self._openai_client = OpenAI(
                base_url=LMSTUDIO_BASE_URL,
                api_key="lm-studio",
            )
            models = self._openai_client.models.list()
            model_ids = [m.id for m in models.data]
            
            # Check if vision model is loaded in LMStudio
            # Find the actual full model ID that contains our search term
            matched_model = None
            for m in model_ids:
                if LMSTUDIO_VISION_MODEL.lower() in m.lower():
                    matched_model = m
                    break
            
            if not matched_model and model_ids:
                log(f"[WARN] Vision model containing '{LMSTUDIO_VISION_MODEL}' not loaded in LMStudio.")
                log(f"       Loaded models: {', '.join(model_ids)}")
                log("       Please load the vision model in LMStudio to enable banner extraction.")
                self._loaded = False
                return False

            # Store the resolved full model ID for API calls
            self._resolved_vision_model = matched_model
            self._loaded = True
            log(f"[OK] Vision: Connected to LMStudio")
            log(f"  Model: {self._resolved_vision_model}")
            return True
        except Exception as e:
            log(f"[ERROR] LMStudio vision connection failed: {e}")
            self._loaded = False
            return False

    def _load_zerogpu(self, log) -> bool:
        """Load Moondream2 for ZeroGPU."""
        log("Loading Moondream2 vision model (ZeroGPU)...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            log(f"  Model: {MOONDREAM_HF_MODEL_ID} (rev: {MOONDREAM_HF_REVISION})")

            self.model = AutoModelForCausalLM.from_pretrained(
                MOONDREAM_HF_MODEL_ID,
                revision=MOONDREAM_HF_REVISION,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                MOONDREAM_HF_MODEL_ID,
                revision=MOONDREAM_HF_REVISION,
                trust_remote_code=True,
            )
            self.model.eval()
            self._loaded = True
            log("[OK] Moondream2 loaded on ZeroGPU")
            return True
        except Exception as e:
            log(f"[ERROR] Vision model loading failed: {e}")
            self._loaded = False
            return False

    def _load_llama_cpp(self, log) -> bool:
        """Load Moondream2 natively using llama-cpp-python from local GGUF models."""
        log("Loading Moondream2 vision model (Local GGUF)...")
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            import os

            text_model = os.path.join(MODELS_DIR, "moondream2-text-model-q4_k_m.gguf")
            mmproj = os.path.join(MODELS_DIR, "moondream2-mmproj-f16.gguf")

            if not os.path.exists(text_model) or not os.path.exists(mmproj):
                log("[ERROR] Vision GGUF models not found in models/ directory.")
                self._loaded = False
                return False

            log(f"  Text model: {os.path.basename(text_model)}")
            log(f"  Vision proj: {os.path.basename(mmproj)}")

            chat_handler = Llava15ChatHandler(clip_model_path=mmproj)
            self.model = Llama(
                model_path=text_model,
                chat_handler=chat_handler,
                n_ctx=2048,
                verbose=False
            )
            self._loaded = True
            log("[OK] Moondream2 loaded natively via llama.cpp")
            return True
        except Exception as e:
            logger.error(f"Failed to load Moondream2 GGUF: {e}")
            log(f"[ERROR] Vision model loading failed: {e}")
            log("[WARN] Banner extraction will be skipped")
            self._loaded = False
            return False

    def unload_model(self):
        """Unload model to free memory for other agents."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._openai_client = None
        self._loaded = False
        gc.collect()
        logger.info("Vision model unloaded, memory freed")

    # ─── Image Utilities ─────────────────────────────────────────────────────

    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download an image from a URL. Returns PIL Image or None."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            }
            response = requests.get(
                url, headers=headers, timeout=BANNER_DOWNLOAD_TIMEOUT, stream=True
            )
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                logger.debug(f"Skipping non-image URL: {url}")
                return None

            # Check size (skip tiny tracking pixels and huge files)
            content_length = int(response.headers.get("Content-Length", 0))
            if content_length > 0 and content_length < 1000:
                logger.debug(f"Skipping tiny image (likely tracker): {url}")
                return None
            if content_length > 10_000_000:
                logger.debug(f"Skipping oversized image: {url}")
                return None

            img = Image.open(io.BytesIO(response.content))
            img = img.convert("RGB")

            # Skip very small images (tracking pixels)
            if img.width < 50 or img.height < 50:
                logger.debug(f"Skipping tiny image {img.width}x{img.height}: {url}")
                return None

            return img

        except Exception as e:
            logger.debug(f"Failed to download image from {url}: {e}")
            return None

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URI for LMStudio vision API."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    # ─── Vision Inference ────────────────────────────────────────────────────

    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        OCR text extraction from a banner image.
        Returns raw extracted text without any classification.
        """
        if not self._loaded:
            return ""

        if self._backend == BACKEND_LMSTUDIO:
            return self._extract_lmstudio_ocr(image)
        elif self._backend == BACKEND_ZEROGPU:
            return self._extract_moondream(image)
        else:
            return self._extract_llama_cpp(image)

    def _extract_lmstudio_ocr(self, image: Image.Image) -> str:
        """
        OCR-only extraction via LMStudio's OpenAI Vision API.
        Extracts all visible text from the image without classification.
        """
        if not self._openai_client:
            return ""

        image_b64 = self._image_to_base64(image)

        try:
            ocr_response = self._openai_client.chat.completions.create(
                model=self._resolved_vision_model,
                messages=[
                    {"role": "system", "content": VISION_SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": VISION_OCR_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_b64}},
                    ]},
                ],
                max_tokens=500,
                temperature=0.1,
            )
            ocr_text = ocr_response.choices[0].message.content.strip()
            logger.info(f"  Vision OCR: {len(ocr_text)} chars extracted")
            return ocr_text if ocr_text and len(ocr_text) >= 10 else ""

        except Exception as e:
            logger.warning(f"LMStudio vision OCR failed: {e}")
            return ""

    def _extract_moondream(self, image: Image.Image) -> str:
        """
        OCR text extraction using Moondream2 (transformers).
        Extracts all visible text without classification.
        """
        if not self._loaded or self.model is None:
            return ""

        try:
            enc_image = self.model.encode_image(image)

            # Pure OCR prompt
            result = self.model.answer_question(
                enc_image,
                VISION_OCR_PROMPT,
                self.tokenizer,
            )
            text = result.strip() if result else ""

            logger.info(f"  Vision OCR: {len(text)} chars extracted via Moondream")
            return text if len(text) >= 10 else ""

        except Exception as e:
            logger.warning(f"Moondream vision OCR failed: {e}")
            return ""

    def _extract_llama_cpp(self, image: Image.Image) -> str:
        """
        OCR text extraction using llama-cpp-python natively.
        """
        if not self._loaded or self.model is None:
            return ""

        try:
            image_b64 = self._image_to_base64(image)
            response = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": VISION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_OCR_PROMPT},
                            {"type": "image_url", "image_url": {"url": image_b64}},
                        ],
                    }
                ],
                max_tokens=500,
                temperature=0.1,
            )
            text = response["choices"][0]["message"]["content"].strip()
            logger.info(f"  Vision OCR: {len(text)} chars extracted via llama.cpp")
            return text if len(text) >= 10 else ""
        except Exception as e:
            logger.warning(f"Llama.cpp vision OCR failed: {e}")
            return ""

            # If too short, try a simpler OCR prompt
            if not text or len(text) < 15:
                result2 = self.model.answer_question(
                    enc_image,
                    "Read all text visible in this image. List every word and number you can see.",
                    self.tokenizer,
                )
                if result2 and len(result2.strip()) > len(text):
                    text = result2.strip()

            return text

        except Exception as e:
            logger.warning(f"Moondream2 inference failed: {e}")
            return ""

    # ─── Banner Processing Pipeline ──────────────────────────────────────────

    def process_email_banners(
        self,
        db: BankingDatabase,
        email_id: int,
        image_urls: List[str],
        bank_name: str,
    ) -> Dict:
        """
        Download and process banner images from an email.
        OCR-only: extracts text and appends to email body_text.
        Classification is deferred to the Classifier Agent (Qwen).
        """
        all_extracted_text = []

        for url in image_urls[:5]:  # Limit to 5 images per email
            logger.info(f"  Vision processing banner: {url[:80]}...")

            image = self._download_image(url)
            if not image:
                continue

            text = self.extract_text_from_image(image)
            if text and len(text) > 10:
                all_extracted_text.append(text[:500])
                logger.info(f"  OCR extracted: {text[:100]}...")

        # Append all extracted text back to the email's body_text
        if all_extracted_text:
            combined_text = "\n".join(all_extracted_text)
            db.append_body_text(email_id, combined_text)

        # Mark email as vision-processed
        db.mark_vision_processed(email_id)
        return {"texts_extracted": len(all_extracted_text)}

    def process_pending_emails(
        self,
        db: BankingDatabase,
        email_agent,
        progress_callback=None,
    ) -> Dict[str, int]:
        """
        Process all emails flagged for vision processing.
        OCR-only: extracts text and appends to email body_text.
        Classification is handled by the Classifier Agent in Phase 3.
        """
        results = {"processed": 0, "texts_extracted": 0, "skipped": 0}

        def log(msg):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        # Get emails flagged for vision
        pending = db.get_emails_needing_vision()

        if not pending:
            log("Vision: No emails need vision processing")
            return results

        log(f"Vision: {len(pending)} emails need OCR processing...")

        # Load model on-demand
        if not self.load_model(progress_callback):
            log("Vision model not available, skipping banner extraction")
            results["skipped"] = len(pending)
            return results

        try:
            for email_data in pending:
                email_id = email_data["id"]
                bank_name = email_data.get("bank_name", "Unknown")

                # Get image URLs from email HTML
                image_urls = email_agent.get_image_urls_for_email(email_id)

                if not image_urls:
                    db.mark_vision_processed(email_id)
                    results["skipped"] += 1
                    continue

                result = self.process_email_banners(db, email_id, image_urls, bank_name)
                results["processed"] += 1
                results["texts_extracted"] += result.get("texts_extracted", 0)

                if result.get("texts_extracted", 0) > 0:
                    log(f"  [{bank_name}] {result['texts_extracted']} texts OCR'd from banners")
                else:
                    log(f"  [{bank_name}] No text found in banners")

        finally:
            # Always unload to free memory for Classifier Agent
            self.unload_model()
            log("Vision model unloaded, memory freed")

        log(f"\nVision processing: {results['processed']} emails, "
            f"{results['texts_extracted']} texts extracted (classification deferred to Qwen)")

        return results
