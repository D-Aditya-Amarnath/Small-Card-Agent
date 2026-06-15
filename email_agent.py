"""
email_agent.py — IMAP Email Agent with Delta Sync & Indian Bank Filtering

Connects to IMAP, fetches only new banking emails since last sync,
extracts metadata and body text, stores in SQLite, and flags emails
needing vision processing.
"""

import email
import email.utils
import re
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Generator
from email.header import decode_header
from bs4 import BeautifulSoup

from imap_tools import MailBox, AND, MailMessage

from database import BankingDatabase
from config import (
    IMAP_SERVERS, IMAP_PORT, INITIAL_SYNC_DAYS,
    IMAP_FOLDERS_GMAIL, IMAP_FOLDERS_OUTLOOK, IMAP_FOLDERS_YAHOO,
    INDIAN_BANK_DOMAINS, VISION_FALLBACK_MIN_BODY_LENGTH,
)

logger = logging.getLogger(__name__)


class EmailAgent:
    """
    IMAP-based email agent that performs delta sync of Indian banking emails.
    
    Security: Credentials are passed as function arguments and never stored.
    After sync completes, all credential references are cleared.
    """

    def __init__(self, db: BankingDatabase):
        self.db = db

    def _get_folders_for_server(self, server_name: str) -> List[str]:
        """Get the appropriate IMAP folder list for the selected server."""
        if server_name == "Gmail":
            return IMAP_FOLDERS_GMAIL
        elif server_name == "Outlook":
            return IMAP_FOLDERS_OUTLOOK
        elif server_name == "Yahoo":
            return IMAP_FOLDERS_YAHOO
        else:
            return ["INBOX"]

    def _extract_domain(self, sender: str) -> str:
        """Extract domain from email address."""
        match = re.search(r'@([\w.-]+)', sender)
        return match.group(1).lower() if match else ""

    def _identify_bank(self, sender_domain: str) -> Optional[str]:
        """Map sender domain to Indian bank name. Returns None if not a bank."""
        # Direct domain match
        if sender_domain in INDIAN_BANK_DOMAINS:
            return INDIAN_BANK_DOMAINS[sender_domain]

        # RBI Mandate: Handle .bank.in
        if sender_domain.endswith(".bank.in"):
            bank_prefix = sender_domain.split(".bank.in")[0].split(".")[-1].title()
            return f"{bank_prefix} Bank"

        # Subdomain match (e.g., mail.hdfcbank.net → HDFC)
        for domain, bank in INDIAN_BANK_DOMAINS.items():
            if sender_domain.endswith("." + domain) or sender_domain == domain:
                return bank

        return None

    def _extract_image_urls(self, html: str) -> List[str]:
        """Extract image URLs from HTML email body for vision agent."""
        if not html:
            return []
        try:
            soup = BeautifulSoup(html, "html.parser")
            urls = []
            for img in soup.find_all("img"):
                src = img.get("src", "")
                if src and src.startswith("http"):
                    # Filter out tracking pixels (tiny images)
                    width = img.get("width", "")
                    height = img.get("height", "")
                    # Skip 1x1 tracking pixels
                    if width in ("1", "0") or height in ("1", "0"):
                        continue
                    urls.append(src)
            return urls
        except Exception as e:
            logger.warning(f"Failed to parse HTML for images: {e}")
            return []

    def _clean_text(self, text: str) -> str:
        """Clean and normalize email text."""
        if not text:
            return ""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove common email footer patterns
        text = re.sub(r'(This (email|message) (is|was|has been).*$)', '', text, flags=re.IGNORECASE | re.DOTALL)
        return text[:10000]  # Cap at 10K chars

    def _determine_vision_needed(self, body_text: str, body_html: str) -> bool:
        """
        Check if vision agent fallback is needed.
        Vision is needed when body text is empty/insufficient AND HTML has images.
        """
        text_length = len(body_text.strip()) if body_text else 0
        has_images = bool(self._extract_image_urls(body_html)) if body_html else False
        return text_length < VISION_FALLBACK_MIN_BODY_LENGTH and has_images

    def sync_emails(
        self,
        email_address: str,
        app_password: str,
        server_name: str = "Gmail",
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Perform delta sync: fetch only new banking emails since last sync.
        
        Args:
            email_address: User's email address
            app_password: App-specific password (not regular password)
            server_name: IMAP server name key from config
            progress_callback: Optional callable(message: str) for progress updates
        
        Returns:
            Dict with sync results: {emails_fetched, folders_synced, errors, sync_id}
        """
        imap_server = IMAP_SERVERS.get(server_name, IMAP_SERVERS["Gmail"])
        folders = self._get_folders_for_server(server_name)
        since_date = self.db.get_sync_since_date()

        results = {
            "emails_fetched": 0,
            "emails_skipped": 0,
            "folders_synced": [],
            "banks_found": set(),
            "errors": [],
            "vision_flagged": 0,
            "new_email_ids": [],
        }

        def log(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        log(f"Starting delta sync from {since_date.strftime('%Y-%m-%d')}...")
        log(f"Connecting to {imap_server}...")

        try:
            # Add a 15-second timeout. HF Spaces blocks port 993, causing infinite hangs otherwise.
            with MailBox(imap_server, IMAP_PORT, timeout=15).login(
                email_address, app_password
            ) as mailbox:
                log(f"[OK] Connected successfully to {imap_server}")

                for folder in folders:
                    sync_log_id = self.db.start_sync_log(folder)
                    folder_count = 0

                    try:
                        log(f"Scanning folder: {folder}")
                        mailbox.folder.set(folder)

                        # Delta sync: only fetch emails since last sync date
                        scanned_count = 0
                        for msg in mailbox.fetch(
                            AND(date_gte=since_date.date()),
                            mark_seen=False,
                            bulk=True,
                        ):
                            scanned_count += 1
                            if scanned_count % 25 == 0:
                                log(f"  Scanning {folder}... ({scanned_count} emails scanned so far)")
                            try:
                                sender = msg.from_ or ""
                                sender_domain = self._extract_domain(sender)
                                bank_name = self._identify_bank(sender_domain)

                                if not bank_name:
                                    results["emails_skipped"] += 1
                                    continue

                                # Extract email content
                                raw_text = msg.text or ""
                                body_html = msg.html or ""
                                if not raw_text.strip() and body_html:
                                    try:
                                        soup = BeautifulSoup(body_html, "html.parser")
                                        raw_text = soup.get_text(separator=" ", strip=True)
                                    except Exception:
                                        pass
                                body_text = self._clean_text(raw_text)
                                msg_date = msg.date
                                if msg_date:
                                    # Strip timezone info — SQLite strftime only
                                    # works with naive ISO format (YYYY-MM-DD HH:MM:SS)
                                    if hasattr(msg_date, 'replace'):
                                        msg_date_naive = msg_date.replace(tzinfo=None)
                                    else:
                                        msg_date_naive = msg_date
                                    received_date = msg_date_naive.isoformat()
                                else:
                                    received_date = datetime.now().isoformat()

                                # Determine if vision fallback is needed
                                vision_needed = self._determine_vision_needed(body_text, body_html)

                                # Build email record
                                email_data = {
                                    "message_id": msg.uid or f"{sender}_{received_date}",
                                    "sender": sender,
                                    "sender_domain": sender_domain,
                                    "bank_name": bank_name,
                                    "subject": msg.subject or "(No Subject)",
                                    "body_text": body_text,
                                    "body_html": body_html,
                                    "received_date": received_date,
                                    "folder": folder,
                                    "has_attachments": len(msg.attachments) > 0,
                                    "email_type": None,  # Will be classified by Classifier Agent
                                    "vision_needed": vision_needed,
                                }

                                # Insert into database (will skip duplicates via UNIQUE constraint)
                                email_id = self.db.insert_email(email_data)
                                if email_id:
                                    folder_count += 1
                                    results["emails_fetched"] += 1
                                    results["banks_found"].add(bank_name)
                                    results["new_email_ids"].append(email_id)
                                    if vision_needed:
                                        results["vision_flagged"] += 1
                                    log(f"  [{bank_name}] {msg.subject[:60]}...")

                            except Exception as e:
                                logger.error(f"Error processing email: {e}")
                                results["errors"].append(str(e))

                        self.db.complete_sync_log(sync_log_id, folder_count, "success")
                        results["folders_synced"].append(folder)
                        log(f"Finished {folder}: Scanned {scanned_count} total, saved {folder_count} new banking emails.")

                    except Exception as e:
                        error_msg = str(e)
                        if "NONEXISTENT" in error_msg.upper():
                            log(f"  [WARN] Folder '{folder}' does not exist on this account, skipping.")
                            self.db.complete_sync_log(sync_log_id, folder_count, "skipped")
                        else:
                            full_error_msg = f"Error scanning {folder}: {e}"
                            logger.error(full_error_msg)
                            results["errors"].append(full_error_msg)
                            self.db.complete_sync_log(sync_log_id, folder_count, "failed")
                            log(f"  [ERROR] {folder}: {full_error_msg}")

        except Exception as e:
            error_msg = f"IMAP connection failed: {e}"
            if "timeout" in str(e).lower():
                error_msg += " (Note: Hugging Face Spaces free tier blocks outgoing IMAP connections on port 993 to prevent spam. Please test Network Sync by running the app locally.)"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            log(f"[ERROR] {error_msg}")

        # Convert set to list for JSON serialization
        results["banks_found"] = list(results["banks_found"])

        log(f"\nSync complete: {results['emails_fetched']} new emails from {len(results['banks_found'])} banks")
        if results["vision_flagged"]:
            log(f"{results['vision_flagged']} emails flagged for vision processing")

        # SECURITY: Clear credentials from local scope
        email_address = None
        app_password = None

        return results

    def get_image_urls_for_email(self, email_id: int) -> List[str]:
        """Get banner image URLs from an email's HTML for vision processing."""
        email_data = self.db.get_email_by_id(email_id)
        if not email_data or not email_data.get("body_html"):
            return []
        return self._extract_image_urls(email_data["body_html"])
