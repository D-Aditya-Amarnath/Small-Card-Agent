"""
database.py — SQLite + FTS5 Database Layer for Banking Email Intelligence

Provides persistent storage for emails, transactions, offers, and sync metadata.
Supports full-text search via SQLite FTS5 for offline RAG retrieval.
"""

import sqlite3
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from contextlib import contextmanager

from config import DB_PATH, DATA_DIR, INITIAL_SYNC_DAYS

logger = logging.getLogger(__name__)


class BankingDatabase:
    """SQLite database with FTS5 full-text search for banking email storage."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with WAL mode."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Create all tables and FTS5 indexes if they don't exist."""
        with self._get_connection() as conn:
            # Core email storage
            conn.execute("""
                CREATE TABLE IF NOT EXISTS emails (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT UNIQUE NOT NULL,
                    sender TEXT NOT NULL,
                    sender_domain TEXT NOT NULL,
                    bank_name TEXT,
                    subject TEXT,
                    body_text TEXT,
                    body_html TEXT,
                    received_date DATETIME NOT NULL,
                    folder TEXT,
                    has_attachments BOOLEAN DEFAULT 0,
                    email_type TEXT,
                    vision_needed BOOLEAN DEFAULT 0,
                    synced_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # FTS5 virtual table for full-text search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS emails_fts USING fts5(
                    subject, body_text, bank_name,
                    content='emails', content_rowid='id'
                )
            """)

            # Triggers to keep FTS5 in sync with emails table
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS emails_fts_insert AFTER INSERT ON emails BEGIN
                    INSERT INTO emails_fts(rowid, subject, body_text, bank_name)
                    VALUES (new.id, new.subject, new.body_text, new.bank_name);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS emails_fts_delete AFTER DELETE ON emails BEGIN
                    INSERT INTO emails_fts(emails_fts, rowid, subject, body_text, bank_name)
                    VALUES ('delete', old.id, old.subject, old.body_text, old.bank_name);
                END
            """)

            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS emails_fts_update AFTER UPDATE ON emails BEGIN
                    INSERT INTO emails_fts(emails_fts, rowid, subject, body_text, bank_name)
                    VALUES ('delete', old.id, old.subject, old.body_text, old.bank_name);
                    INSERT INTO emails_fts(rowid, subject, body_text, bank_name)
                    VALUES (new.id, new.subject, new.body_text, new.bank_name);
                END
            """)

            # Parsed transactions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_id INTEGER REFERENCES emails(id) ON DELETE CASCADE,
                    bank_name TEXT,
                    amount REAL,
                    currency TEXT DEFAULT 'INR',
                    transaction_type TEXT,
                    category TEXT,
                    merchant TEXT,
                    card_last4 TEXT,
                    transaction_date DATETIME,
                    raw_text TEXT
                )
            """)

            # Promotional offers
            conn.execute("""
                CREATE TABLE IF NOT EXISTS offers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_id INTEGER REFERENCES emails(id) ON DELETE CASCADE,
                    bank_name TEXT,
                    offer_text TEXT,
                    source TEXT DEFAULT 'email_body',
                    category TEXT,
                    valid_until DATETIME,
                    extracted_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Sync log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sync_started DATETIME,
                    sync_completed DATETIME,
                    emails_fetched INTEGER DEFAULT 0,
                    folder TEXT,
                    status TEXT DEFAULT 'started'
                )
            """)

            # Indexes for frequent queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_emails_received
                ON emails(received_date DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_emails_bank
                ON emails(bank_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_emails_type
                ON emails(email_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_transactions_date
                ON transactions(transaction_date DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_transactions_bank
                ON transactions(bank_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_offers_bank
                ON offers(bank_name)
            """)

        logger.info(f"Database initialized at {self.db_path}")

    # ─── Delta Sync Methods ──────────────────────────────────────────────────

    def get_latest_sync_date(self) -> Optional[datetime]:
        """Get the latest received_date from the emails table for delta sync."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT MAX(received_date) as latest FROM emails"
            ).fetchone()
            if row and row["latest"]:
                try:
                    return datetime.fromisoformat(row["latest"])
                except (ValueError, TypeError):
                    # Try parsing other date formats
                    try:
                        return datetime.strptime(row["latest"], "%Y-%m-%d %H:%M:%S")
                    except (ValueError, TypeError):
                        return None
            return None

    def get_initial_sync_date(self) -> datetime:
        """Get the date to start from on first sync (90 days ago)."""
        return datetime.now() - timedelta(days=INITIAL_SYNC_DAYS)

    def get_sync_since_date(self) -> datetime:
        """Get the date to sync from — latest stored date or 90 days ago."""
        latest = self.get_latest_sync_date()
        if latest:
            return latest
        return self.get_initial_sync_date()

    # ─── Email CRUD ──────────────────────────────────────────────────────────

    def insert_email(self, email_data: Dict[str, Any]) -> Optional[int]:
        """
        Insert or ignore an email record. Returns the row ID if inserted, None if duplicate.
        email_data keys: message_id, sender, sender_domain, bank_name, subject,
                         body_text, body_html, received_date, folder, has_attachments,
                         email_type, vision_needed
        """
        with self._get_connection() as conn:
            try:
                cursor = conn.execute("""
                    INSERT OR IGNORE INTO emails
                    (message_id, sender, sender_domain, bank_name, subject,
                     body_text, body_html, received_date, folder, has_attachments,
                     email_type, vision_needed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    email_data.get("message_id"),
                    email_data.get("sender"),
                    email_data.get("sender_domain"),
                    email_data.get("bank_name"),
                    email_data.get("subject"),
                    email_data.get("body_text"),
                    email_data.get("body_html"),
                    email_data.get("received_date"),
                    email_data.get("folder"),
                    email_data.get("has_attachments", False),
                    email_data.get("email_type"),
                    email_data.get("vision_needed", False),
                ))
                if cursor.rowcount > 0:
                    return cursor.lastrowid
                return None
            except sqlite3.IntegrityError:
                logger.debug(f"Duplicate email: {email_data.get('message_id')}")
                return None

    def get_email_by_id(self, email_id: int) -> Optional[Dict]:
        """Get a single email record by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM emails WHERE id = ?", (email_id,)).fetchone()
            return dict(row) if row else None

    def get_emails_needing_vision(self) -> List[Dict]:
        """Get emails flagged for vision processing."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM emails WHERE vision_needed = 1 ORDER BY received_date DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_vision_processed(self, email_id: int):
        """Mark an email as no longer needing vision processing."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE emails SET vision_needed = 0 WHERE id = ?", (email_id,)
            )

    def update_email_type(self, email_id: int, email_type: str):
        """Update the classified type of an email."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE emails SET email_type = ? WHERE id = ?",
                (email_type, email_id)
            )

    def get_recent_emails(self, limit: int = 50) -> List[Dict]:
        """Get recent emails ordered by received date."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM emails ORDER BY received_date DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_email_count(self) -> int:
        """Get total number of stored emails."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM emails").fetchone()
            return row["cnt"] if row else 0

    # ─── Transaction CRUD ────────────────────────────────────────────────────

    def insert_transaction(self, txn_data: Dict[str, Any]) -> int:
        """Insert a parsed transaction record. Returns row ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO transactions
                (email_id, bank_name, amount, currency, transaction_type,
                 category, merchant, card_last4, transaction_date, raw_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                txn_data.get("email_id"),
                txn_data.get("bank_name"),
                txn_data.get("amount"),
                txn_data.get("currency", "INR"),
                txn_data.get("transaction_type"),
                txn_data.get("category"),
                txn_data.get("merchant"),
                txn_data.get("card_last4"),
                txn_data.get("transaction_date"),
                txn_data.get("raw_text"),
            ))
            return cursor.lastrowid

    def get_transactions(
        self,
        bank_name: Optional[str] = None,
        transaction_type: Optional[str] = None,
        days: Optional[int] = None,
        limit: int = 200,
    ) -> List[Dict]:
        """Get transactions with optional filters, including email subject and received_date."""
        query = """
            SELECT t.*, e.subject AS email_subject, e.received_date AS email_received_date
            FROM transactions t
            LEFT JOIN emails e ON t.email_id = e.id
            WHERE 1=1
        """
        params: List[Any] = []

        if bank_name:
            query += " AND t.bank_name = ?"
            params.append(bank_name)
        if transaction_type:
            query += " AND t.transaction_type = ?"
            params.append(transaction_type)
        if days:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            query += " AND COALESCE(e.received_date, t.transaction_date) >= ?"
            params.append(cutoff)

        query += " ORDER BY COALESCE(e.received_date, t.transaction_date) DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_transaction_count(self) -> int:
        """Get total number of parsed transactions."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM transactions").fetchone()
            return row["cnt"] if row else 0

    def get_spending_summary(self) -> Dict[str, Any]:
        """Get aggregated spending data for visualization."""
        with self._get_connection() as conn:
            # By category
            by_category = conn.execute("""
                SELECT category, SUM(amount) as total, COUNT(*) as count
                FROM transactions
                WHERE transaction_type = 'debit' AND category IS NOT NULL
                GROUP BY category
                ORDER BY total DESC
            """).fetchall()

            # By bank
            by_bank = conn.execute("""
                SELECT bank_name, SUM(amount) as total, COUNT(*) as count
                FROM transactions
                WHERE transaction_type = 'debit' AND bank_name IS NOT NULL
                GROUP BY bank_name
                ORDER BY total DESC
            """).fetchall()

            # Daywise trend — use email received_date (reliable ISO) for grouping
            # Use substr() as fallback since strftime cannot parse timezone offsets
            daywise = conn.execute("""
                SELECT COALESCE(
                    date(COALESCE(e.received_date, t.transaction_date)),
                    substr(COALESCE(e.received_date, t.transaction_date), 1, 10),
                    'Unknown'
                ) as day,
                SUM(CASE WHEN t.transaction_type='debit' THEN t.amount ELSE 0 END) as debits,
                SUM(CASE WHEN t.transaction_type='credit' THEN t.amount ELSE 0 END) as credits,
                COUNT(*) as count
                FROM transactions t
                LEFT JOIN emails e ON t.email_id = e.id
                GROUP BY day
                HAVING day != 'Unknown' AND day IS NOT NULL AND length(day) = 10
                ORDER BY day
            """).fetchall()

            # Totals
            totals = conn.execute("""
                SELECT
                    SUM(CASE WHEN transaction_type='debit' THEN amount ELSE 0 END) as total_debits,
                    SUM(CASE WHEN transaction_type='credit' THEN amount ELSE 0 END) as total_credits,
                    COUNT(*) as total_count
                FROM transactions
            """).fetchone()

            return {
                "by_category": [dict(r) for r in by_category],
                "by_bank": [dict(r) for r in by_bank],
                "daywise": [dict(r) for r in daywise],
                "total_debits": totals["total_debits"] or 0 if totals else 0,
                "total_credits": totals["total_credits"] or 0 if totals else 0,
                "total_count": totals["total_count"] or 0 if totals else 0,
            }

    # ─── Offer CRUD ──────────────────────────────────────────────────────────

    def insert_offer(self, offer_data: Dict[str, Any]) -> int:
        """Insert an extracted offer record. Returns row ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO offers
                (email_id, bank_name, offer_text, source, category, valid_until)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                offer_data.get("email_id"),
                offer_data.get("bank_name"),
                offer_data.get("offer_text"),
                offer_data.get("source", "email_body"),
                offer_data.get("category"),
                offer_data.get("valid_until"),
            ))
            return cursor.lastrowid

    def get_offers(self, bank_name: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get all cached offers with optional bank filter."""
        query = "SELECT o.*, e.subject as email_subject, e.body_text as email_body, e.sender as email_sender, e.received_date as email_date FROM offers o LEFT JOIN emails e ON o.email_id = e.id WHERE 1=1"
        params: List[Any] = []

        if bank_name:
            query += " AND o.bank_name = ?"
            params.append(bank_name)

        query += " ORDER BY o.extracted_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_offer_count(self) -> int:
        """Get total number of extracted offers."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM offers").fetchone()
            return row["cnt"] if row else 0

    # ─── FTS5 Search ─────────────────────────────────────────────────────────

    def search_fts(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Full-text search across emails using FTS5.
        Returns matching email records with relevance ranking.
        """
        with self._get_connection() as conn:
            # Use FTS5 MATCH with BM25 ranking
            rows = conn.execute("""
                SELECT e.*, rank
                FROM emails_fts fts
                JOIN emails e ON e.id = fts.rowid
                WHERE emails_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit)).fetchall()
            return [dict(r) for r in rows]

    def search_offers_fts(self, query: str, limit: int = 20) -> List[Dict]:
        """Search emails classified as offers using FTS5."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT e.*, rank
                FROM emails_fts fts
                JOIN emails e ON e.id = fts.rowid
                WHERE emails_fts MATCH ?
                AND e.email_type = 'offer'
                ORDER BY rank
                LIMIT ?
            """, (query, limit)).fetchall()
            return [dict(r) for r in rows]

    # ─── Sync Log ────────────────────────────────────────────────────────────

    def start_sync_log(self, folder: str) -> int:
        """Start a sync log entry. Returns the log ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO sync_log (sync_started, folder, status)
                VALUES (?, ?, 'in_progress')
            """, (datetime.now().isoformat(), folder))
            return cursor.lastrowid

    def complete_sync_log(self, log_id: int, emails_fetched: int, status: str = "success"):
        """Complete a sync log entry."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE sync_log
                SET sync_completed = ?, emails_fetched = ?, status = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), emails_fetched, status, log_id))

    def get_sync_stats(self) -> Dict[str, Any]:
        """Get sync statistics."""
        with self._get_connection() as conn:
            total_syncs = conn.execute("SELECT COUNT(*) as cnt FROM sync_log").fetchone()
            total_emails = conn.execute("SELECT SUM(emails_fetched) as total FROM sync_log WHERE status='success'").fetchone()
            last_sync = conn.execute(
                "SELECT * FROM sync_log ORDER BY sync_started DESC LIMIT 1"
            ).fetchone()

            return {
                "total_syncs": total_syncs["cnt"] if total_syncs else 0,
                "total_emails_fetched": total_emails["total"] or 0 if total_emails else 0,
                "last_sync": dict(last_sync) if last_sync else None,
            }

    # ─── Dashboard Stats ─────────────────────────────────────────────────────

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get overall dashboard statistics."""
        return {
            "email_count": self.get_email_count(),
            "transaction_count": self.get_transaction_count(),
            "offer_count": self.get_offer_count(),
            "sync_stats": self.get_sync_stats(),
            "latest_sync_date": self.get_latest_sync_date(),
        }

    # ─── Vision Agent Support ────────────────────────────────────────────────

    def append_body_text(self, email_id: int, extracted_text: str):
        """
        Append vision-extracted text to an email's body_text field.
        Used by the vision agent to store OCR results so the classifier
        can process them naturally.
        """
        with self._get_connection() as conn:
            current = conn.execute(
                "SELECT body_text FROM emails WHERE id = ?", (email_id,)
            ).fetchone()
            if current:
                existing = current["body_text"] or ""
                updated = existing + "\n[VISION EXTRACTED]\n" + extracted_text
                conn.execute(
                    "UPDATE emails SET body_text = ? WHERE id = ?",
                    (updated, email_id)
                )
                logger.info(f"Appended vision text to email {email_id} ({len(extracted_text)} chars)")

    # ─── Demo: Clear All Data ────────────────────────────────────────────────

    def clear_all_data(self):
        """
        Truncate all tables for demo cleanup on exit.
        Keeps schema intact but removes all data.
        """
        with self._get_connection() as conn:
            conn.execute("DELETE FROM offers")
            conn.execute("DELETE FROM transactions")
            conn.execute("DELETE FROM sync_log")
            conn.execute("DELETE FROM emails")
            # Rebuild FTS5 index
            conn.execute("INSERT INTO emails_fts(emails_fts) VALUES('rebuild')")
            logger.info("All database tables cleared (demo cleanup)")

