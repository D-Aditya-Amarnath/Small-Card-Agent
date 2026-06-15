"""
app.py -- Gradio UI Orchestrator for Indian Banking Email Intelligence

Layout:
  Sidebar (left) -- Email account addition + Backend selector
  Tabs (right)   -- Sync Log | Analytics | Offers | Data Browser

Features:
  - Manual backend selector (LMStudio / ZeroGPU / CPU)
  - Fixed progress tracking with incremental values
  - Plotly interactive charts (instant client-side rendering)
  - Demo mode: DB cleared on exit (atexit + signal handlers)
"""

import atexit
import logging
import signal
import sys
import time
import gradio as gr
from datetime import datetime
from typing import Optional

from database import BankingDatabase
from email_agent import EmailAgent
from classifier_agent import ClassifierAgent
from vision_agent import VisionAgent
from config import (
    IMAP_SERVERS, format_inr,
    GRADIO_PRIMARY, GRADIO_BG_DARK, GRADIO_SURFACE_DARK, GRADIO_TEXT,
    INFERENCE_BACKENDS, DEFAULT_BACKEND,
    BACKEND_LMSTUDIO, BACKEND_ZEROGPU, BACKEND_GGUF,
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Global State ---
from config import DB_PATH

db = BankingDatabase()
email_agent = EmailAgent(db)
classifier_agent = ClassifierAgent()
vision_agent = VisionAgent()


# --- Demo Cleanup: Clear DB on Exit ---
def _cleanup_db():
    """Truncate all database tables on exit (demo mode)."""
    try:
        db.clear_all_data()
        logger.info("Demo cleanup: All database tables cleared on exit.")
    except Exception as e:
        logger.error(f"Demo cleanup failed: {e}")


atexit.register(_cleanup_db)


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful demo cleanup."""
    logger.info(f"Received signal {signum}, cleaning up...")
    _cleanup_db()
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# --- Custom CSS ---
CUSTOM_CSS = """
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

/* Main container — warm dark background */
.gradio-container {
    background: #12100e !important;
    max-width: 100% !important;
}

/* Sidebar styling — warm gradient */
#sidebar {
    background: linear-gradient(180deg, #1e1a16 0%, #1a1510 100%) !important;
    border-right: 1px solid #3a3228 !important;
    border-radius: 16px !important;
    padding: 16px !important;
    min-height: 85vh !important;
}

/* Card styling */
.card {
    background: #1e1a16 !important;
    border: 1px solid #3a3228 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

/* Header gradient — warm amber to rose */
.header-gradient {
    background: linear-gradient(135deg, #F59E0B 0%, #EF4444 50%, #F97316 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 1.5em;
}

/* Stats cards */
.stat-card {
    background: linear-gradient(135deg, #1e1a16 0%, #1a1510 100%) !important;
    border: 1px solid #F59E0B33 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    text-align: center;
}

/* Buttons — warm amber */
.primary-btn {
    background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #12100e !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(245, 158, 11, 0.4) !important;
}

/* Agent Status */
#agent-status {
    background: linear-gradient(135deg, #1e1a16 0%, #1a1510 100%) !important;
    border: 1px solid #F59E0B66 !important;
    border-radius: 12px !important;
    padding: 15px 20px !important;
    text-align: left !important;
    margin-bottom: 15px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3) !important;
}

#agent-status h3 {
    margin: 0 0 4px 0 !important;
    font-size: 1em !important;
    line-height: 1.4 !important;
    color: #f0ebe4 !important;
}

#agent-status p, #agent-status em {
    margin: 0 !important;
    font-size: 0.9em !important;
    color: #c4b8a8 !important;
}

/* Backend status indicator */
#backend-status {
    background: #12100e !important;
    border: 1px solid #3a3228 !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    font-size: 0.8em !important;
    margin-top: 8px !important;
    color: #c4b8a8 !important;
}

/* Sync button — warm emerald */
.sync-btn {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

.danger-btn {
    background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%) !important;
}

/* Log output */
.log-output {
    background: #0d0b09 !important;
    border: 1px solid #3a3228 !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 12px !important;
    color: #FBBF24 !important;
}

/* Tab styling — warm tabs */
.tab-nav button {
    background: #1e1a16 !important;
    color: #c4b8a8 !important;
    border: 1px solid #3a3228 !important;
    border-radius: 8px 8px 0 0 !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.tab-nav button.selected {
    background: #F59E0B !important;
    color: #12100e !important;
    border-color: #F59E0B !important;
    font-weight: 700 !important;
}

/* ── Markdown output — global white text for dark background ── */
.markdown-text,
.markdown-text *,
.prose,
.prose * {
    color: #f0ebe4 !important;
    line-height: 1.7 !important;
}

/* Markdown headings */
.markdown-text h1, .markdown-text h2, .markdown-text h3,
.markdown-text h4, .markdown-text h5, .markdown-text h6,
.prose h1, .prose h2, .prose h3,
.prose h4, .prose h5, .prose h6 {
    color: #fcd9a0 !important;
    border-bottom-color: #3a3228 !important;
}

/* Markdown links */
.markdown-text a, .prose a {
    color: #F59E0B !important;
}

/* Markdown bold */
.markdown-text strong, .prose strong {
    color: #ffffff !important;
}

/* Markdown code — inline and block */
.markdown-text code, .prose code {
    background: #2a2420 !important;
    color: #FBBF24 !important;
    border-radius: 4px !important;
    padding: 2px 5px !important;
}
.markdown-text pre, .prose pre {
    background: #1a1510 !important;
    border: 1px solid #3a3228 !important;
    border-radius: 8px !important;
}
.markdown-text pre code, .prose pre code {
    background: transparent !important;
    color: #FBBF24 !important;
    padding: 0 !important;
}

/* ── Markdown Tables — white text, warm borders ── */
.markdown-text table, .prose table {
    border-collapse: collapse !important;
    width: 100% !important;
}
.markdown-text table th, .prose table th {
    background: #2a2420 !important;
    color: #fcd9a0 !important;
    font-weight: 600 !important;
    border: 1px solid #3a3228 !important;
    padding: 10px 14px !important;
    text-align: left !important;
}
.markdown-text table td, .prose table td {
    background: #1e1a16 !important;
    color: #f0ebe4 !important;
    border: 1px solid #3a3228 !important;
    padding: 8px 14px !important;
}
.markdown-text table tr:hover td, .prose table tr:hover td {
    background: #2a2420 !important;
}

/* ── Markdown lists ── */
.markdown-text li, .prose li {
    color: #f0ebe4 !important;
}
.markdown-text li::marker, .prose li::marker {
    color: #F59E0B !important;
}

/* Input fields — warm tones */
.gradio-container .box,
.gradio-container .container,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .secondary-wrap {
    background-color: #1a1510 !important;
    border-color: #3a3228 !important;
    color: #f0ebe4 !important;
}

.gradio-container .box:focus-within,
.gradio-container .container:focus-within {
    border-color: #F59E0B !important;
}

#email-input label span,
#password-input label span,
#server-select label span,
#backend-select label span {
    background-color: transparent !important;
    border: none !important;
    color: #c4b8a8 !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #F59E0B !important;
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.2) !important;
}

/* Accordion */
.accordeon {
    background: #1e1a16 !important;
    border: 1px solid #3a3228 !important;
    border-radius: 10px !important;
}

/* Divider */
hr {
    border-color: #3a3228 !important;
}

/* Glow effect for active sections — warm amber glow */
.glow-border {
    border: 1px solid #F59E0B !important;
    box-shadow: 0 0 15px rgba(245, 158, 11, 0.15) !important;
}

/* Footer */
.footer-text {
    color: #7a6e5e !important;
    text-align: center;
    font-size: 0.8em;
    padding: 10px;
}

/* Label styling */
label span {
    color: #c4b8a8 !important;
    font-weight: 500 !important;
}

/* Plotly chart containers */
.plotly-chart .js-plotly-plot {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Progress bar fix — warm gradient */
.progress-bar {
    background: linear-gradient(90deg, #F59E0B, #F97316) !important;
    border-radius: 4px !important;
}
.wrap.svelte-j1gjts {
    max-width: 100% !important;
}
.progress-text {
    text-align: center !important;
    font-size: 0.85em !important;
    color: #c4b8a8 !important;
}
.generating {
    border: none !important;
}
.progress-level {
    justify-content: center !important;
}

/* Hide duplicate floating progress bars */
#sync-log .wrap, #sync-log .generating, #sync-log .progress-container, #sync-log [class*="progress"],
#stats .wrap, #stats .generating, #stats .progress-container, #stats [class*="progress"],
#accounts-display .wrap, #accounts-display .generating, #accounts-display .progress-container, #accounts-display [class*="progress"] {
    display: none !important;
}

/* ── Global Gradio overrides for white text ── */
.svelte-1ed2p3z, .svelte-1kcgrr5, .svelte-s1r2ej {
    color: #f0ebe4 !important;
}
.secondary-wrap, .wrap-inner {
    color: #f0ebe4 !important;
}
.block .label-wrap span, .block label span {
    color: #c4b8a8 !important;
}
.markdown-text details, .prose details {
    border: 1px solid #3a3228 !important;
    border-radius: 8px !important;
    padding: 8px !important;
    background: #1a1510 !important;
}
.markdown-text summary, .prose summary {
    color: #F59E0B !important;
    cursor: pointer !important;
}
"""


# --- Helper Functions ---

def get_dashboard_stats_html() -> str:
    """Generate HTML for dashboard statistics cards."""
    stats = db.get_dashboard_stats()
    latest = stats.get("latest_sync_date")
    latest_str = latest.strftime("%d %b %Y, %H:%M") if latest else "Never"

    return f"""
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 10px 0;">
    <div style="background: linear-gradient(135deg, #1e1a16, #1a1510); border: 1px solid #F59E0B44; border-radius: 12px; padding: 18px; text-align: center;">
        <div style="font-size: 2em; font-weight: 700; color: #F59E0B;">{stats['email_count']}</div>
        <div style="color: #c4b8a8; font-size: 0.85em; margin-top: 4px;">Emails</div>
    </div>
    <div style="background: linear-gradient(135deg, #1e1a16, #1a1510); border: 1px solid #10B98144; border-radius: 12px; padding: 18px; text-align: center;">
        <div style="font-size: 2em; font-weight: 700; color: #10B981;">{stats['transaction_count']}</div>
        <div style="color: #c4b8a8; font-size: 0.85em; margin-top: 4px;">Transactions</div>
    </div>
    <div style="background: linear-gradient(135deg, #1e1a16, #1a1510); border: 1px solid #EF444444; border-radius: 12px; padding: 18px; text-align: center;">
        <div style="font-size: 2em; font-weight: 700; color: #EF4444;">{stats['offer_count']}</div>
        <div style="color: #c4b8a8; font-size: 0.85em; margin-top: 4px;">Offers</div>
    </div>
    <div style="background: linear-gradient(135deg, #1e1a16, #1a1510); border: 1px solid #FBBF2444; border-radius: 12px; padding: 18px; text-align: center;">
        <div style="font-size: 2em; font-weight: 700; color: #FBBF24;">{latest_str}</div>
        <div style="color: #c4b8a8; font-size: 0.85em; margin-top: 4px;">Last Sync</div>
    </div>
</div>
"""


def get_backend_info_html() -> str:
    """Generate HTML showing current backend and model info."""
    backend = classifier_agent.get_backend()
    if backend == BACKEND_LMSTUDIO:
        model_info = "Qwen 2.5 3B via LMStudio API"
        vision_info = "Moondream2 via LMStudio Vision"
    elif backend == BACKEND_ZEROGPU:
        model_info = "Qwen 2.5 3B on ZeroGPU (T4)"
        vision_info = "Moondream2 on ZeroGPU"
    else:
        model_info = "Qwen 2.5 3B on CPU (slow)"
        vision_info = "Moondream2 on CPU (slow)"

    return f"""<div style="margin-top: 20px; padding: 12px; background: #12100e;
                border-radius: 10px; border: 1px solid #3a3228;">
        <div style="color: #7a6e5e; font-size: 0.75em;">
            <div>Model: {model_info}</div>
            <div>Vision: {vision_info}</div>
            <div>DB: SQLite + FTS5</div>
            <div>100% Offline Analytics</div>
        </div>
    </div>"""


# --- Backend Handler ---

def handle_backend_change(backend_name):
    """Handle backend dropdown change."""
    classifier_agent.set_backend(backend_name)
    vision_agent.set_backend(backend_name)

    # Check connectivity
    status = classifier_agent.check_backend_status()
    return status, get_backend_info_html()


# --- Sync Handler ---

def add_account(email, password, server, accounts):
    """Add an account to the sync list."""
    if email and password:
        accounts.append({"email": email, "password": password, "server": server})

    if not accounts:
        display = "No accounts added yet."
    else:
        lines = ["**Added Accounts:**"]
        for acc in accounts:
            lines.append(f"- {acc['email']} ({acc['server']}) - *********")
        display = "\n".join(lines)

    return accounts, display, "", ""


def handle_sync(accounts, progress=gr.Progress()):
    """Handle the Network Sync button click for multiple accounts."""
    if not accounts:
        yield (
            "### Agent Status: `Error`\n*No accounts configured.*",
            "Please add at least one email account before syncing.",
            get_dashboard_stats_html(),
            accounts,
            "No accounts added yet."
        )
        return

    log_messages = []
    total_steps = len(accounts) * 3 + 3  # accounts * (connect+scan+save) + vision + classify + summary
    current_step = 0

    def progress_callback(msg):
        log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def update_progress(desc):
        nonlocal current_step
        current_step += 1
        frac = min(current_step / total_steps, 0.99)
        progress(frac, desc=desc)

    # Phase 1: Email sync via IMAP for all accounts
    progress_callback("=" * 50)
    progress_callback("PHASE 1: Email Delta Sync (Multi-Account)")
    progress_callback("=" * 50)
    update_progress("Phase 1: Connecting to IMAP servers...")

    yield (
        "### Agent Status: `Running`\n*Phase 1/3 -- Connecting to IMAP servers...*",
        "\n".join(log_messages),
        get_dashboard_stats_html(),
        accounts,
        "**Added Accounts:**\n" + "\n".join([f"- {a['email']} ({a['server']}) - *********" for a in accounts])
    )

    all_new_ids = []
    total_fetched = 0
    all_banks = set()
    all_errors = []

    for i, acc in enumerate(accounts):
        progress_callback(f"\nSyncing account: {acc['email']} ({acc['server']})")
        update_progress(f"Syncing {acc['email']}...")

        yield (
            f"### Agent Status: `Running`\n*Phase 1/3 -- Syncing {acc['email']}... ({i+1}/{len(accounts)})*",
            "\n".join(log_messages),
            get_dashboard_stats_html(),
            accounts,
            "**Added Accounts:**\n" + "\n".join([f"- {a['email']} ({a['server']}) - *********" for a in accounts])
        )

        sync_results = email_agent.sync_emails(
            email_address=acc['email'],
            app_password=acc['password'],
            server_name=acc['server'],
            progress_callback=progress_callback,
        )

        all_new_ids.extend(sync_results.get("new_email_ids", []))
        total_fetched += sync_results.get("emails_fetched", 0)
        all_banks.update(sync_results.get("banks_found", []))
        if sync_results.get("errors"):
            all_errors.extend(sync_results["errors"])
        update_progress(f"Account {acc['email']} complete")

    # SECURITY: Clear credentials from memory after all IMAP syncs
    for acc in accounts:
        acc["password"] = None
    accounts.clear()

    if not all_new_ids:
        progress_callback("\nNo new banking emails found across accounts. Database is up to date.")
        progress(1.0, desc="Sync complete -- no new emails")
        yield ("### Agent Status: `Idle`\n*Finished sync -- database up to date.*", "\n".join(log_messages), get_dashboard_stats_html(), accounts, "No accounts added yet.")
        return

    # Phase 2: Vision Agent (OCR fallback for emails with insufficient body text)
    progress_callback("")
    progress_callback("=" * 50)
    progress_callback("PHASE 2: Vision Agent (OCR Text Extraction)")
    progress_callback("=" * 50)
    update_progress("Phase 2: Vision agent extracting text from banners...")

    yield ("### Agent Status: `Running`\n*Phase 2/3 -- Vision Agent extracting text from banners...*", "\n".join(log_messages), get_dashboard_stats_html(), accounts, "No accounts added yet.")

    vision_results = vision_agent.process_pending_emails(
        db=db,
        email_agent=email_agent,
        progress_callback=progress_callback,
    )
    update_progress("Vision processing complete")

    yield ("### Agent Status: `Running`\n*Phase 2/3 -- Vision extraction complete.*", "\n".join(log_messages), get_dashboard_stats_html(), accounts, "No accounts added yet.")

    # Phase 3: Classifier Agent (classify + extract using Qwen)
    progress_callback("")
    progress_callback("=" * 50)
    progress_callback("PHASE 3: Classifier Agent (Qwen - Classify & Extract)")
    progress_callback("=" * 50)
    update_progress("Phase 3: Classifying & extracting data...")

    yield ("### Agent Status: `Running`\n*Phase 3/3 -- Classifier Agent (Qwen classify & extract)...*", "\n".join(log_messages), get_dashboard_stats_html(), accounts, "No accounts added yet.")

    classifier_agent.load_model(progress_callback)
    classify_results = classifier_agent.process_new_emails(
        db=db,
        email_ids=all_new_ids,
        progress_callback=progress_callback,
        progress_bar=progress,
    )
    update_progress("Classification complete")

    yield ("### Agent Status: `Running`\n*Phase 3/3 -- Classification complete.*", "\n".join(log_messages), get_dashboard_stats_html(), accounts, "No accounts added yet.")

    # Final summary
    progress_callback("")
    progress_callback("=" * 50)
    progress_callback("SYNC COMPLETE")
    progress_callback("=" * 50)
    progress_callback(f"Total New emails: {total_fetched}")
    progress_callback(f"Banks: {', '.join(all_banks)}")
    progress_callback(f"Transactions extracted: {classify_results['transactions_extracted']}")
    progress_callback(f"Offers extracted: {classify_results['offers_extracted']}")
    progress_callback(f"Vision OCR processed: {vision_results.get('processed', 0)}")

    if all_errors:
        progress_callback(f"\nErrors: {len(all_errors)}")
        for err in all_errors[:5]:
            progress_callback(f"  - {err}")

    progress(1.0, desc="Sync complete!")
    yield ("### Agent Status: `Idle`\n*Sync Complete -- all phases finished.*", "\n".join(log_messages), get_dashboard_stats_html(), accounts, "No accounts added yet.")


# --- Analytics Handlers ---

def handle_analytics():
    """Handle Offline Analytics -- 100% offline, no network."""
    yield (
        "Generating charts and loading LLM for analysis...",
        None, None, None,
        get_dashboard_stats_html(),
    )

    # Generate Plotly charts (instant -- client-side rendering)
    cat_plot = classifier_agent.plot_spending_by_category(db)
    bank_plot = classifier_agent.plot_spending_by_bank(db)
    trend_plot = classifier_agent.plot_daywise_trend(db)

    yield (
        "Charts ready! Analyzing spending patterns with local LLM...",
        cat_plot,
        bank_plot,
        trend_plot,
        get_dashboard_stats_html(),
    )

    # Load model if needed and generate analysis
    classifier_agent.load_model()
    analysis_text = classifier_agent.analyze_spending(db)

    yield (
        analysis_text,
        cat_plot,
        bank_plot,
        trend_plot,
        get_dashboard_stats_html(),
    )

def handle_offers():
    """Handle Offline RAG Offer Summaries -- 100% offline, no network."""
    yield "Loading offers from database...", get_dashboard_stats_html()

    offers = db.get_offers(limit=30)

    if not offers:
        yield "No offers found in the database. Run a **Network Sync** first to fetch banking emails.", get_dashboard_stats_html()
        return

    # Build offer cards with email review
    lines = ["# Banking Offers\n"]
    lines.append(f"**{len(offers)} offers** found across your synced accounts.\n")

    for i, offer in enumerate(offers, 1):
        bank = offer.get("bank_name", "Unknown")
        subject = offer.get("email_subject", "(No Subject)")
        offer_text = offer.get("offer_text", "")[:300]
        email_body = offer.get("email_body", "")
        email_date = (offer.get("email_date") or "")[:16]
        category = offer.get("category", "General")

        lines.append(f"---")
        lines.append(f"### {i}. [{bank}] {subject}")
        lines.append(f"**Date:** {email_date}  |  **Category:** {category}\n")
        lines.append(f"**Offer:** {offer_text}\n")

        # Show original email body for review (truncated)
        if email_body:
            # Clean up the body for display
            body_preview = email_body.strip()[:500]
            body_preview = body_preview.replace("\n\n\n", "\n\n").replace("  ", " ")
            lines.append(f"<details><summary><b>Review Original Email</b></summary>\n")
            lines.append(f"```\n{body_preview}\n```\n")
            lines.append(f"</details>\n")

    # Now add LLM summary
    lines.append("\n---\n")
    lines.append("## AI Summary\n")
    lines.append("*Generating summary with local LLM...*")
    yield "\n".join(lines), get_dashboard_stats_html()

    # Load model if needed
    classifier_agent.load_model()

    # RAG: Query database + LLM summarize
    summary = classifier_agent.summarize_offers(db)

    # Replace the placeholder with actual summary
    lines = lines[:-1]  # Remove the "Generating..." line
    lines.append(summary)

    yield "\n".join(lines), get_dashboard_stats_html()


def handle_recent_emails():
    """Get recent emails from the database for display."""
    emails = db.get_recent_emails(limit=20)
    if not emails:
        return "No emails in database. Run a Network Sync first."

    lines = ["# Recent Banking Emails\n"]
    for e in emails:
        bank = e.get("bank_name", "?")
        subject = e.get("subject", "(No Subject)")
        date = e.get("received_date", "")[:16]
        etype = e.get("email_type", "unclassified")
        icons = {
            "FUNDS_CREDITED_ALERT": "[CREDIT]",
            "FUNDS_DEBITED_ALERT": "[DEBIT]",
            "OTP_SECURITY_ALERT": "[SECURITY]",
            "CREDIT_LOAN_PROMOTION": "[PROMO]",
            "REGULATORY_KYC_NOTICE": "[KYC]",
            "ACCOUNT_STATEMENT_BILL": "[BILL]",
            "CREDIT_SCORE_BUREAU_ALERT": "[BUREAU]",
            "CREDIT_LIMIT_CARD_MANAGEMENT": "[CARD]",
        }
        icon = icons.get(etype, "[EMAIL]")
        lines.append(f"- {icon} **[{bank}]** {subject}  \n  _{date}_ | `{etype}`")

    return "\n".join(lines)


def handle_transactions():
    """Get recent transactions from database."""
    txns = db.get_transactions(limit=30)
    if not txns:
        return "No transactions found. Run Network Sync first."

    lines = ["# Recent Transactions\n"]
    lines.append("| Email Date | Subject | Bank | Type | Amount | Merchant | Category |")
    lines.append("|------------|---------|------|------|--------|----------|----------|")

    for t in txns:
        # Use email received date (reliable) instead of LLM-extracted transaction_date
        email_date = str(t.get("email_received_date", "") or "")[:10]
        subject = t.get("email_subject", "--") or "--"
        # Truncate long subjects
        if len(subject) > 40:
            subject = subject[:37] + "..."
        bank = t.get("bank_name", "?")
        ttype = t.get("transaction_type", "?")
        icon = "🔴" if ttype == "debit" else "🟢"
        amount = format_inr(t.get("amount", 0))
        merchant = t.get("merchant", "--") or "--"
        cat = t.get("category", "--") or "--"
        lines.append(f"| {email_date} | {subject} | {bank} | {icon} {ttype} | {amount} | {merchant} | {cat} |")

    return "\n".join(lines)


# --- Build Gradio App ---

def create_app():
    """Build and return the Gradio application."""

    with gr.Blocks(
        title="Indian Banking Email Intelligence",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.Color(
                c50="#fffbeb", c100="#fef3c7", c200="#fde68a",
                c300="#fcd34d", c400="#fbbf24", c500="#F59E0B",
                c600="#D97706", c700="#B45309", c800="#92400E",
                c900="#78350F", c950="#451a03",
            ),
            neutral_hue=gr.themes.Color(
                c50="#faf8f5", c100="#f0ebe4", c200="#e0d6c8",
                c300="#c4b8a8", c400="#a89888", c500="#7a6e5e",
                c600="#5e5448", c700="#3a3228", c800="#1e1a16",
                c900="#1a1510", c950="#12100e",
            ),
        ).set(
            input_background_fill="#1a1510",
            input_background_fill_dark="#1a1510",
            input_border_color="#3a3228",
            input_border_color_dark="#3a3228",
            block_background_fill="#1e1a16",
            block_background_fill_dark="#1e1a16",
        ),
    ) as app:
        accounts_state = gr.State([])

        # --- Header ---
        with gr.Row():
            gr.HTML("""
                <div style="text-align: center; padding: 20px 0 10px 0;">
                    <h1 style="background: linear-gradient(135deg, #F59E0B 0%, #EF4444 50%, #F97316 100%);
                               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                               font-size: 2em; font-weight: 700; margin: 0;">
                        Indian Banking Email Intelligence
                    </h1>
                    <p style="color: #c4b8a8; margin-top: 8px; font-size: 0.95em;">
                        Multi-Agent Dashboard | Offline RAG | Local LLM | Delta Sync
                    </p>
                </div>
            """)

        # --- Dashboard Stats ---
        stats_display = gr.HTML(value=get_dashboard_stats_html(), elem_id="stats")

        # --- Main Layout: Sidebar + Content ---
        with gr.Row():

            # --- LEFT SIDEBAR (Accounts + Backend Only) ---
            with gr.Column(scale=1, min_width=280, elem_id="sidebar"):

                gr.HTML("""
                    <div style="padding: 8px 0; margin-bottom: 10px;">
                        <div style="color: #6C63FF; font-weight: 600; font-size: 1.1em;">
                            Control Panel
                        </div>
                        <div style="color: #555577; font-size: 0.8em; margin-top: 4px;">
                            Add accounts & configure backend
                        </div>
                    </div>
                """)

                # --- Email Accounts Section ---
                with gr.Accordion("Email Accounts", open=True):
                    gr.Markdown("Add your IMAP email accounts below for multi-account sync.")

                    with gr.Row():
                        email_input = gr.Textbox(
                            label="Email Address",
                            placeholder="you@gmail.com",
                            elem_id="email-input",
                            scale=2
                        )
                        server_dropdown = gr.Dropdown(
                            label="IMAP Server",
                            choices=list(IMAP_SERVERS.keys()),
                            value="Gmail",
                            elem_id="server-select",
                            scale=1
                        )

                    password_input = gr.Textbox(
                        label="App Password",
                        type="password",
                        placeholder="xxxx xxxx xxxx xxxx",
                        elem_id="password-input",
                    )

                    gr.HTML("""
                        <div style='font-size: 0.8em; color: #8888aa; margin-top: -10px; margin-bottom: 10px;'>
                            Need an App Password?
                            <a href='https://myaccount.google.com/apppasswords' target='_blank' style='color: #43E8D8;'>Gmail</a> |
                            <a href='https://account.live.com/proofs/manage/additional' target='_blank' style='color: #43E8D8;'>Outlook</a> |
                            <a href='https://login.yahoo.com/account/security' target='_blank' style='color: #43E8D8;'>Yahoo</a>
                        </div>
                    """)

                    add_account_btn = gr.Button("Add Account", variant="secondary")
                    accounts_display = gr.Markdown("No accounts added yet.", elem_classes=["markdown-text"])

                gr.HTML('<hr style="margin: 8px 0; border-color: #2a2a4a;">')

                # --- Backend Selector (compact) ---
                with gr.Accordion("Inference Backend", open=False):
                    gr.Markdown("Select where the LLM runs. LMStudio for local, ZeroGPU for HF Spaces.")
                    backend_dropdown = gr.Dropdown(
                        label="Backend",
                        choices=INFERENCE_BACKENDS,
                        value=DEFAULT_BACKEND,
                        elem_id="backend-select",
                    )
                    backend_status = gr.Textbox(
                        label="Status",
                        value="Click 'Check' or change backend to see status.",
                        interactive=False,
                        elem_id="backend-status",
                        lines=2,
                        max_lines=3,
                    )
                    check_backend_btn = gr.Button("Check Connection", variant="secondary", size="sm")

                # System Info (dynamic)
                system_info = gr.HTML(value=get_backend_info_html(), elem_id="system-info")

            # --- RIGHT CONTENT AREA (Tabs) ---
            with gr.Column(scale=3):

                with gr.Tabs() as tabs:

                    # --- Tab 1: Sync Log ---
                    with gr.TabItem("Sync Log", id="sync-tab"):
                        sync_btn = gr.Button(
                            "Network Sync",
                            variant="primary",
                            elem_classes=["sync-btn"],
                            elem_id="sync-btn",
                        )
                        agent_status = gr.Markdown(
                            "### Agent Status: `Idle`\n*Waiting for user action.*",
                            elem_id="agent-status",
                        )
                        sync_log_output = gr.Textbox(
                            label="Sync Progress",
                            lines=20,
                            max_lines=30,
                            interactive=False,
                            elem_classes=["log-output"],
                            elem_id="sync-log",
                            value="Ready. Click 'Network Sync' to fetch new banking emails.",
                        )

                    # --- Tab 2: Analytics ---
                    with gr.TabItem("Analytics", id="analytics-tab") as analytics_tab:
                        analytics_text = gr.Markdown(
                            value="Select this tab to view your spending analysis. It will populate automatically.",
                            elem_classes=["markdown-text"],
                            elem_id="analytics-text",
                        )

                        with gr.Row():
                            cat_plot_output = gr.Plot(
                                label="Spending by Category",
                                elem_id="cat-plot",
                            )
                            bank_plot_output = gr.Plot(
                                label="Spending by Bank",
                                elem_id="bank-plot",
                            )

                        trend_plot_output = gr.Plot(
                            label="Monthly Trend",
                            elem_id="trend-plot",
                        )

                    # --- Tab 3: Offers ---
                    with gr.TabItem("Offers", id="offers-tab"):
                        offers_btn = gr.Button(
                            "Offer Summaries",
                            variant="primary",
                            elem_classes=["primary-btn"],
                            elem_id="offers-btn",
                        )
                        offers_text = gr.Markdown(
                            value="Click **Offer Summaries** to generate a RAG summary of banking offers from your database.",
                            elem_classes=["markdown-text"],
                            elem_id="offers-text",
                        )

                    # --- Tab 4: Data Browser ---
                    with gr.TabItem("Data Browser", id="data-tab"):
                        with gr.Row():
                            emails_btn = gr.Button(
                                "Recent Emails",
                                variant="secondary",
                                elem_id="emails-btn",
                            )
                            txns_btn = gr.Button(
                                "Transaction Table",
                                variant="secondary",
                                elem_id="txns-btn",
                            )
                        data_output = gr.Markdown(
                            value="Click **Recent Emails** or **Transaction Table** to browse your data.",
                            elem_classes=["markdown-text"],
                            elem_id="data-output",
                        )

        # --- Footer ---
        gr.HTML("""
            <div style="text-align: center; padding: 15px; color: #7a6e5e; font-size: 0.8em; margin-top: 10px;">
                Indian Banking Email Intelligence | Built with Gradio | Powered by Qwen 2.5 & Moondream2 | 100% Offline RAG
            </div>
        """)

        # --- Event Handlers ---

        # Backend selector
        backend_dropdown.change(
            fn=handle_backend_change,
            inputs=[backend_dropdown],
            outputs=[backend_status, system_info],
        )
        check_backend_btn.click(
            fn=handle_backend_change,
            inputs=[backend_dropdown],
            outputs=[backend_status, system_info],
        )

        add_account_btn.click(
            fn=add_account,
            inputs=[email_input, password_input, server_dropdown, accounts_state],
            outputs=[accounts_state, accounts_display, email_input, password_input]
        )

        sync_btn.click(
            fn=handle_sync,
            inputs=[accounts_state],
            outputs=[agent_status, sync_log_output, stats_display, accounts_state, accounts_display],
        )

        analytics_tab.select(
            fn=handle_analytics,
            inputs=[],
            outputs=[analytics_text, cat_plot_output, bank_plot_output,
                     trend_plot_output, stats_display],
        )

        offers_btn.click(
            fn=handle_offers,
            inputs=[],
            outputs=[offers_text, stats_display],
        )

        emails_btn.click(
            fn=handle_recent_emails,
            inputs=[],
            outputs=[data_output],
        )

        txns_btn.click(
            fn=handle_transactions,
            inputs=[],
            outputs=[data_output],
        )

    return app


# --- Launch ---

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
