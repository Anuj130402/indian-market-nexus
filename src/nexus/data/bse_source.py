"""
BSE corporate-announcement ingestion — the first "Eyes" data source.

Uses the (unofficial) `bse` library [pip install bse] to pull corporate
announcements for the universe. Each announcement already carries a HEADLINE,
a subject, and a category/subcategory as PLAIN TEXT, so this yields a usable
text signal WITHOUT any PDF parsing. PDF attachments are noted (for later deep
extraction) but not required for the first-pass NLP.

STATUS: ingestion scaffolding. The row-mapping is tested; running it hits BSE
from YOUR machine (this is the only step that needs the network).
"""
from __future__ import annotations
import time
from datetime import datetime, date
from typing import Iterator

# BSE hosts attachments under this path (live filings); historical use AttachHis.
ATTACH_BASE = "https://www.bseindia.com/xml-data/corpfiling/AttachLive/"


def _parse_dt(s: str | None):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except (ValueError, TypeError):
        return None


def _to_row(ann: dict, ticker: str) -> dict:
    """Map one raw BSE announcement dict -> a clean, flat row for our dataset."""
    headline = (ann.get("HEADLINE") or "").strip()
    subject = (ann.get("NEWSSUB") or "").strip()
    attach = (ann.get("ATTACHMENTNAME") or "").strip()
    return {
        "news_id": ann.get("NEWSID"),
        "ticker": ticker,
        "scrip_code": str(ann.get("SCRIP_CD") or ""),
        "date": _parse_dt(ann.get("DT_TM") or ann.get("NEWS_DT")),
        "headline": headline,
        "subject": subject,
        # the NLP text: headline + subject is enough for a first-pass signal
        "text": (headline + " " + subject).strip(),
        "category": ann.get("CATEGORYNAME") or "",
        "subcategory": ann.get("SUBCATNAME") or "",
        "has_pdf": bool(int(ann.get("PDFFLAG") or 0)),
        "pdf_url": (ATTACH_BASE + attach) if attach else "",
    }


class BSEAnnouncements:
    """Thin wrapper over the `bse` library that yields clean announcement rows."""

    def __init__(self, download_folder: str = "./data/raw/bse_pdfs", polite_delay: float = 0.6):
        from bse import BSE  # imported lazily so the module loads without the dep
        self.bse = BSE(download_folder=download_folder)
        self.delay = polite_delay

    def scrip_code(self, ticker: str) -> str | None:
        """Map our ticker (e.g. 'RELIANCE.NS') to a BSE scrip code (e.g. '500325')."""
        name = ticker.replace(".NS", "").replace(".BO", "").strip()
        try:
            return self.bse.getScripCode(name)
        except Exception:
            return None

    def fetch_ticker(self, ticker: str, from_date: datetime, to_date: datetime) -> Iterator[dict]:
        """Paginate all announcements for one ticker in [from_date, to_date]."""
        code = self.scrip_code(ticker)
        if not code:
            return
        page = 1
        while True:
            resp = self.bse.announcements(page_no=page, from_date=from_date,
                                          to_date=to_date, scripcode=str(code))
            table = resp.get("Table", []) if resp else []
            if not table:
                break
            for ann in table:
                yield _to_row(ann, ticker)
            # Table1[0]['ROWCNT'] is the total; stop when we've paged past it.
            total = (resp.get("Table1") or [{}])[0].get("ROWCNT", 0)
            if page * len(table) >= total or len(table) == 0:
                break
            page += 1
            time.sleep(self.delay)      # be polite to BSE's servers

    def close(self):
        try:
            self.bse.exit()
        except Exception:
            pass