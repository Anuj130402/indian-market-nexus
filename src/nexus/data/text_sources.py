"""
Text-source ingestion (The "Eyes" — raw material for the NLP graph builder).

This is the *structural* data stream: annual reports, earnings-call transcripts,
and news. NLP reads these to decide which companies are wired together.

STATUS: STUB. These are the interfaces the NLP layer will consume. The priority
order from the spec sheet is annual reports (1) > transcripts (2) > news (3),
so implement `AnnualReportSource` first.

Design note: every source yields the same `Document` shape, so the NLP pipeline
never cares where the text came from. Add a source, not a special case.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class Document:
    """One unit of text plus the metadata NLP needs to attribute it."""
    source_type: str          # "annual_report" | "transcript" | "news"
    primary_ticker: str | None  # the company the doc is *about*, if known
    published: date | None
    text: str
    url: str | None = None


class TextSource:
    """Interface every concrete source implements."""
    def fetch(self, ticker: str) -> list[Document]:
        raise NotImplementedError


class AnnualReportSource(TextSource):
    """
    TODO(v1, priority 1): fetch annual report PDFs and extract text.
    - Source: BSE/NSE corporate filings, or company IR pages.
    - Use the `pdf` extraction path (pdfplumber / PyPDF) to get text.
    - Richest source of *structural* edges (subsidiaries, JVs, key customers).
    """
    def fetch(self, ticker: str) -> list["Document"]:
        raise NotImplementedError("AnnualReportSource.fetch — implement in v1")


class TranscriptSource(TextSource):
    """TODO(v1, priority 2): earnings-call transcripts (forward-looking edges)."""
    def fetch(self, ticker: str) -> list["Document"]:
        raise NotImplementedError("TranscriptSource.fetch — implement in v1")


class NewsSource(TextSource):
    """TODO(v2, priority 3): news feed — the real-time shock *triggers*."""
    def fetch(self, ticker: str) -> list["Document"]:
        raise NotImplementedError("NewsSource.fetch — implement in v2")
