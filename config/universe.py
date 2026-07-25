"""
Stock universe for The Indian Market Nexus.

v1 = Nifty 50 (liquid, clean, small enough that the graph is legible).

NOTE ON DATA FRESHNESS
----------------------
Index composition changes over time (constituents get added/dropped at
periodic reviews). The list below is a reasonable large-cap starting set,
but you should NOT trust it as the live Nifty 50 for production research.

TODO(anuj): replace `NIFTY50` with a live pull of the official constituent
list (NSE publishes it as a CSV) so your universe is point-in-time correct.
For a backtest you actually want the *historical* membership on each date to
avoid survivorship bias — that's a v2 concern, flagged here so we don't forget.

Tickers use the Yahoo Finance ".NS" suffix (NSE). For BSE you'd use ".BO".
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core universe: {ticker: display_name}
# ---------------------------------------------------------------------------
NIFTY50: dict[str, str] = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "INFY.NS": "Infosys",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "BAJFINANCE.NS": "Bajaj Finance",
    "AXISBANK.NS": "Axis Bank",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki",
    "HCLTECH.NS": "HCL Technologies",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "TITAN.NS": "Titan Company",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "WIPRO.NS": "Wipro",
    "NESTLEIND.NS": "Nestle India",
    "ONGC.NS": "Oil & Natural Gas Corp",
    "NTPC.NS": "NTPC",
    "POWERGRID.NS": "Power Grid Corp",
    "M&M.NS": "Mahindra & Mahindra",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS": "Tata Steel",
    "JSWSTEEL.NS": "JSW Steel",
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports & SEZ",
    "COALINDIA.NS": "Coal India",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "HDFCLIFE.NS": "HDFC Life Insurance",
    "SBILIFE.NS": "SBI Life Insurance",
    "BPCL.NS": "Bharat Petroleum",
    "GRASIM.NS": "Grasim Industries",
    "HINDALCO.NS": "Hindalco Industries",
    "DIVISLAB.NS": "Divi's Laboratories",
    "DRREDDY.NS": "Dr. Reddy's Laboratories",
    "CIPLA.NS": "Cipla",
    "BRITANNIA.NS": "Britannia Industries",
    "EICHERMOT.NS": "Eicher Motors",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "TATACONSUM.NS": "Tata Consumer Products",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "INDUSINDBK.NS": "IndusInd Bank",
    "TECHM.NS": "Tech Mahindra",
    "LTIM.NS": "LTIMindtree",
    "UPL.NS": "UPL",
}

# ---------------------------------------------------------------------------
# Sector map: {ticker: sector}
# Same-sector membership is one of your five edge types, and it's a useful
# sanity check — a "discovered" linkage that's just two stocks in the same
# sector is NOT the non-obvious insight you're hunting for.
# ---------------------------------------------------------------------------
SECTORS: dict[str, str] = {
    "RELIANCE.NS": "Energy/Conglomerate",
    "ONGC.NS": "Energy", "BPCL.NS": "Energy", "COALINDIA.NS": "Energy",
    "NTPC.NS": "Utilities", "POWERGRID.NS": "Utilities",
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT", "WIPRO.NS": "IT",
    "TECHM.NS": "IT", "LTIM.NS": "IT",
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking",
    "KOTAKBANK.NS": "Banking", "AXISBANK.NS": "Banking", "INDUSINDBK.NS": "Banking",
    "BAJFINANCE.NS": "Financials", "BAJAJFINSV.NS": "Financials",
    "HDFCLIFE.NS": "Insurance", "SBILIFE.NS": "Insurance",
    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG", "TATACONSUM.NS": "FMCG",
    "MARUTI.NS": "Auto", "TATAMOTORS.NS": "Auto", "M&M.NS": "Auto",
    "EICHERMOT.NS": "Auto", "HEROMOTOCO.NS": "Auto", "BAJAJ-AUTO.NS": "Auto",
    "TATASTEEL.NS": "Metals", "JSWSTEEL.NS": "Metals", "HINDALCO.NS": "Metals",
    "SUNPHARMA.NS": "Pharma", "DIVISLAB.NS": "Pharma", "DRREDDY.NS": "Pharma",
    "CIPLA.NS": "Pharma", "APOLLOHOSP.NS": "Healthcare",
    "ULTRACEMCO.NS": "Cement", "GRASIM.NS": "Cement",
    "ASIANPAINT.NS": "Materials", "UPL.NS": "Chemicals",
    "LT.NS": "Infrastructure", "TITAN.NS": "Consumer Durables",
    "BHARTIARTL.NS": "Telecom",
    "ADANIENT.NS": "Conglomerate", "ADANIPORTS.NS": "Logistics",
}

# ---------------------------------------------------------------------------
# Business-group map: {ticker: group}
# The archive's core example ("Tata Steel drops -> Tata Motors may follow")
# is a group relationship. Seeding these hard-coded group links gives the
# graph a spine before NLP fills in the subtler edges.
# ---------------------------------------------------------------------------
GROUPS: dict[str, str] = {
    "TCS.NS": "Tata", "TATAMOTORS.NS": "Tata", "TATASTEEL.NS": "Tata",
    "TATACONSUM.NS": "Tata", "TITAN.NS": "Tata",
    "ADANIENT.NS": "Adani", "ADANIPORTS.NS": "Adani",
    "BAJFINANCE.NS": "Bajaj", "BAJAJFINSV.NS": "Bajaj", "BAJAJ-AUTO.NS": "Bajaj",
    "HDFCBANK.NS": "HDFC", "HDFCLIFE.NS": "HDFC",
    "SBIN.NS": "SBI", "SBILIFE.NS": "SBI",
    "RELIANCE.NS": "Reliance",
    "M&M.NS": "Mahindra",
}


def tickers() -> list[str]:
    """Return the list of tickers in the active universe."""
    return list(NIFTY50.keys())


def name_to_ticker() -> dict[str, str]:
    """Inverse map (display name -> ticker) for entity resolution."""
    return {name: ticker for ticker, name in NIFTY50.items()}
