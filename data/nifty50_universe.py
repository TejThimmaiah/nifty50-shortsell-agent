"""
Nifty 50 Universe — The ONLY stocks this agent trades.
Updated as of March 2026. Refresh annually or when NSE announces rebalancing.

All 50 stocks are:
  - Highly liquid (> 2M shares/day average volume)
  - F&O eligible (can be shorted intraday with MIS margin)
  - Institutional grade (well-researched, tight spreads)
  - Part of India's benchmark index

IMPORTANT: Only short-selling on these 50 stocks. No exceptions.
No Nifty Next 50, no mid-caps, no small-caps, no F&O-only names.
Tight focus = deep expertise = better edge.
"""

# ─────────────────────────────────────────────────────────────────────────────
# NIFTY 50 CONSTITUENTS (NSE symbols)
# ─────────────────────────────────────────────────────────────────────────────
NIFTY50 = [
    # Financial Services
    "HDFCBANK",      # HDFC Bank
    "ICICIBANK",     # ICICI Bank
    "SBIN",          # State Bank of India
    "AXISBANK",      # Axis Bank
    "KOTAKBANK",     # Kotak Mahindra Bank
    "BAJFINANCE",    # Bajaj Finance
    "BAJAJFINSV",    # Bajaj Finserv
    "INDUSINDBK",    # IndusInd Bank
    "SHRIRAMFIN",    # Shriram Finance

    # Information Technology
    "TCS",           # Tata Consultancy Services
    "INFY",          # Infosys
    "HCLTECH",       # HCL Technologies
    "WIPRO",         # Wipro
    "TECHM",         # Tech Mahindra
    "LTIM",          # LTIMindtree

    # Energy & Oil
    "RELIANCE",      # Reliance Industries
    "ONGC",          # Oil & Natural Gas Corp
    "BPCL",          # Bharat Petroleum
    "NTPC",          # NTPC (Power)
    "POWERGRID",     # Power Grid Corp

    # Consumer & FMCG
    "HINDUNILVR",    # Hindustan Unilever
    "ITC",           # ITC
    "NESTLEIND",     # Nestle India
    "BRITANNIA",     # Britannia Industries
    "GODREJCP",      # Godrej Consumer Products

    # Automotive
    "TATAMOTORS",    # Tata Motors
    "MARUTI",        # Maruti Suzuki
    "BAJAJ-AUTO",    # Bajaj Auto
    "EICHERMOT",     # Eicher Motors
    "HEROMOTOCO",    # Hero MotoCorp
    "M&M",           # Mahindra & Mahindra

    # Pharma & Healthcare
    "SUNPHARMA",     # Sun Pharmaceutical
    "DRREDDY",       # Dr. Reddy's Laboratories
    "CIPLA",         # Cipla
    "DIVISLAB",      # Divi's Laboratories
    "APOLLOHOSP",    # Apollo Hospitals

    # Metals & Mining
    "TATASTEEL",     # Tata Steel
    "JSWSTEEL",      # JSW Steel
    "HINDALCO",      # Hindalco Industries
    "COALINDIA",     # Coal India

    # Infrastructure & Capital Goods
    "LARSEN",        # Larsen & Toubro
    "ADANIENT",      # Adani Enterprises
    "ADANIPORTS",    # Adani Ports
    "ULTRACEMCO",    # UltraTech Cement
    "GRASIM",        # Grasim Industries

    # Telecom & Media
    "BHARTIARTL",    # Bharti Airtel

    # Consumer Discretionary
    "TITAN",         # Titan Company
    "TRENT",         # Trent
    "ASIANPAINT",    # Asian Paints

    # Financial
    "JSWENERGY",     # JSW Energy (as of latest rebalancing)
]

# Ensure exactly 50
assert len(NIFTY50) == 50, f"Nifty 50 list has {len(NIFTY50)} stocks, expected 50"

# ─────────────────────────────────────────────────────────────────────────────
# SECTOR MAPPING (Nifty 50 stocks only)
# ─────────────────────────────────────────────────────────────────────────────
NIFTY50_SECTORS = {
    "FINANCIAL":  ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK",
                   "BAJFINANCE", "BAJAJFINSV", "INDUSINDBK", "SHRIRAMFIN"],
    "IT":         ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM"],
    "ENERGY":     ["RELIANCE", "ONGC", "BPCL", "NTPC", "POWERGRID", "JSWENERGY"],
    "FMCG":       ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "GODREJCP"],
    "AUTO":       ["TATAMOTORS", "MARUTI", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "M&M"],
    "PHARMA":     ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP"],
    "METALS":     ["TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA"],
    "INFRA":      ["LARSEN", "ADANIENT", "ADANIPORTS", "ULTRACEMCO", "GRASIM"],
    "TELECOM":    ["BHARTIARTL"],
    "CONSUMER":   ["TITAN", "TRENT", "ASIANPAINT"],
}

def get_sector(symbol: str) -> str:
    """Return the sector for a Nifty 50 symbol."""
    for sector, stocks in NIFTY50_SECTORS.items():
        if symbol in stocks:
            return sector
    return "UNKNOWN"


def is_nifty50(symbol: str) -> bool:
    """Check if a symbol is in Nifty 50."""
    return symbol in NIFTY50


# ── Instrument token reverse lookup (NSE token → symbol name) ─────────────────
# Used by FreeTickStreamer._token_to_symbol()
# Note: These are approximate NSE instrument tokens. Verify against kite.instruments("NSE").
NIFTY50_TOKENS: dict = {
    738561:  "RELIANCE",
    341249:  "TCS",
    341251:  "HDFCBANK",
    408065:  "INFY",
    315394:  "HINDUNILVR",
    1270529: "ICICIBANK",
    492033:  "KOTAKBANK",
    2939649: "LARSEN",
    779521:  "SBIN",
    2714625: "BHARTIARTL",
    60417:   "ASIANPAINT",
    5633:    "AXISBANK",
    4268801: "BAJFINANCE",
    3787521: "WIPRO",
    2815745: "MARUTI",
    1330177: "HCLTECH",
    2952193: "ULTRACEMCO",
    897537:  "TITAN",
    2977281: "NTPC",
    857857:  "SUNPHARMA",
    2809089: "POWERGRID",
    4598529: "NESTLEIND",
    4267265: "BAJAJFINSV",
    3465729: "TECHM",
    3505921: "TATASTEEL",
    225537:  "DRREDDY",
    2800641: "DIVISLAB",
    1660673: "HINDALCO",
    3861249: "ADANIENT",
    633601:  "ONGC",
    3001089: "JSWSTEEL",
    3456768: "TATAMOTORS",
    1215745: "GRASIM",
    694978:  "CIPLA",
    694144:  "COALINDIA",
    4364289: "BRITANNIA",
    526849:  "BPCL",
    232961:  "EICHERMOT",
    288513:  "M&M",
    119553:  "BAJAJ-AUTO",
    345089:  "HEROMOTOCO",
    1510401: "INDUSINDBK",
    3832577: "SHRIRAMFIN",
    4343041: "TRENT",
    3834113: "LTIM",
    4458241: "GODREJCP",
    # Additional tokens for remaining Nifty50 stocks can be added as discovered
}
