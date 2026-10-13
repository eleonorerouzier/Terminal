"""List of assets followed by the terminal. Edit freely: {category: {display name: Yahoo Finance ticker}}.
Prices are in each asset's local currency. Always double-check a ticker on finance.yahoo.com."""

UNIVERSE = {
    "Sport / Running / Fitness": {
        "ASICS (ADR)": "ASCCY", "Garmin": "GRMN", "On Holding": "ONON", "Adidas": "ADS.DE",
        "Technogym": "TGYM.MI", "Nike": "NKE", "Puma": "PUM.DE",
    },
    "Cars": {
        "Hyundai": "005380.KS", "Kia": "000270.KS", "Toyota (and Lexus)": "7203.T", "Honda": "7267.T",
        "Mazda": "7261.T", "Suzuki": "7269.T", "BYD": "1211.HK", "Geely Automobile (Lynk & Co)": "0175.HK",
        "Leapmotor": "9863.HK", "Great Wall Motor": "2333.HK", "Li Auto": "LI", "Xpeng": "XPEV",
        "SAIC Motor (MG)": "600104.SS", "Ferrari": "RACE",
    },
    "Industrials / Defense / Energy": {
        "Siemens": "SIE.DE", "Safran": "SAF.PA", "Thales": "HO.PA", "Airbus": "AIR.PA",
        "Rheinmetall": "RHM.DE", "Dassault Aviation": "AM.PA", "Schneider Electric": "SU.PA",
        "TotalEnergies": "TTE.PA", "Air Liquide": "AI.PA",
    },
    "Health / Pharma": {
        "Eli Lilly": "LLY", "Pfizer": "PFE", "Sanofi (ADR)": "SNY",
    },
    "Tech / Semiconductors / Media": {
        "Microsoft": "MSFT", "Apple": "AAPL", "Alphabet": "GOOGL", "Nvidia": "NVDA", "ASML": "ASML",
        "TSMC": "TSM", "Intel": "INTC", "Netflix": "NFLX",
    },
    "Finance / Payments": {
        "Mastercard": "MA", "Visa": "V", "JPMorgan": "JPM", "SoFi": "SOFI", "Webull": "BULL",
    },
    "Defensive / Dividend": {
        "Berkshire Hathaway": "BRK-B", "Procter & Gamble": "PG", "Nestlé": "NESN.SW",
        "Johnson & Johnson": "JNJ", "Coca-Cola": "KO",
    },
    "Income funds & REITs": {
        "BlackRock Capital Allocation Trust": "BCAT", "Blackstone Mortgage Trust": "BXMT",
        "Goldman Sachs BDC": "GSBD",
    },
    "Core ETFs": {
        "MSCI World ETF": "IWDA.AS", "FTSE All-World ETF": "VWCE.DE",
    },
    "Bonds & cash ETFs": {
        "Euro government bonds ETF": "IEGA.AS", "Euro money market ETF": "XEON.DE",
    },
}

# Categories treated as the calm "core" in the Core vs themes tab and in the what-if shock default
CORE_CATEGORIES = {"Core ETFs", "Bonds & cash ETFs"}
# Categories available in the sidebar but NOT selected by default
OPTIONAL_CATEGORIES = {"Bonds & cash ETFs"}

# Example amounts (EUR) pre-filled in the Portfolio tab: replace them with your own in the app.
EXAMPLE_AMOUNTS = {
    "Microsoft": 500, "Mastercard": 500, "Toyota (and Lexus)": 500,
    "Garmin": 250, "Air Liquide": 250, "Eli Lilly": 250,
}


# ------------------------------------------------------------------ currencies
# Currency deduced from the Yahoo ticker suffix (no suffix = US listing, USD).
SUFFIX_CCY = {
    ".T": "JPY", ".KS": "KRW", ".KQ": "KRW", ".HK": "HKD", ".SW": "CHF", ".CO": "DKK", ".NS": "INR",
    ".BO": "INR", ".ST": "SEK", ".OL": "NOK", ".AX": "AUD", ".TO": "CAD", ".SS": "CNY", ".SZ": "CNY",
    ".TW": "TWD", ".SI": "SGD", ".L": "GBX",   # London quotes mostly in pence (GBX)
    ".DE": "EUR", ".PA": "EUR", ".AS": "EUR", ".MC": "EUR", ".MI": "EUR", ".BR": "EUR", ".LS": "EUR",
    ".HE": "EUR", ".VI": "EUR", ".IR": "EUR",
}


def currency_of(ticker):
    t = ticker.upper()
    if t.endswith("=X"):
        return "?"                      # currency pair: no conversion
    for suffix, ccy in SUFFIX_CCY.items():
        if t.endswith(suffix):
            return ccy
    return "USD"


def fx_pair(ccy):
    """Yahoo ticker giving units of `ccy` per 1 EUR, and a scale factor (pence -> pounds)."""
    if ccy == "GBX":
        return "EURGBP=X", 100.0
    return f"EUR{ccy}=X", 1.0
