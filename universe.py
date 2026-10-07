"""List of assets followed by the terminal. Edit freely: {category: {display name: Yahoo Finance ticker}}.
Prices are in each asset's local currency. Always double-check a ticker on finance.yahoo.com."""

UNIVERSE = {
    "Asian cars": {
        "Toyota": "7203.T", "Hyundai": "005380.KS", "BYD": "1211.HK",
    },
    "Sport / Running / Fitness": {
        "On Holding": "ONON", "Amer Sports": "AS", "ASICS": "7936.T", "Adidas": "ADS.DE",
        "Garmin": "GRMN", "Planet Fitness": "PLNT", "Basic-Fit": "BFIT.AS",
        "Technogym": "TGYM.MI", "Lululemon": "LULU", "Deckers (HOKA)": "DECK",
    },
    "Tech / AI / Semiconductors": {
        "Nvidia": "NVDA", "Microsoft": "MSFT", "Alphabet": "GOOGL", "Broadcom": "AVGO",
        "TSMC": "TSM", "ASML": "ASML", "Meta": "META", "Amazon": "AMZN",
        "Palantir": "PLTR", "AMD": "AMD", "SAP": "SAP", "Oracle": "ORCL",
    },
    "Health / Pharma / Medtech": {
        "Eli Lilly": "LLY", "Novo Nordisk": "NVO", "Novartis": "NVS", "Roche": "ROG.SW",
        "AstraZeneca": "AZN", "Sanofi": "SAN.PA", "Intuitive Surgical": "ISRG",
        "Vertex": "VRTX", "Thermo Fisher": "TMO",
    },
    "Energy / Transition / Utilities": {
        "NextEra Energy": "NEE", "Iberdrola": "IBE.MC", "Enel": "ENEL.MI",
        "Schneider Electric": "SU.PA", "Siemens Energy": "ENR.DE", "Vestas": "VWS.CO",
        "First Solar": "FSLR", "TotalEnergies": "TTE.PA", "Exxon Mobil": "XOM",
        "Brent crude (futures)": "BZ=F",
    },
    "Finance / Payments / Insurance": {
        "JPMorgan": "JPM", "Visa": "V", "Mastercard": "MA", "Goldman Sachs": "GS",
        "Berkshire Hathaway": "BRK-B", "BNP Paribas": "BNP.PA", "AXA": "CS.PA", "Allianz": "ALV.DE",
    },
    "Luxury / Consumer": {
        "LVMH": "MC.PA", "Hermès": "RMS.PA", "L'Oréal": "OR.PA", "Ferrari": "RACE",
        "Nestlé": "NESN.SW", "Procter & Gamble": "PG", "Costco": "COST", "Walmart": "WMT",
    },
    "Industrials / Aerospace / Defense": {
        "Safran": "SAF.PA", "Airbus": "AIR.PA", "Thales": "HO.PA", "Rheinmetall": "RHM.DE",
        "Siemens": "SIE.DE", "GE Aerospace": "GE", "Caterpillar": "CAT",
    },
    "Materials / Gold / Chemicals": {
        "Air Liquide": "AI.PA", "Linde": "LIN", "Newmont": "NEM", "Agnico Eagle": "AEM",
        "Freeport-McMoRan": "FCX", "BHP": "BHP", "Rio Tinto": "RIO",
    },
    "Emerging markets": {
        "Alibaba": "BABA", "Tencent": "0700.HK", "MercadoLibre": "MELI",
        "Samsung Electronics": "005930.KS", "Reliance Industries": "RELIANCE.NS", "HDFC Bank": "HDB",
    },
    "Core (safe ETFs)": {
        "MSCI World ETF": "IWDA.AS", "Euro government bonds ETF": "IEGA.AS",
        "Euro money market ETF": "XEON.DE",
    },
}

CORE_CATEGORY = "Core (safe ETFs)"

# Example amounts (EUR) pre-filled in the Portfolio tab: replace them with your own in the app.
EXAMPLE_AMOUNTS = {
    "MSCI World ETF": 4000, "Euro government bonds ETF": 2000, "Euro money market ETF": 1000,
    "Toyota": 500, "Adidas": 250, "ASICS": 250, "TotalEnergies": 250,
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
