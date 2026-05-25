"""Theme taxonomy — the bridge between newsletter/news sentiment and the
expression vehicles in `config/expressions.py`.

Newsletters and news talk in *themes* ("semis", "uranium", "biotech"), not in
the 12 SPDR sector tickers and rarely in specific ETF tickers. A Theme groups
the expression tickers that share a narrative so we can:

  * tag newsletters at theme grain (parallel to the 12-sector `sector_ratings`),
  * pull theme news via a query string, and
  * rank which expression to buy inside a firing sector by its theme's sentiment.

Rules:
  * Every expression ticker maps to AT MOST ONE theme.
  * The plain sector ETFs (XLK, VGT, XLY, ... and each sector's plain proxies)
    map to NO theme — they ARE the parent; their sentiment is the sector score.
  * `parent_sector` of every theme is a key in config.settings.SECTOR_ETFS.

Keep THEME_KEYS in sync with schemas.ThemeKey (a drift test enforces this).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    key: str                            # stable id, e.g. "SEMIS"
    label: str                          # human label, e.g. "Semiconductors"
    parent_sector: str                  # owning SPDR sector, e.g. "XLK"
    expression_tickers: tuple[str, ...]  # vehicles that express this theme
    news_query: str                     # Google News RSS search string
    keywords: tuple[str, ...]           # hints for the LLM newsletter tagger


THEMES: dict[str, Theme] = {
    # --- XLK ---------------------------------------------------------------
    "SEMIS": Theme(
        "SEMIS", "Semiconductors", "XLK", ("SOXX", "SMH"),
        "semiconductor stocks",
        ("semiconductor", "semis", "chips", "foundry", "wafer", "TSMC",
         "Nvidia", "AMD", "ASML", "memory", "HBM", "fab")),
    "SOFTWARE_CLOUD": Theme(
        "SOFTWARE_CLOUD", "Software & Cloud", "XLK", ("IGV", "WCLD", "SKYY"),
        "enterprise software cloud stocks",
        ("software", "SaaS", "cloud", "enterprise software", "subscription",
         "ARR", "hyperscaler")),
    "AI_ROBOTICS": Theme(
        "AI_ROBOTICS", "AI & Robotics", "XLK", ("AIQ", "BOTZ", "ROBO"),
        "artificial intelligence robotics stocks",
        ("artificial intelligence", "AI", "machine learning", "robotics",
         "automation", "LLM", "inference", "datacenter AI")),
    "CYBER": Theme(
        "CYBER", "Cybersecurity", "XLK", ("CIBR", "HACK"),
        "cybersecurity stocks",
        ("cybersecurity", "cyber", "security software", "ransomware",
         "zero trust", "endpoint")),
    "QUANTUM": Theme(
        "QUANTUM", "Quantum Computing", "XLK", ("QTUM", "WQTM"),
        "quantum computing stocks",
        ("quantum computing", "quantum", "qubit", "IonQ", "Rigetti")),
    "CLEAN_ENERGY": Theme(
        "CLEAN_ENERGY", "Clean Energy Tech", "XLK", ("ICLN",),
        "clean energy solar stocks",
        ("clean energy", "solar", "renewable", "wind", "green energy")),
    # --- XLY ---------------------------------------------------------------
    "HOMEBUILDERS": Theme(
        "HOMEBUILDERS", "Homebuilders", "XLY", ("XHB", "ITB"),
        "homebuilder stocks housing",
        ("homebuilder", "housing", "home construction", "Lennar", "DR Horton",
         "new home sales", "mortgage rates")),
    "RETAIL": Theme(
        "RETAIL", "Retail", "XLY", ("XRT",),
        "retail stocks consumer spending",
        ("retail", "consumer spending", "same-store sales", "e-commerce",
         "discretionary")),
    "AUTOS_EV": Theme(
        "AUTOS_EV", "Autos & EVs", "XLY", ("IDRV", "CARZ"),
        "electric vehicle auto stocks",
        ("auto", "automaker", "electric vehicle", "EV", "Tesla", "self-driving",
         "vehicle sales")),
    "AIRLINES_TRAVEL": Theme(
        "AIRLINES_TRAVEL", "Airlines & Travel", "XLY", ("JETS", "PEJ"),
        "airline travel leisure stocks",
        ("airline", "travel", "leisure", "hotels", "cruise", "load factor",
         "jet fuel")),
    # --- XLF ---------------------------------------------------------------
    "REGIONAL_BANKS": Theme(
        "REGIONAL_BANKS", "Banks (Regional & Broad)", "XLF", ("KRE", "KBE"),
        "regional bank stocks",
        ("regional bank", "bank", "deposits", "net interest margin", "NIM",
         "loan growth", "credit losses")),
    "CAPITAL_MARKETS": Theme(
        "CAPITAL_MARKETS", "Capital Markets & Brokers", "XLF", ("IAI", "KCE"),
        "investment bank broker dealer stocks",
        ("capital markets", "broker", "dealer", "investment bank", "trading",
         "IPO", "M&A", "exchange")),
    "INSURANCE": Theme(
        "INSURANCE", "Insurance", "XLF", ("KIE", "IAK", "KBWP"),
        "insurance stocks",
        ("insurance", "insurer", "P&C", "property casualty", "underwriting",
         "reinsurance", "combined ratio")),
    "FINTECH": Theme(
        "FINTECH", "Fintech & Payments", "XLF", ("FINX", "IPAY"),
        "fintech payments stocks",
        ("fintech", "payments", "digital payments", "Visa", "Mastercard",
         "PayPal", "neobank")),
    # --- XLI ---------------------------------------------------------------
    "DEFENSE_AERO": Theme(
        "DEFENSE_AERO", "Defense & Aerospace", "XLI", ("ITA", "XAR"),
        "defense aerospace stocks",
        ("defense", "aerospace", "military", "Lockheed", "RTX", "missile",
         "defense budget", "fighter")),
    "INFRASTRUCTURE": Theme(
        "INFRASTRUCTURE", "Infrastructure", "XLI", ("PAVE",),
        "infrastructure construction stocks",
        ("infrastructure", "construction", "roads", "grid", "reshoring",
         "capex", "spending bill")),
    "TRANSPORTS": Theme(
        "TRANSPORTS", "Transports", "XLI", ("IYT", "XTN"),
        "transportation rail trucking stocks",
        ("transport", "rail", "railroad", "trucking", "freight", "logistics",
         "air freight", "shipping")),
    # --- XLB ---------------------------------------------------------------
    "METALS_MINING": Theme(
        "METALS_MINING", "Metals & Mining (broad)", "XLB", ("XME",),
        "metals mining steel stocks",
        ("metals", "mining", "steel", "diversified miners", "coal", "iron ore")),
    "GOLD_SILVER_MINERS": Theme(
        "GOLD_SILVER_MINERS", "Gold & Silver Miners", "XLB",
        ("GDX", "GDXJ", "SIL", "SILJ"),
        "gold silver mining stocks",
        ("gold", "silver", "gold miner", "precious metals", "AISC", "bullion",
         "Newmont", "Barrick")),
    "COPPER": Theme(
        "COPPER", "Copper Miners", "XLB", ("COPX",),
        "copper mining stocks",
        ("copper", "copper miner", "Freeport", "red metal", "grid copper")),
    "LITHIUM_BATTERY": Theme(
        "LITHIUM_BATTERY", "Lithium & Battery", "XLB", ("LIT",),
        "lithium battery materials stocks",
        ("lithium", "battery", "battery materials", "cathode", "Albemarle",
         "energy storage")),
    "URANIUM": Theme(
        "URANIUM", "Uranium", "XLB", ("URA", "URNM"),
        "uranium mining nuclear stocks",
        ("uranium", "nuclear", "U3O8", "enrichment", "Cameco", "SMR",
         "reactor", "yellowcake")),
    "RARE_EARTH": Theme(
        "RARE_EARTH", "Rare Earth & Strategic Metals", "XLB", ("REMX",),
        "rare earth strategic metals stocks",
        ("rare earth", "strategic metals", "critical minerals", "MP Materials",
         "magnet", "neodymium")),
    "AGRIBUSINESS": Theme(
        "AGRIBUSINESS", "Agribusiness", "XLB", ("MOO",),
        "agribusiness fertilizer stocks",
        ("agribusiness", "fertilizer", "potash", "crop", "Deere", "grain",
         "ag chemicals")),
    # --- XLE ---------------------------------------------------------------
    "OIL_GAS_EP": Theme(
        "OIL_GAS_EP", "Oil & Gas E&P / Services", "XLE", ("XOP", "OIH", "FCG"),
        "oil gas exploration production stocks",
        ("oil", "crude", "WTI", "natural gas", "E&P", "shale", "drilling",
         "oilfield services", "rig count")),
    "MIDSTREAM": Theme(
        "MIDSTREAM", "Midstream / MLPs", "XLE", ("AMLP",),
        "midstream pipeline MLP stocks",
        ("midstream", "pipeline", "MLP", "Energy Transfer", "distribution",
         "throughput")),
    # --- XLV ---------------------------------------------------------------
    "BIOTECH": Theme(
        "BIOTECH", "Biotech", "XLV", ("XBI", "IBB"),
        "biotech stocks",
        ("biotech", "biotechnology", "FDA", "clinical trial", "drug approval",
         "phase 3", "M&A biotech")),
    "PHARMA_DEVICES": Theme(
        "PHARMA_DEVICES", "Pharma & Medical Devices", "XLV",
        ("IHI", "XPH", "PPH"),
        "pharmaceutical medical device stocks",
        ("pharma", "pharmaceutical", "medical device", "med-tech", "drug pricing",
         "GLP-1", "diagnostics")),
    "HEALTH_PROVIDERS": Theme(
        "HEALTH_PROVIDERS", "Healthcare Providers & Insurers", "XLV", ("IHF",),
        "health insurer managed care stocks",
        ("managed care", "health insurer", "HMO", "UnitedHealth", "Medicare",
         "medical loss ratio", "hospital")),
    # --- XLU ---------------------------------------------------------------
    "SMART_GRID": Theme(
        "SMART_GRID", "Smart Grid & Power Infrastructure", "XLU", ("GRID",),
        "electric grid power infrastructure stocks",
        ("grid", "power infrastructure", "electrification", "transmission",
         "data center power", "utilities capex")),
    # --- XLRE --------------------------------------------------------------
    "DATACENTER_REIT": Theme(
        "DATACENTER_REIT", "Data Center REITs", "XLRE", ("DTCR",),
        "data center REIT stocks",
        ("data center", "data center REIT", "Equinix", "Digital Realty",
         "colocation", "AI capacity")),
    "MORTGAGE_REIT": Theme(
        "MORTGAGE_REIT", "Mortgage REITs", "XLRE", ("REM", "MORT"),
        "mortgage REIT stocks",
        ("mortgage REIT", "mREIT", "agency MBS", "book value", "Annaly",
         "rate spread")),
    # --- UFO (parent UFO ticker excluded — it's the sector proxy) ----------
    "SPACE": Theme(
        "SPACE", "Space", "UFO", ("ARKX", "ROKT", "NASA", "ORBX", "XOVR"),
        "space industry stocks launch satellite",
        ("space", "satellite", "launch", "Rocket Lab", "SpaceX", "Starlink",
         "orbital", "lunar", "defense space")),
}


# Canonical ordered tuple of theme keys — schemas.ThemeKey must match this
# (enforced by tests/test_themes.py). Order is insertion order of THEMES.
THEME_KEYS: tuple[str, ...] = tuple(THEMES.keys())


def all_theme_keys() -> list[str]:
    return list(THEME_KEYS)


def themes_for_sector(sector: str) -> list[Theme]:
    """Themes whose parent_sector == sector, in declaration order."""
    return [t for t in THEMES.values() if t.parent_sector == sector]


def theme_for_ticker(ticker: str) -> Theme | None:
    """Reverse lookup: the (single) theme that owns an expression ticker, or
    None for plain sector proxies that map to no theme."""
    for t in THEMES.values():
        if ticker in t.expression_tickers:
            return t
    return None
