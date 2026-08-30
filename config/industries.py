"""Industry-level taxonomy for the Industry Rotation Grid (src/industry_rotation.py).

FINER-GRAINED than the 11 SPDR sectors in config/settings.py::SECTOR_ETFS —
this exists specifically to catch a move inside a sector that the 11-bucket
convergence model can't see on its own (e.g. "Gold Miners" specifically
turning, not just "Materials" broadly). It is deliberately independent of,
and does not modify, src/signals.py / refine_signals / target_weights /
signal_snapshots — see the build notes at the bottom of this file.

Relationship to the other two taxonomies in config/
----------------------------------------------------
- config/expressions.py: a CURATED "what to actually buy" vehicle list per
  BUY-class sector, used for position sizing guidance.
- config/themes.py: a narrower (33-entry) taxonomy built to let the LLM
  newsletter-tagger and the Expressions-tab self-check reason about
  sentiment at sub-sector grain.
- config/industries.py (this file): built for PRICE-ONLY rotation-state
  classification (src/industry_rotation.py) across a wider industry list,
  independent of sentiment coverage — same reasoning src/breakout_signals.py
  already gives for being un-gated.

Every `IndustryProxy.ticker` is either (a) already used elsewhere in this
codebase (config/expressions.py or config/expanded_universe.py — reused,
never duplicated as a new Expression/AssetInstrument object) or (b) a
long-established, high-AUM, single-industry fund chosen specifically
because it's NOT a fragile pick. Where no such fund exists for an
IBD-style industry, the industry is OMITTED rather than forced onto a
shaky ticker — see "Known gaps" at the bottom. This keeps the taxonomy
smaller than the full ~150 IBD industry-group list, but every entry that
IS here should hold up under a real yfinance pull.

Rules (mirrors config/expressions.py's own):
  * NO leveraged or inverse products (no daily-reset 2x/3x, no "short" funds).
  * NO individual stocks — funds only, always.
  * Every ticker used here MUST also be in the price-update universe —
    scripts/update_prices.py's ticker list AND app.py::_full_price_universe()
    both pull from INDUSTRIES; see tests/test_new_universe_config.py for the
    coverage guard.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndustryProxy:
    key: str                  # stable id, e.g. "SEMICONDUCTORS"
    label: str                 # human label, e.g. "Semiconductors"
    parent_sector: str         # owning SECTOR_ETFS key this industry rolls up to
    ticker: str                 # single fund used as the price/RS proxy
    note: str = ""              # rationale / caveats / alternates


INDUSTRIES: dict[str, IndustryProxy] = {
    # === XLK — Technology ====================================================
    "SEMICONDUCTORS": IndustryProxy(
        "SEMICONDUCTORS", "Semiconductors", "XLK", "SMH",
        "Reused from config/expressions.py (SEMIS theme). SOXX is the "
        "documented alternate — same industry, cap- vs modified-cap-weighted."),
    "SOFTWARE_ENTERPRISE": IndustryProxy(
        "SOFTWARE_ENTERPRISE", "Enterprise Software", "XLK", "IGV",
        "Reused from config/expressions.py / SOFTWARE_CLOUD theme."),
    "CLOUD_COMPUTING": IndustryProxy(
        "CLOUD_COMPUTING", "Cloud Computing", "XLK", "SKYY",
        "Reused from config/expressions.py — narrower pure-play SaaS/IaaS "
        "than IGV's broader enterprise-software basket."),
    "CYBERSECURITY": IndustryProxy(
        "CYBERSECURITY", "Cybersecurity", "XLK", "CIBR",
        "Reused from config/expressions.py (CYBER theme). HACK is the "
        "documented alternate."),
    "AI_ROBOTICS": IndustryProxy(
        "AI_ROBOTICS", "AI & Robotics", "XLK", "BOTZ",
        "Reused from config/expressions.py (AI_ROBOTICS theme). AIQ/ROBO "
        "are the documented alternates."),
    "QUANTUM_COMPUTING": IndustryProxy(
        "QUANTUM_COMPUTING", "Quantum Computing", "XLK", "QTUM",
        "Reused from config/expressions.py (QUANTUM theme)."),
    "CLEAN_ENERGY_TECH": IndustryProxy(
        "CLEAN_ENERGY_TECH", "Clean Energy Tech", "XLK", "ICLN",
        "Reused from config/expressions.py — themes.py files this under "
        "XLK (not XLU), kept consistent here."),
    "IT_HARDWARE_BROAD": IndustryProxy(
        "IT_HARDWARE_BROAD", "IT Hardware & Services (broad)", "XLK", "IYW",
        "Broad fallback (the config/expressions.py XLK iShares alternate) "
        "for hardware/IT-services names not covered by a narrower "
        "pure-play — no single-ticker 'IT services only' fund is liquid "
        "enough to trust here."),

    # === XLC — Communication Services ========================================
    "TELECOM_SERVICES": IndustryProxy(
        "TELECOM_SERVICES", "Telecom Services", "XLC", "IYZ",
        "New addition — iShares U.S. Telecommunications ETF, long-established "
        "single-industry fund; not used elsewhere in this codebase."),
    "COMMUNICATIONS_BROAD": IndustryProxy(
        "COMMUNICATIONS_BROAD", "Communication Services (broad)", "XLC", "VOX",
        "Reused from config/expressions.py — no confidently-liquid "
        "media-only or streaming-only single-ticker fund exists distinct "
        "from this broad basket, so broad is what we show."),

    # === XLY — Consumer Discretionary ========================================
    "HOME_BUILDERS": IndustryProxy(
        "HOME_BUILDERS", "Homebuilders", "XLY", "ITB",
        "Reused from config/expressions.py — pure-play builders, narrower "
        "than XHB's equal-weight basket."),
    "HOME_BUILDING_PRODUCTS": IndustryProxy(
        "HOME_BUILDING_PRODUCTS", "Homebuilding Products & Suppliers", "XLY", "XHB",
        "Reused from config/expressions.py — SPDR's equal-weight basket "
        "spans builders + building-products + home-furnishings names, "
        "genuinely broader than ITB; kept as a second, distinct read."),
    "SPECIALTY_RETAIL": IndustryProxy(
        "SPECIALTY_RETAIL", "Retail (broad)", "XLY", "XRT",
        "Reused from config/expressions.py (RETAIL theme)."),
    "EV_AUTONOMOUS_TECH": IndustryProxy(
        "EV_AUTONOMOUS_TECH", "EV & Autonomous-Driving Tech", "XLY", "IDRV",
        "Reused from config/expressions.py — EV/AV supply-chain tilt, "
        "distinct from CARZ's broader traditional-plus-EV basket."),
    "AUTO_MANUFACTURERS": IndustryProxy(
        "AUTO_MANUFACTURERS", "Auto Manufacturers (broad)", "XLY", "CARZ",
        "Reused from config/expressions.py (AUTOS_EV theme) — global "
        "automaker basket, distinct enough from IDRV's tech tilt to carry "
        "its own rotation read."),
    "AIRLINES": IndustryProxy(
        "AIRLINES", "Airlines", "XLY", "JETS",
        "Reused from config/expressions.py (AIRLINES_TRAVEL theme)."),
    "LEISURE_TRAVEL": IndustryProxy(
        "LEISURE_TRAVEL", "Leisure & Travel Services", "XLY", "PEJ",
        "Reused from config/expressions.py — hotels/cruise/leisure, "
        "distinct from JETS' pure-airline focus."),

    # === XLF — Financials =====================================================
    "REGIONAL_BANKS": IndustryProxy(
        "REGIONAL_BANKS", "Regional Banks", "XLF", "KRE",
        "Reused from config/expressions.py (REGIONAL_BANKS theme)."),
    "DIVERSIFIED_BANKS": IndustryProxy(
        "DIVERSIFIED_BANKS", "Diversified / Money-Center Banks", "XLF", "KBE",
        "Reused from config/expressions.py — broader bank basket than "
        "KRE's regional-only focus."),
    "CAPITAL_MARKETS_BROKERS": IndustryProxy(
        "CAPITAL_MARKETS_BROKERS", "Capital Markets & Broker-Dealers", "XLF", "IAI",
        "Reused from config/expressions.py (CAPITAL_MARKETS theme)."),
    "CAPITAL_MARKETS_EXCHANGES": IndustryProxy(
        "CAPITAL_MARKETS_EXCHANGES", "Capital Markets (exchanges, EW)", "XLF", "KCE",
        "Reused from config/expressions.py — SPDR equal-weight capital "
        "markets basket, includes exchanges/asset managers alongside "
        "brokers; distinct enough from IAI to carry its own read."),
    "INSURANCE_PC": IndustryProxy(
        "INSURANCE_PC", "Insurance — Property & Casualty", "XLF", "KBWP",
        "Reused from config/expressions.py (INSURANCE theme) — P&C-specific."),
    "INSURANCE_DIVERSIFIED": IndustryProxy(
        "INSURANCE_DIVERSIFIED", "Insurance — Diversified", "XLF", "IAK",
        "Reused from config/expressions.py (INSURANCE theme) — broad "
        "diversified insurance basket vs KBWP's P&C-only focus."),
    "FINTECH_PAYMENTS": IndustryProxy(
        "FINTECH_PAYMENTS", "Fintech & Payments", "XLF", "IPAY",
        "Reused from config/expressions.py (FINTECH theme)."),

    # === XLI — Industrials ====================================================
    "DEFENSE_AEROSPACE": IndustryProxy(
        "DEFENSE_AEROSPACE", "Aerospace & Defense", "XLI", "ITA",
        "Reused from config/expressions.py (DEFENSE_AERO theme). XAR "
        "(equal-weight) is the documented alternate."),
    "INFRASTRUCTURE_CONSTRUCTION": IndustryProxy(
        "INFRASTRUCTURE_CONSTRUCTION", "Infrastructure & Engineering/Construction",
        "XLI", "PAVE",
        "Reused from config/expressions.py (INFRASTRUCTURE theme). Closest "
        "available proxy for Engineering & Construction too — a literal "
        "E&C-only fund isn't liquid enough to trust separately."),
    "RAIL_TRUCKING": IndustryProxy(
        "RAIL_TRUCKING", "Rail & Trucking", "XLI", "IYT",
        "Reused from config/expressions.py (TRANSPORTS theme)."),
    "AIR_FREIGHT_LOGISTICS": IndustryProxy(
        "AIR_FREIGHT_LOGISTICS", "Air Freight & Logistics (EW)", "XLI", "XTN",
        "Reused from config/expressions.py — SPDR equal-weight transports "
        "basket, broader than IYT's rail/trucking tilt."),

    # === XLB — Materials =======================================================
    "METALS_MINING_BROAD": IndustryProxy(
        "METALS_MINING_BROAD", "Metals & Mining (broad)", "XLB", "XME",
        "Reused from config/expressions.py (METALS_MINING theme)."),
    "GOLD_MINERS": IndustryProxy(
        "GOLD_MINERS", "Gold Miners", "XLB", "GDX",
        "Reused from config/expressions.py (GOLD_SILVER_MINERS theme)."),
    "JUNIOR_GOLD_MINERS": IndustryProxy(
        "JUNIOR_GOLD_MINERS", "Junior Gold Miners", "XLB", "GDXJ",
        "Reused from config/expressions.py — higher operating leverage "
        "than GDX, genuinely a distinct rotation read (juniors often turn "
        "first, and harder, in both directions)."),
    "SILVER_MINERS": IndustryProxy(
        "SILVER_MINERS", "Silver Miners", "XLB", "SIL",
        "Reused from config/expressions.py (GOLD_SILVER_MINERS theme)."),
    "JUNIOR_SILVER_MINERS": IndustryProxy(
        "JUNIOR_SILVER_MINERS", "Junior Silver Miners", "XLB", "SILJ",
        "Reused from config/expressions.py — same senior/junior split "
        "logic as gold above."),
    "COPPER_MINERS": IndustryProxy(
        "COPPER_MINERS", "Copper Miners", "XLB", "COPX",
        "Reused from config/expressions.py (COPPER theme)."),
    "LITHIUM_BATTERY_MATERIALS": IndustryProxy(
        "LITHIUM_BATTERY_MATERIALS", "Lithium & Battery Materials", "XLB", "LIT",
        "Reused from config/expressions.py (LITHIUM_BATTERY theme)."),
    "URANIUM_MINING": IndustryProxy(
        "URANIUM_MINING", "Uranium Mining", "XLB", "URA",
        "Reused from config/expressions.py (URANIUM theme). URNM is the "
        "documented pure-play-miner alternate."),
    "RARE_EARTH_STRATEGIC_METALS": IndustryProxy(
        "RARE_EARTH_STRATEGIC_METALS", "Rare Earth & Strategic Metals", "XLB", "REMX",
        "Reused from config/expressions.py (RARE_EARTH theme)."),
    "AGRIBUSINESS": IndustryProxy(
        "AGRIBUSINESS", "Agribusiness", "XLB", "MOO",
        "Reused from config/expressions.py (AGRIBUSINESS theme)."),
    "STEEL": IndustryProxy(
        "STEEL", "Steel", "XLB", "SLX",
        "New addition — VanEck Steel ETF, long-established single-industry "
        "fund; not used elsewhere in this codebase."),
    "FOREST_PRODUCTS_TIMBER": IndustryProxy(
        "FOREST_PRODUCTS_TIMBER", "Forest Products & Timber", "XLB", "WOOD",
        "New addition — iShares Global Timber & Forestry ETF. Global, not "
        "pure-US — noted as a caveat, still the best available liquid "
        "single-ticker proxy for this industry."),
    "PRECIOUS_METALS_GOLD_SPOT": IndustryProxy(
        "PRECIOUS_METALS_GOLD_SPOT", "Precious Metals — Gold (spot)", "XLB", "GLD",
        "Reused from config/expanded_universe.py. Spot gold, not miners — "
        "a genuinely distinct rotation signal from GOLD_MINERS above "
        "(miners carry equity-market beta gold itself doesn't)."),
    "PRECIOUS_METALS_SILVER_SPOT": IndustryProxy(
        "PRECIOUS_METALS_SILVER_SPOT", "Precious Metals — Silver (spot)", "XLB", "SLV",
        "Reused from config/expanded_universe.py. Same miners-vs-spot "
        "logic as gold above."),

    # === XLE — Energy ==========================================================
    "OIL_GAS_EXPLORATION_PRODUCTION": IndustryProxy(
        "OIL_GAS_EXPLORATION_PRODUCTION", "Oil & Gas Exploration & Production",
        "XLE", "XOP",
        "Reused from config/expressions.py (OIL_GAS_EP theme)."),
    "OILFIELD_SERVICES": IndustryProxy(
        "OILFIELD_SERVICES", "Oilfield Services & Equipment", "XLE", "OIH",
        "Reused from config/expressions.py (OIL_GAS_EP theme)."),
    "NATURAL_GAS": IndustryProxy(
        "NATURAL_GAS", "Natural Gas", "XLE", "FCG",
        "Reused from config/expressions.py (OIL_GAS_EP theme)."),
    "MIDSTREAM_MLPS": IndustryProxy(
        "MIDSTREAM_MLPS", "Midstream / MLPs", "XLE", "AMLP",
        "Reused from config/expressions.py (MIDSTREAM theme)."),

    # === XLV — Health Care =====================================================
    "BIOTECH_SMALL_MID_CAP": IndustryProxy(
        "BIOTECH_SMALL_MID_CAP", "Biotechnology (small/mid-cap, EW)", "XLV", "XBI",
        "Reused from config/expressions.py — equal-weight basket carries "
        "more small/mid-cap torque than IBB below; genuinely distinct "
        "rotation behaviour (XBI often turns first)."),
    "BIOTECH_LARGE_CAP": IndustryProxy(
        "BIOTECH_LARGE_CAP", "Biotechnology (large-cap)", "XLV", "IBB",
        "Reused from config/expressions.py (BIOTECH theme)."),
    "PHARMACEUTICALS": IndustryProxy(
        "PHARMACEUTICALS", "Pharmaceuticals", "XLV", "XPH",
        "Reused from config/expressions.py (PHARMA_DEVICES theme)."),
    "MEDICAL_DEVICES": IndustryProxy(
        "MEDICAL_DEVICES", "Medical Devices", "XLV", "IHI",
        "Reused from config/expressions.py (PHARMA_DEVICES theme)."),
    "HEALTH_INSURERS_PROVIDERS": IndustryProxy(
        "HEALTH_INSURERS_PROVIDERS", "Health Insurers & Providers", "XLV", "IHF",
        "Reused from config/expressions.py (HEALTH_PROVIDERS theme)."),

    # === XLP — Consumer Staples ================================================
    "CONSUMER_STAPLES_BROAD": IndustryProxy(
        "CONSUMER_STAPLES_BROAD", "Consumer Staples (broad)", "XLP", "VDC",
        "Reused from config/expressions.py. No confidently-liquid "
        "single-ticker Food & Beverage / Household Products fund exists "
        "distinct from this broad basket — see 'Known gaps' below."),

    # === XLU — Utilities =======================================================
    "SMART_GRID_POWER_INFRA": IndustryProxy(
        "SMART_GRID_POWER_INFRA", "Smart Grid & Power Infrastructure", "XLU", "GRID",
        "Reused from config/expressions.py (SMART_GRID theme)."),
    "WATER_INFRASTRUCTURE": IndustryProxy(
        "WATER_INFRASTRUCTURE", "Water Infrastructure & Utilities", "XLU", "FIW",
        "New addition — First Trust Water ETF, long-established "
        "single-industry fund; not used elsewhere in this codebase."),

    # === XLRE — Real Estate ====================================================
    "DATA_CENTER_REITS": IndustryProxy(
        "DATA_CENTER_REITS", "Data Center REITs", "XLRE", "DTCR",
        "Reused from config/expressions.py (DATACENTER_REIT theme)."),
    "MORTGAGE_REITS": IndustryProxy(
        "MORTGAGE_REITS", "Mortgage REITs", "XLRE", "REM",
        "Reused from config/expressions.py (MORTGAGE_REIT theme). MORT is "
        "the documented alternate."),
    "RESIDENTIAL_REITS": IndustryProxy(
        "RESIDENTIAL_REITS", "Residential & Multi-Sector REITs", "XLRE", "REZ",
        "New addition — iShares Residential and Multisector Real Estate "
        "ETF; not used elsewhere in this codebase."),
    "REIT_BROAD": IndustryProxy(
        "REIT_BROAD", "REITs (broad)", "XLRE", "VNQ",
        "Reused from config/expressions.py."),

    # === UFO — Space (supplementary) ===========================================
    "SPACE_EXPLORATION": IndustryProxy(
        "SPACE_EXPLORATION", "Space Exploration & Technology", "UFO", "ARKX",
        "Reused from config/expressions.py (SPACE theme)."),
}


def all_industry_tickers() -> list[str]:
    """Deduped ticker list, insertion order — for the price-update universe."""
    seen: set[str] = set()
    out: list[str] = []
    for ind in INDUSTRIES.values():
        if ind.ticker not in seen:
            seen.add(ind.ticker)
            out.append(ind.ticker)
    return out


def industries_for_sector(sector: str) -> list[IndustryProxy]:
    return [i for i in INDUSTRIES.values() if i.parent_sector == sector]


# ---------------------------------------------------------------------------
# Known gaps — IBD-style industries deliberately NOT included, and why.
# Extend this taxonomy by adding a new IndustryProxy entry above once a
# liquid single-ticker fund proxy can be identified with confidence; do
# not silently graft one of these onto an unrelated ticker in the
# meantime. (config/industries.py::INDUSTRIES currently covers 59
# industries — a deliberately-scoped starter set, not the full ~150. Every
# entry is a fund proxy chosen for confidence over count; see the module
# docstring above for the omission policy.)
#
#   Semiconductor Equipment  — no equipment-only fund distinct from SMH/SOXX
#   Internet Content & Info  — candidates are thin/uncertain listing status
#   Media & Entertainment    — no confidently-liquid pure-play distinct from VOX
#   Restaurants               — no confidently-liquid single-ticker fund
#   Apparel & Footwear        — no confidently-liquid single-ticker fund
#   Asset Managers & Trusts  — no fund distinct enough from IAI/KCE above
#   Chemicals                 — no dedicated liquid single-ticker fund
#   Coal                      — historically high fund-closure risk in this niche
#   Refiners                  — no confidently-liquid single-ticker fund
#   Food & Beverage           — no confidently-liquid single-ticker fund
#   Household & Personal Care — no confidently-liquid single-ticker fund
#   Industrial/Retail REITs  — no confidently-liquid single-ticker fund
#     distinct from VNQ (broad) or REZ (residential) above
# ---------------------------------------------------------------------------