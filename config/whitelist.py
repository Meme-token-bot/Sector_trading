"""Domain whitelist for following links and PDFs out of email bodies.

Only URLs whose host ends with one of these suffixes are fetched and
appended to the LLM context. Everything else is dropped silently.

Edit this list to taste — adding a domain is the cheapest unit of
extension in this app.
"""
from __future__ import annotations

# Substack hosts everything as <author>.substack.com
SUBSTACK_DOMAINS: list[str] = [
    "substack.com",
]

# Newsletter platforms that host author content publicly
# (e.g. <author>.beehiiv.com, <author>.ghost.io)
NEWSLETTER_PLATFORMS: list[str] = [
    "beehiiv.com",
    "ghost.io",
    "buttondown.email",
    "buttondown.com",
    "kit.com",
    "mirror.xyz",
    "medium.com",
]

# Author/publication root domains
AUTHOR_DOMAINS: list[str] = [
    "lynalden.com",
    "themacrocompass.com",
    "doomberg.com",
    "hedgeye.com",
    "ssga.com",
    "blackrock.com",
    "apolloacademy.com",
    "apollo.com",
    "jpmorgan.com",
    "goldmansachs.com",
    "biancoresearch.com",  # Bianco Research
    "topdowncharts.com",   # Callum Thomas / Weekly ChartStorm
    "tker.co",             # Sam Ro / TKer
    "axios.com",           # Dan Primack (Pro Rata) and other Axios newsletters
    "etfmathguy.com",      # ETFMathGuy
    "calculatedriskblog.com",  # Bill McBride — housing, employment, XLRE
    "fedguy.com",              # Joseph Wang — Fed liquidity, money markets
    "wolfstreet.com",          # Wolf Richter — autos, CRE, credit
    "stratechery.com",         # Ben Thompson — tech / comms (XLK, XLC)
    "bespokepremium.com",      # Bespoke Investment — sector breadth
    "thefelderreport.com",     # Jesse Felder — contrarian / value
    # --- Expression-theme newsletters (theme-level coverage) ---
    "semianalysis.com",        # SemiAnalysis — semiconductors (SOXX/SMH)
    "thequantuminsider.com",   # The Quantum Insider — quantum (QTUM/WQTM)
    "breakingdefense.com",     # Breaking Defense — defense/aero (ITA/XAR)
    "aviationweek.com",        # Aviation Week — defense/aero, airlines
    "freightwaves.com",        # FreightWaves — transports (IYT/XTN)
    "uraniuminsider.com",      # Justin Huhn — uranium (URA/URNM)
    "benchmarkminerals.com",   # Benchmark Mineral Intelligence — lithium/rare-earth
    "goehringrozencwajg.com",  # G&R — natural resources / miners (GDX/XME/COPX)
    "hfir.com",                # HFI Research — oil & gas (XOP/OIH/FCG)
    "endpoints.news",          # Endpoints News — biotech (XBI/IBB)
    "endpts.com",              # Endpoints News (legacy domain, still redirects)
    "statnews.com",            # STAT — biotech/pharma (XBI/XPH)
    "timmermanreport.com",     # Luke Timmerman — biotech (XBI)
    "datacenterfrontier.com",  # Data Center Frontier — data-center REITs (DTCR)
    "heatmap.news",            # Heatmap — energy transition / grid (GRID/ICLN)
    "netinterest.co",          # Marc Rubinstein — banks/fintech (KRE/FINX)
]

# Government / public data sources
GOVERNMENT_DOMAINS: list[str] = [
    "fred.stlouisfed.org",
    "stlouisfed.org",
    "bls.gov",
    "eia.gov",
    "treasurydirect.gov",
    "treasury.gov",
    "federalreserve.gov",
    "bea.gov",
    "census.gov",
    "imf.org",
    "worldbank.org",
]

# Major financial press (mostly paywalled — fetched gracefully but often 401)
FINANCIAL_PRESS_DOMAINS: list[str] = [
    "ft.com",
    "bloomberg.com",
    "reuters.com",
    "wsj.com",
    "marketwatch.com",
    "barrons.com",
    "economist.com",
    "cnbc.com",
]

WHITELIST: list[str] = (
    SUBSTACK_DOMAINS
    + NEWSLETTER_PLATFORMS
    + AUTHOR_DOMAINS
    + GOVERNMENT_DOMAINS
    + FINANCIAL_PRESS_DOMAINS
)

# Link-wrapper / tracker domains. URLs on these hosts are not fetched
# directly — instead a HEAD redirect-resolve is run, and the final URL
# is re-checked against WHITELIST. Lets us follow newsletter platforms'
# tracking links through to whatever they actually point at.
TRACKER_DOMAINS: list[str] = [
    # beehiiv
    "mail.beehiiv.com",
    # ConvertKit / Kit
    "convertkit-mail.com",
    "convertkit-mail2.com",
    "convertkit-mail3.com",
    "convertkit-mail4.com",
    "ck.page",
    # Mailchimp
    "list-manage.com",
    "mailchi.mp",
    # SendGrid
    "sendgrid.net",
    "sendgrid.com",
    # HubSpot
    "hubspotemail.net",
    "hubspotlinks.com",
    # MailerLite
    "mailerlite.com",
    "mlsend.com",
    # Iterable / Customer.io / Klaviyo / Braze
    "iterable.com",
    "customeriomail.com",
    "klclick.com",
    "klclick1.com",
    "klclick2.com",
    "klclick3.com",
    "bemail.io",
    # Substack outbound link wrapper
    "substack.com/redirect",
    # Generic URL shorteners commonly used in newsletters
    "bit.ly",
    "t.co",
    "lnkd.in",
    "buff.ly",
    "ow.ly",
    "tinyurl.com",
]


def _host_matches(host: str, suffixes: list[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in suffixes)


def _parse_host(url: str) -> str:
    from urllib.parse import urlparse
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def is_whitelisted(url: str) -> bool:
    """True if `url`'s host ends with any whitelist suffix."""
    host = _parse_host(url)
    if not host:
        return False
    return _host_matches(host, WHITELIST)


def is_tracker(url: str) -> bool:
    """True if `url` is a known link-wrapper / tracker that should be
    resolved through redirects before being whitelisted."""
    host = _parse_host(url)
    if not host:
        return False
    return _host_matches(host, TRACKER_DOMAINS)
