# sp500_list_optimized.py
# ------------------------------------------------------------
# Ultra-fast SP500 ticker detection module for BullSignalsAI
# ------------------------------------------------------------

import re

# ------------------------------------------------------------
# 1️⃣ SP500 Ticker List (imported from your app version)
# ------------------------------------------------------------
SP500_LIST = {
    "A","AAPL","ABBV","ABNB","ABT","ACGL","ACN","ADBE","ADI","ADM","ADP","ADSK",
    "AEE","AEP","AES","AFL","AIG","AIZ","AJG","AKAM","ALB","ALGN","ALL","ALLE",
    "AMAT","AMCR","AMD","AME","AMGN","AMP","AMT","AMZN","ANET","AON","AOS","APA",
    "APD","APH","APO","APTV","ARE","ATO","AVB","AVGO","AVY","AWK","AXON","AXP",
    "AZO","BA","BAC","BALL","BAX","BBY","BDX","BEN","BF.B","BG","BIIB","BK",
    "BKNG","BKR","BLDR","BLK","BMY","BR","BRK.B","BRO","BSX","BX","BXP","C","CAG",
    "CAH","CARR","CAT","CB","CBOE","CBRE","CCI","CCL","CDNS","CDW","CEG","CF",
    "CFG","CHD","CHRW","CHTR","CI","CINF","CL","CLX","CMCSA","CME","CMG","CMI",
    "CMS","CNC","CNP","COF","COIN","COO","COP","COR","COST","CPAY","CPB","CPRT",
    "CPT","CRL","CRM","CRWD","CSCO","CSGP","CSX","CTAS","CTRA","CTSH","CTVA",
    "CVS","CVX","CZR","D","DAL","DASH","DAY","DD","DDOG","DE","DECK","DELL","DG",
    "DGX","DHI","DHR","DIS","DLR","DLTR","DOC","DOV","DOW","DPZ","DRI","DTE","DUK",
    "DVA","DVN","DXCM","EA","EBAY","ECL","ED","EFX","EG","EIX","EL","ELV","EMN",
    "EMR","ENPH","EOG","EPAM","EQIX","EQR","EQT","ERIE","ES","ESS","ETN","ETR",
    "EVRG","EW","EXC","EXE","EXPD","EXPE","EXR","F","FANG","FAST","FCX","FDS",
    "FDX","FE","FFIV","FI","FICO","FIS","FITB","FOX","FOXA","FRT","FSLR","FTNT",
    "FTV","GD","GDDY","GE","GEHC","GEN","GEV","GILD","GIS","GL","GLW","GM","GNRC",
    "GOOG","GOOGL","GPC","GPN","GRMN","GS","GWW","HAL","HAS","HBAN","HCA","HD",
    "HIG","HII","HLT","HOLX","HON","HPE","HPQ","HRL","HSIC","HST","HSY","HUBB",
    "HUM","HWM","IBM","ICE","IDXX","IEX","IFF","INCY","INTC","INTU","INVH","IP",
    "IPG","IQV","IR","IRM","ISRG","IT","ITW","IVZ","J","JBHT","JBL","JCI","JKHY",
    "JNJ","JPM","K","KDP","KEY","KEYS","KHC","KIM","KKR","KLAC","KMB","KMI","KMX",
    "KO","KR","KVUE","L","LDOS","LEN","LH","LHX","LII","LIN","LKQ","LLY","LMT",
    "LNT","LOW","LRCX","LULU","LUV","LVS","LW","LYB","LYV","MA","MAA","MAR","MAS",
    "MCD","MCHP","MCK","MCO","MDLZ","MDT","MET","META","MGM","MHK","MKC","MKTX",
    "MLM","MMC","MMM","MNST","MO","MOH","MOS","MPC","MPWR","MRK","MRNA","MS",
    "MSCI","MSFT","MSI","MTB","MTCH","MTD","MU","NCLH","NDAQ","NDSN","NEE","NEM",
    "NFLX","NI","NKE","NOC","NOW","NRG","NSC","NTAP","NTRS","NUE","NVDA","NVR",
    "NWS","NWSA","NXPI","O","ODFL","OKE","OMC","ON","ORCL","ORLY","OTIS","OXY",
    "PANW","PAYC","PAYX","PCAR","PCG","PEG","PEP","PFE","PFG","PG","PGR","PH",
    "PHM","PKG","PLD","PLTR","PM","PNC","PNR","PNW","PODD","POOL","PPG","PPL",
    "PRU","PSA","PSKY","PSX","PTC","PWR","PYPL","QCOM","RCL","REG","REGN","RF",
    "RJF","RL","RMD","ROK","ROL","ROP","ROST","RSG","RTX","RVTY","SBAC","SBUX",
    "SCHW","SHW","SJM","SLB","SMCI","SNA","SNPS","SO","SOLV","SPG","SPGI","SRE",
    "STE","STLD","STT","STX","STZ","SW","SWK","SWKS","SYF","SYK","SYY","T","TAP",
    "TDG","TDY","TECH","TEL","TER","TFC","TGT","TJX","TKO","TMO","TMUS","TPL",
    "TPR","TRGP","TRMB","TROW","TRV","TSCO","TSLA","TSN","TT","TTD","TTWO","TXN",
    "TXT","TYL","UAL","UBER","UDR","UHS","ULTA","UNH","UNP","UPS","URI","USB",
    "V","VICI","VLO","VLTO","VMC","VRSK","VRSN","VRTX","VST","VTR","VTRS","VZ",
    "WAB","WAT","WBA","WBD","WDAY","WDC","WEC","WELL","WFC","WM","WMB","WMT","WRB",
    "WSM","WST","WTW","WY","WYNN","XEL","XOM","XYL","XYZ","YUM","ZBH","ZBRA","ZTS",
    "HOOD","COIN","SOFI","UPST","RBLX","SNAP","DKNG","PATH","AI","U","CRSP","ROKU",
    "ZM","PINS","ETSY","SE","NU","GRAB","TOST","GTLB","DUOL","ARM","RIVN","LCID","LI",
    "NIO","XPEV","BILI","PDD","BABA","TME","BIDU","JD","BEKE","SMCI","APP","IONQ","RKLB",
    "ASTS","SOUN","AMC","GME","BB","MARA","RIOT","CLSK","HUT","CIFR","BITF","PLTR","SHOP",
    "Z","CRWD","SNOW","NET","DDOG","TEAM","MDB","TTD","HUBS","OKTA","TWLO","DOCU","ESTC",
    "APPF","FRSH","NCNO","PCOR","BILL","ASAN","PATH","GTLB","DUOL","IONQ","RKLB","ASTS",
    "UP","HOOD","COIN","SOFI","UPST","RBLX","SNAP","DKNG","ROKU","ZM","PINS","ETSY","SE",
    "NU","GRAB","RIVN","LCID","LI","NIO","XPEV","BILI","PDD","BABA","TME","BIDU","JD","BEKE",
    "SMCI","APP","IONQ","RKLB","ASTS","SOUN","AMC","GME","BB","MARA","RIOT","OPEN","IONQ","QUBT","PACB",
    "EDIT","LAC","ASST","ATYR","RVPH","RXRX","BYND","BMNR"
}

# ------------------------------------------------------------
# 2️⃣ Ultra-fast indices
# ------------------------------------------------------------

# Length-indexed tickers (1–5 chars)
TICKERS_BY_LEN = {}
for ticker in SP500_LIST:
    TICKERS_BY_LEN.setdefault(len(ticker), set()).add(ticker)

# First-letter index (A–Z)
FIRST_LETTER_INDEX = {}
for t in SP500_LIST:
    FIRST_LETTER_INDEX.setdefault(t[0], set()).add(t)

# Precompiled regex for instant ticker extraction
TICKER_REGEX = re.compile(
    r"\b(" + "|".join(sorted(SP500_LIST, key=lambda x: -len(x))) + r")\b"
)

# ------------------------------------------------------------
# 3️⃣ Ticker detection helpers
# ------------------------------------------------------------

def is_valid_ticker(sym: str) -> bool:
    if not sym:
        return False
    return sym.upper() in SP500_LIST


def extract_ticker(text: str):
    """Ultra-fast ticker detection from headline or summary."""
    if not text:
        return None

    text = text.upper()

    # Regex match first (fastest)
    match = TICKER_REGEX.search(text)
    if match:
        return match.group(1)

    # Manual fallback (rare)
    words = re.findall(r"\b[A-Z]{1,5}\b", text)
    for w in words:
        if is_valid_ticker(w):
            return w

    return None

# ------------------------------------------------------------
# 4️⃣ Category detection
# ------------------------------------------------------------

CATEGORY_KEYWORDS = {
    "Earnings": ["earnings", "eps", "revenue", "guidance", "profit", "loss", "beat", "miss"],
    "Tech / AI": ["ai", "chip", "nvidia", "gpu", "machine learning", "cloud", "semiconductor"],
    "M&A": ["merger", "acquisition", "deal", "buyout", "takeover"],
    "Fed / Macro": ["inflation", "fed", "cpi", "ppi", "jobs", "treasury", "rates", "economic"],
}

def detect_category(text: str):
    t = text.lower()
    for cat, keys in CATEGORY_KEYWORDS.items():
        if any(k in t for k in keys):
            return cat
    return "General"
