# backend/market_helpers.py


def _get_market_overview_quick():
    """
    Lightweight market overview:
    - SPY as proxy for S&P 500
    - VIX index
    - Synthetic Fear & Greed from VIX
    """
    overview = {
        "sp500_change": 0.0,
        "vix": 15.0,
        "fearGreed": {"value": 50, "label": "Neutral"},
        "risk_level": "Moderate Risk",
    }

    try:
        # --- S&P proxy via SPY ETF ---
        spy_quote = backend_fetch_quote("SPY") or {}
        change_pct = spy_quote.get("changePct")
        if change_pct is None:
            price = spy_quote.get("price")
            prev = spy_quote.get("prevClose")
            if price and prev and prev != 0:
                change_pct = ((price - prev) / prev) * 100.0
            else:
                change_pct = 0.0
        overview["sp500_change"] = float(change_pct)

    except Exception as e:
        print("SPY quote error:", e)

    try:
        # --- VIX index ---
        vix_quote = backend_fetch_quote("^VIX") or backend_fetch_quote("VIX") or {}
        vix_price = vix_quote.get("price") or vix_quote.get("close") or 15.0
        overview["vix"] = float(vix_price)
    except Exception as e:
        print("VIX quote error:", e)

    # --- Synthetic Fear & Greed from VIX ---
    vix_val = overview["vix"]
    fg_value = 50
    fg_label = "Neutral"

    if vix_val < 14:
        fg_value = 70
        fg_label = "Greed"
    elif vix_val < 18:
        fg_value = 55
        fg_label = "Slight Greed"
    elif vix_val < 22:
        fg_value = 45
        fg_label = "Slight Fear"
    elif vix_val < 28:
        fg_value = 35
        fg_label = "Fear"
    else:
        fg_value = 25
        fg_label = "Extreme Fear"

    overview["fearGreed"] = {"value": fg_value, "label": fg_label}
    overview["risk_level"] = _compute_risk_from_stats(
        overview["sp500_change"], vix_val, fg_value
    )

    return overview

    pass


def _analyze_headline_sentiment_py(headlines):
    bullish_words = [
        "gain", "gains", "rise", "rises", "soar", "soars",
        "beat", "beats", "growth", "surge", "surges",
        "optimism", "rebound", "rebounds", "strong",
        "rally", "record high", "expands", "up", "advance",
        "higher", "jumps", "spikes"
    ]

    bearish_words = [
        "drop", "drops", "fall", "falls", "slip", "plunge", "plunges",
        "loss", "losses", "slowdown", "decline", "declines",
        "cut", "cuts", "layoff", "layoffs", "weak", "selloff",
        "tumbles", "down", "pressure", "warning", "downgrade",
        "guidance cut"
    ]

    results = []
    for h in headlines:
        title = _clean_text_py(h or "")
        lower = title.lower()

        tag = "⚖️"  # default neutral
        if any(w in lower for w in bullish_words):
            tag = "📈"
        if any(w in lower for w in bearish_words):
            tag = "📉"

        results.append({"title": title, "tag": tag})

    return results


FALLBACK_SENTENCES = {
    "bullish": [
        "Markets show improving breadth across key sectors.",
        "Investor risk appetite firms during the early session.",
        "Equities strengthen as buying momentum builds.",
        "Positive flows support upside stability.",
        "Growth stocks continue their leadership trend.",
    ],
    "neutral": [
        "Markets remain steady as traders await key catalysts.",
        "Equities trade sideways amid balanced sentiment.",
        "Mixed sector rotation keeps indexes stable.",
        "Traders monitor macro signals for direction.",
        "Volatility holds near average levels.",
    ],
    "bearish": [
        "Market participants show caution amid uncertainty.",
        "Risk-off flows build as volatility edges higher.",
        "Selling pressure emerges in selective sectors.",
        "Equities pull back as momentum cools.",
        "Weakness appears across multiple asset groups.",
    ],
}


def _ensure_five(items, key):
    arr = list(items)
    if len(arr) >= 5:
        return arr[:5]
    needed = 5 - len(arr)
    return arr + FALLBACK_SENTENCES[key][:needed]


def _compute_risk_from_stats(sp500_change: float, vix: float, fg_value: int) -> str:
    # Same logic as before
    if vix < 15 and fg_value > 60:
        return "Low Risk"
    if vix > 20 or fg_value < 30:
        return "High Risk"
    return "Moderate Risk"
    """

def _clean_text_py(s: str) -> str:
    if not s:
        return ""
    # Basic cleaning similar to frontend
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _analyze_headline_sentiment_py(headlines):
    bullish_words = [
        "gain", "gains", "rise", "rises", "soar", "soars",
        "beat", "beats", "growth", "surge", "surges",
        "optimism", "rebound", "rebounds", "strong",
        "rally", "record high", "expands", "up", "advance",
        "higher", "jumps", "spikes"
    ]

    bearish_words = [
        "drop", "drops", "fall", "falls", "slip", "plunge", "plunges",
        "loss", "losses", "slowdown", "decline", "declines",
        "cut", "cuts", "layoff", "layoffs", "weak", "selloff",
        "tumbles", "down", "pressure", "warning", "downgrade",
        "guidance cut"
    ]

    results = []
    for h in headlines:
        title = _clean_text_py(h or "")
        lower = title.lower()

        tag = "⚖️"  # default neutral
        if any(w in lower for w in bullish_words):
            tag = "📈"
        if any(w in lower for w in bearish_words):
            tag = "📉"

        results.append({"title": title, "tag": tag})

    return results


FALLBACK_SENTENCES = {
    "bullish": [
        "Markets show improving breadth across key sectors.",
        "Investor risk appetite firms during the early session.",
        "Equities strengthen as buying momentum builds.",
        "Positive flows support upside stability.",
        "Growth stocks continue their leadership trend.",
    ],
    "neutral": [
        "Markets remain steady as traders await key catalysts.",
        "Equities trade sideways amid balanced sentiment.",
        "Mixed sector rotation keeps indexes stable.",
        "Traders monitor macro signals for direction.",
        "Volatility holds near average levels.",
    ],
    "bearish": [
        "Market participants show caution amid uncertainty.",
        "Risk-off flows build as volatility edges higher.",
        "Selling pressure emerges in selective sectors.",
        "Equities pull back as momentum cools.",
        "Weakness appears across multiple asset groups.",
    ],
}


def _ensure_five(items, key):
    arr = list(items)
    if len(arr) >= 5:
        return arr[:5]
    needed = 5 - len(arr)
    return arr + FALLBACK_SENTENCES[key][:needed]


def _compute_risk_from_stats(sp500_change: float, vix: float, fg_value: int) -> str:
    # Same logic as before
    if vix < 15 and fg_value > 60:
        return "Low Risk"
    if vix > 20 or fg_value < 30:
        return "High Risk"
    return "Moderate Risk"


def _get_market_overview_quick():
    """
    Lightweight market overview:
    - SPY as proxy for S&P 500
    - VIX index
    - Synthetic Fear & Greed from VIX
    """
    overview = {
        "sp500_change": 0.0,
        "vix": 15.0,
        "fearGreed": {"value": 50, "label": "Neutral"},
        "risk_level": "Moderate Risk",
    }

    try:
        # --- S&P proxy via SPY ETF ---
        spy_quote = backend_fetch_quote("SPY") or {}
        change_pct = spy_quote.get("changePct")
        if change_pct is None:
            price = spy_quote.get("price")
            prev = spy_quote.get("prevClose")
            if price and prev and prev != 0:
                change_pct = ((price - prev) / prev) * 100.0
            else:
                change_pct = 0.0
        overview["sp500_change"] = float(change_pct)

    except Exception as e:
        print("SPY quote error:", e)

    try:
        # --- VIX index ---
        vix_quote = backend_fetch_quote("^VIX") or backend_fetch_quote("VIX") or {}
        vix_price = vix_quote.get("price") or vix_quote.get("close") or 15.0
        overview["vix"] = float(vix_price)
    except Exception as e:
        print("VIX quote error:", e)

    # --- Synthetic Fear & Greed from VIX ---
    vix_val = overview["vix"]
    fg_value = 50
    fg_label = "Neutral"

    if vix_val < 14:
        fg_value = 70
        fg_label = "Greed"
    elif vix_val < 18:
        fg_value = 55
        fg_label = "Slight Greed"
    elif vix_val < 22:
        fg_value = 45
        fg_label = "Slight Fear"
    elif vix_val < 28:
        fg_value = 35
        fg_label = "Fear"
    else:
        fg_value = 25
        fg_label = "Extreme Fear"

    overview["fearGreed"] = {"value": fg_value, "label": fg_label}
    overview["risk_level"] = _compute_risk_from_stats(
        overview["sp500_change"], vix_val, fg_value
    )

    return overview

@app.get("/market-pulse")
def market_pulse():
    """
    Firestore read-only endpoint.
    Cron job is the single writer.
    """
    try:
        db = firestore.client()
        doc = db.collection("bullsignals_ai").document("market_pulse").get()

        if not doc.exists:
            return {
                "highlights_grouped": {
                    "bullish": [],
                    "neutral": [],
                    "bearish": [],
                },
                "news_grouped": {
                    "today": [],
                    "yesterday": [],
                    "week": [],
                    "older": [],
                },
                "updated_at": None,
            }

        return doc.to_dict()

    except Exception as e:
        backend.log(f"[market-pulse] Firestore read error: {e}")
        return {
            "highlights_grouped": {
                "bullish": [],
                "neutral": [],
                "bearish": [],
            },
            "news_grouped": {
                "today": [],
                "yesterday": [],
                "week": [],
                "older": [],
            },
            "updated_at": None,
        }

