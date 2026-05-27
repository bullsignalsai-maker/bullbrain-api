# backend/market_momentum.py

import datetime
from collections import defaultdict, Counter
from typing import Dict, Any, List

import firebase_admin
from firebase_admin import firestore

COL_ROOT = "bullsignals_ai"
LOOKBACK_SNAPSHOTS_DEFAULT = 12


def get_db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    return firestore.client()


def utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _today_utc_date() -> datetime.date:
    return datetime.datetime.now(datetime.timezone.utc).date()


def _safe_float(v, default=0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _round(v, n=2):
    try:
        return round(float(v), n)
    except Exception:
        return v


def _clean_symbol(symbol: Any) -> str:
    s = str(symbol or "").strip().upper()
    if len(s) % 2 == 0:
        half = len(s) // 2
        if s[:half] == s[half:]:
            s = s[:half]
    return s



def _label_for_positive(appearances: int, avg_change: float, latest_change: float) -> str:
    if appearances >= 4 and avg_change >= 4:
        return "Strong Continuation"
    if appearances >= 3 and latest_change >= 5:
        return "Momentum Spike"
    if appearances >= 3:
        return "3-Day Runner"
    if avg_change >= 2:
        return "Steady Strength"
    return "Positive Momentum"


def _label_for_negative(appearances: int, avg_change: float) -> str:
    if appearances >= 4:
        return "Repeated Pullback"
    if avg_change <= -3:
        return "Bearish Pressure"
    return "Pullback Watch"


def _label_for_momentum(
    appearances: int,
    avg_change: float,
    latest_change: float,
    net_move: float,
    positive_sessions: int,
    negative_sessions: int,
) -> str:
    if net_move >= 25 and positive_sessions >= 4 and negative_sessions <= 1:
        return "Elite Momentum"

    if net_move >= 15 and positive_sessions >= 3:
        return "Strong Continuation"

    if latest_change >= 8:
        return "Momentum Spike"

    if net_move >= 8:
        return "Steady Strength"

    if appearances >= 3:
        return "3-Day Runner"

    return "Positive Momentum"


def _is_true_pullback(
    net_move: float,
    latest_change: float,
    positive_sessions: int,
    negative_sessions: int,
) -> bool:
    return (
        negative_sessions > positive_sessions
        and net_move < 0
        and (
            negative_sessions >= 3
            or latest_change <= -3
            or net_move <= -4
        )
    )


def _recency_rank(date_value: Any) -> int:
    try:
        if not date_value:
            return 0

        d = datetime.date.fromisoformat(str(date_value)[:10])
        today = _today_utc_date()
        age = (today - d).days

        if age <= 0:
            return 5
        if age <= 2:
            return 4
        if age <= 5:
            return 3
        if age <= 10:
            return 2
        return 1
    except Exception:
        return 0
    
def _score_market_mover(
    appearances: int,
    avg_change: float,
    latest_change: float,
    net_move: float = 0.0,
    positive_sessions: int = 0,
    negative_sessions: int = 0,
) -> float:
    persistence_score = min(appearances, 8) * 7
    expansion_score = max(net_move, 0) * 1.8
    avg_strength_score = max(avg_change, 0) * 4.5
    latest_score = max(latest_change, 0) * 1.6

    clean_trend_bonus = 0
    if appearances >= 3 and negative_sessions == 0:
        clean_trend_bonus = 18
    elif appearances >= 4 and positive_sessions > negative_sessions:
        clean_trend_bonus = 10

    score = (
        persistence_score
        + expansion_score
        + avg_strength_score
        + latest_score
        + clean_trend_bonus
    )

    return min(100, round(score, 1))

def _dedupe_items_by_symbol(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []

    for item in items:
        sym = _clean_symbol(item.get("symbol"))
        if not sym or sym in seen:
            continue

        seen.add(sym)
        out.append(item)

    return out

def _score_pullback(appearances: int, avg_change: float, latest_change: float) -> float:
    score = appearances * 16 + abs(min(avg_change, 0)) * 4 + abs(min(latest_change, 0)) * 2
    return min(100, round(score, 1))


def _read_latest_docs(db, collection_name: str, limit: int) -> List[Dict[str, Any]]:
    """
    Reads latest available market-memory snapshot docs.
    Snapshot-based (not calendar-day based).
    Avoids Firestore index requirements.
    """

    docs = []

    snaps = (
        db.collection(COL_ROOT)
        .document("market_memory")
        .collection(collection_name)
        .stream()
    )

    for snap in snaps:
        if snap.exists:
            data = snap.to_dict() or {}
            data.setdefault("date", snap.id)
            docs.append(data)

    docs.sort(
        key=lambda d: str(d.get("date") or ""),
        reverse=True,
    )

    return docs[:limit]

def _read_daily_movers(db, lookback_snapshots: int) -> List[Dict[str, Any]]:
    return _read_latest_docs(db, "daily_movers", lookback_snapshots)


def _read_daily_alpha_intelligence(db, lookback_snapshots: int) -> List[Dict[str, Any]]:
    return _read_latest_docs(db, "daily_alpha_intelligence", lookback_snapshots)


def _read_alpha_watch_current(db) -> Dict[str, Any]:
    snap = db.collection(COL_ROOT).document("alpha_watch").get()
    return snap.to_dict() if snap.exists else {}


def _latest_quote_for_symbol(db, symbol: str) -> Dict[str, Any]:
    symbol = _clean_symbol(symbol)
    if not symbol:
        return {}

    snap = (
        db.collection(COL_ROOT)
        .document("quotes")
        .collection("symbols")
        .document(symbol)
        .get()
    )

    if snap.exists:
        return snap.to_dict() or {}

    stock_snap = (
        db.collection(COL_ROOT)
        .document("stocks")
        .collection("symbols")
        .document(symbol)
        .get()
    )

    if stock_snap.exists:
        stock = stock_snap.to_dict() or {}
        return stock.get("quote") or {}

    return {}


def _sparkline_from_rows(rows: List[Dict[str, Any]]) -> List[float]:
    ordered = sorted(rows, key=lambda r: str(r.get("date") or r.get("market_day") or ""))
    values = []

    for r in ordered:
        cp = r.get("changePct")
        if cp is None:
            cp = r.get("change_pct")
        try:
            values.append(round(float(cp), 2))
        except Exception:
            pass

    return values[-10:]


def _build_continuous_movers(daily_docs: List[Dict[str, Any]], lookback_snapshots: int):
    by_symbol = defaultdict(list)

    for doc in daily_docs:
        date = doc.get("date")

        for row in (doc.get("gainers") or []):
            sym = _clean_symbol(row.get("symbol"))
            if sym:
                by_symbol[sym].append({
                    **row,
                    "date": date,
                    "direction": "up",
                })

        for row in (doc.get("losers") or []):
            sym = _clean_symbol(row.get("symbol"))
            if sym:
                by_symbol[sym].append({
                    **row,
                    "date": date,
                    "direction": "down",
                })

    momentum_movers = []
    pullback_watch = []
    for sym, rows in by_symbol.items():
        if len(rows) < 2:
            continue

        rows = sorted(
            rows,
            key=lambda r: str(r.get("date") or ""),
            reverse=True,
        )

        latest = rows[0]

        positive_sessions = 0
        negative_sessions = 0

        changes = []

        for r in rows:
            cp = _safe_float(r.get("changePct"))

            if cp > 0:
                positive_sessions += 1
            elif cp < 0:
                negative_sessions += 1

            changes.append(cp)

        net_move = sum(changes)
        avg_change = net_move / len(changes)

        dominant_direction = (
            "up"
            if positive_sessions >= negative_sessions
            else "down"
        )

        payload = {
            "symbol": sym,
            "companyName": latest.get("company_name")
            or latest.get("companyName")
            or sym,

            "price": _round(latest.get("price")),
            "change": _round(latest.get("change")),
            "changePct": _round(latest.get("changePct"), 2),

            "direction": dominant_direction,

            "appearances": len(rows),
            "positiveSessions": positive_sessions,
            "negativeSessions": negative_sessions,

            "lookbackSnapshots": lookback_snapshots,

            "netMovePct": _round(net_move, 2),
            "avgMovePct": _round(avg_change, 2),

            "latestMovePct": _round(
                latest.get("changePct"),
                2,
            ),

            "sparkline": _sparkline_from_rows(rows),

            "quote_updated_at": latest.get("quote_updated_at"),

            "source": "daily_movers",
        }

        if dominant_direction == "up":
            payload["momentumScore"] = _score_market_mover(
                len(rows),
                avg_change,
                _safe_float(latest.get("changePct")),
                net_move,
                positive_sessions,
                negative_sessions,
            )

            payload["momentumLabel"] = _label_for_momentum(
                len(rows),
                avg_change,
                _safe_float(latest.get("changePct")),
                net_move,
                positive_sessions,
                negative_sessions,
            )
            is_real_momentum = (
                net_move >= 8
                or avg_change >= 2
                or (positive_sessions >= 4 and net_move > 4)
            )

            if not is_real_momentum:
                continue
            momentum_movers.append(payload)

        else:
            latest_change = _safe_float(latest.get("changePct"))

            if not _is_true_pullback(
                net_move,
                latest_change,
                positive_sessions,
                negative_sessions,
            ):
                continue

            payload["momentumScore"] = _score_pullback(
                len(rows),
                avg_change,
                latest_change,
            )

            payload["momentumLabel"] = _label_for_negative(
                len(rows),
                avg_change,
            )

          
            pullback_watch.append(payload)
    momentum_movers.sort(
        key=lambda x: (
            x.get("momentumScore", 0),
            _safe_float(x.get("netMovePct")),
            _safe_float(x.get("avgMovePct")),
            x.get("positiveSessions", 0),
            x.get("appearances", 0),
        ),
        reverse=True,
    )

    pullback_watch.sort(
        key=lambda x: (
            x.get("momentumScore", 0),
            abs(_safe_float(x.get("netMovePct"))),
            x.get("negativeSessions", 0),
            x.get("appearances", 0),
            abs(_safe_float(x.get("latestMovePct"))),
        ),
        reverse=True,
    )

    momentum_movers = _dedupe_items_by_symbol(momentum_movers)
    pullback_watch = _dedupe_items_by_symbol(pullback_watch)

    return momentum_movers[:20], pullback_watch[:12]


def _build_alpha_memory(alpha_docs: List[Dict[str, Any]], lookback_snapshots: int):
    by_symbol = defaultdict(list)
    sector_counter = Counter()
    catalyst_counter = Counter()
    session_counter = Counter()

    for doc in alpha_docs:
        date = doc.get("date")
        sessions = doc.get("sessions") or {}

        for session_name, session_payload in sessions.items():
            session_type = str(session_name or session_payload.get("session_type") or "").upper()
            session_counter[session_type] += 1

            for item in session_payload.get("items") or []:
                sym = _clean_symbol(item.get("symbol"))
                if not sym:
                    continue

                enriched = {
                    **item,
                    "symbol": sym,
                    "date": date,
                    "session_type": session_type,
                }

                by_symbol[sym].append(enriched)

                sector = str(item.get("sector") or "").strip()
                if sector and sector.lower() not in {"other", "unknown", "general", "mixed"}:
                    sector_counter[sector] += 1

                catalysts = str(item.get("primaryCatalysts") or "").replace(",", "|").split("|")
                for c in catalysts:
                    c = c.strip()
                    if c:
                        catalyst_counter[c] += 1

    alpha_items = []

    for sym, rows in by_symbol.items():
        rows = sorted(
            rows,
            key=lambda r: (
                str(r.get("date") or ""),
                str(r.get("generated_at") or ""),
            ),
            reverse=True,
        )

        latest = rows[0]
        scores = [_safe_float(r.get("grokAlphaPriorityScore")) for r in rows]
        avg_score = sum(scores) / len(scores) if scores else 0

        sessions = sorted(list({r.get("session_type") for r in rows if r.get("session_type")}))
        catalysts = []
        seen_catalysts = set()

        for r in rows:
            raw = str(r.get("primaryCatalysts") or "")
            for c in raw.replace(",", "|").split("|"):
                clean = c.strip()
                key = clean.lower()
                if clean and key not in seen_catalysts:
                    seen_catalysts.add(key)
                    catalysts.append(clean)

        alpha_items.append({
            "symbol": sym,
            "companyName": latest.get("companyName") or latest.get("company_name") or sym,
            "sector": latest.get("sector"),
            "moverQuality": latest.get("moverQuality"),
            "primaryCatalysts": catalysts[:4],
            "reason": latest.get("reason"),
            "riskLevel": latest.get("riskLevel"),
            "alphaAppearances": len(rows),
            "sessions": sessions,
            "lookbackSnapshots": lookback_snapshots,
            "avgAlphaScore": _round(avg_score, 1),
            "latestAlphaScore": _round(latest.get("grokAlphaPriorityScore"), 1),
            "lastSession": latest.get("session_type"),
            "lastSeenDate": latest.get("date"),
            "source": "daily_alpha_intelligence",
        })

    alpha_items.sort(
        key=lambda x: (
            x.get("alphaAppearances", 0),
            _recency_rank(x.get("lastSeenDate")),
            _safe_float(x.get("avgAlphaScore")),
            _safe_float(x.get("latestAlphaScore")),
        ),
        reverse=True,
    )

    return {
        "historicalAlphaMomentum": alpha_items[:20],
        "alphaBySymbol": {x["symbol"]: x for x in alpha_items},
        "topSector": sector_counter.most_common(1)[0][0] if sector_counter else "Mixed",
        "topCatalyst": catalyst_counter.most_common(1)[0][0] if catalyst_counter else "Mixed",
        "sessionCount": sum(session_counter.values()),
    }


def _score_confirmed(mover: Dict[str, Any], alpha: Dict[str, Any]) -> float:
    return min(
        100,
        round(
            _safe_float(mover.get("momentumScore")) * 0.55
            + min(_safe_float(alpha.get("alphaAppearances")) * 8, 30)
            + _safe_float(alpha.get("avgAlphaScore")) * 0.15,
            1,
        ),
    )


def _build_confirmed_momentum(momentum_movers: List[Dict[str, Any]], alpha_by_symbol: Dict[str, Any]):
    out = []

    for mover in momentum_movers:
        sym = _clean_symbol(mover.get("symbol"))
        alpha = alpha_by_symbol.get(sym)

        if not alpha:
            continue

        out.append({
            "symbol": sym,
            "companyName": mover.get("companyName") or alpha.get("companyName") or sym,
            "price": mover.get("price"),
            "change": mover.get("change"),
            "changePct": mover.get("changePct"),
            "direction": mover.get("direction"),
            "momentumScore": _score_confirmed(mover, alpha),
            "momentumLabel": "Confirmed Momentum",
            "appearances": {
                "dailyMovers": mover.get("appearances", 0),
                "alphaSessions": alpha.get("alphaAppearances", 0),
            },
            "avgMovePct": mover.get("avgMovePct"),
            "sparkline": mover.get("sparkline") or [],
            "sector": alpha.get("sector"),
            "moverQuality": alpha.get("moverQuality"),
            "primaryCatalysts": alpha.get("primaryCatalysts") or [],
            "reason": alpha.get("reason"),
            "riskLevel": alpha.get("riskLevel"),
            "lastSession": alpha.get("lastSession"),
            "source": "daily_movers+daily_alpha_intelligence",
        })

    out.sort(key=lambda x: x.get("momentumScore", 0), reverse=True)
    return out[:12]


def _score_alpha_watch_item(item: Dict[str, Any]) -> float:
    fs = item.get("factorScores") or {}

    score = (
        _safe_float(item.get("opportunityScore")) * 0.35
        + _safe_float(item.get("confidence")) * 0.15
        + _safe_float(fs.get("momentum")) * 0.20
        + _safe_float(fs.get("trend")) * 0.15
        + _safe_float(fs.get("volume")) * 0.10
        + max(_safe_float(item.get("changePct")), 0) * 0.8
    )

    return min(100, round(score, 1))


def _build_top_ai_setup(alpha_watch: Dict[str, Any]) -> Dict[str, Any]:
    items = alpha_watch.get("items") or []

    if not items:
        return {}

    ranked = []
    for item in items:
        fs = item.get("factorScores") or {}

        ranked.append({
            "symbol": _clean_symbol(item.get("symbol")),
            "companyName": item.get("companyName") or item.get("symbol"),
            "price": _round(item.get("price")),
            "change": _round(item.get("change")),
            "changePct": _round(item.get("changePct"), 2),
            "signal": item.get("signal") or "HOLD",
            "confidence": _round(item.get("confidence"), 2),
            "probUp": _round(item.get("probUp"), 4),
            "opportunityScore": _round(item.get("opportunityScore"), 2),
            "alphaScore": _round(item.get("score"), 2),
            "setupLabel": item.get("setupLabel") or item.get("pattern") or "AI Momentum Setup",
            "pattern": item.get("pattern"),
            "riskLevel": item.get("riskLevel"),
            "riskFlags": item.get("riskFlags") or [],
            "marketRegime": item.get("marketRegime"),
            "theme": item.get("theme"),
            "reason": item.get("reason"),
            "whyNow": item.get("whyNow") or [],
            "factorScores": {
                "momentum": _round(fs.get("momentum"), 1),
                "trend": _round(fs.get("trend"), 1),
                "volume": _round(fs.get("volume"), 1),
                "pattern": _round(fs.get("pattern"), 1),
                "bullbrain": _round(fs.get("bullbrain"), 1),
                "early_expansion": _round(fs.get("early_expansion"), 1),
            },
            "momentumScore": _score_alpha_watch_item(item),
            "computed_at": item.get("computed_at"),
            "quote_updated_at": item.get("quote_updated_at"),
            "source": "alpha_watch",
        })

    ranked.sort(
        key=lambda x: (
            x.get("momentumScore", 0),
            x.get("opportunityScore", 0),
            x.get("confidence", 0),
        ),
        reverse=True,
    )

    return ranked[0] if ranked else {}


def _enrich_latest_quotes(db, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []

    for item in items:
        sym = _clean_symbol(item.get("symbol"))
        quote = _latest_quote_for_symbol(db, sym)

        if quote:
            item = {
                **item,
                "price": _round(quote.get("price"), 2) if quote.get("price") is not None else item.get("price"),
                "change": _round(quote.get("change"), 2) if quote.get("change") is not None else item.get("change"),
                "changePct": _round(quote.get("changePct"), 2) if quote.get("changePct") is not None else item.get("changePct"),
                "quote_updated_at": quote.get("updated_at") or item.get("quote_updated_at"),
            }

        out.append(item)

    return out


def build_market_momentum_payload(
    lookback_snapshots: int = LOOKBACK_SNAPSHOTS_DEFAULT,
) -> Dict[str, Any]:
    db = get_db()

    daily_docs = _read_daily_movers(db, lookback_snapshots)
    alpha_memory_docs = _read_daily_alpha_intelligence(db, lookback_snapshots)
    alpha_watch = _read_alpha_watch_current(db)

    momentum_movers, pullback_watch = _build_continuous_movers(
    daily_docs,
    lookback_snapshots,
    )
    alpha_memory = _build_alpha_memory(
        alpha_memory_docs,
        lookback_snapshots,
    )

    confirmed = _build_confirmed_momentum(
        momentum_movers,
        alpha_memory.get("alphaBySymbol") or {},
    )

    top_ai_setup = _build_top_ai_setup(alpha_watch)

    momentum_movers = _enrich_latest_quotes(db, momentum_movers[:20])
    pullback_watch = _enrich_latest_quotes(db, pullback_watch[:12])
    confirmed = _enrich_latest_quotes(db, confirmed[:12])

    top_leader = dict(momentum_movers[0]) if momentum_movers else {}

    positive_count = len(momentum_movers)
    pullback_count = len(pullback_watch)
    repeated_alpha = len(alpha_memory.get("historicalAlphaMomentum") or [])

    all_avg = [abs(_safe_float(x.get("avgMovePct"))) for x in momentum_movers]
    avg_upside = sum(all_avg) / len(all_avg) if all_avg else 0

    raw_pulse_score = (
        positive_count * 1.2
        + pullback_count * 0.35
        + repeated_alpha * 0.9
        + avg_upside * 2.2
        + (_safe_float(top_ai_setup.get("momentumScore")) * 0.18 if top_ai_setup else 0)
    )

    pulse_score = min(95, round(raw_pulse_score, 1))

    market_regime = (
        top_ai_setup.get("marketRegime")
        or alpha_watch.get("market_regime")
        or "UNKNOWN"
    )

    ai_theme = str(top_ai_setup.get("theme") or "").strip()
    memory_theme = str(alpha_memory.get("topSector") or "").strip()

    if ai_theme and ai_theme.lower() not in {"other", "unknown", "general", "mixed"}:
        top_theme = ai_theme
    elif memory_theme and memory_theme.lower() not in {"other", "unknown", "general", "mixed"}:
        top_theme = memory_theme
    else:
        top_theme = "Mixed"

    payload = {
        "status": "ok",
        "screen": "market_momentum",
        "schema_version": "market_momentum_v2",
        "updated_at": utc_now_iso(),
        "lookbackSnapshots": lookback_snapshots,
        "memorySnapshotCount": len(daily_docs),

        "pulse": {
            "marketBias": "Bullish" if positive_count >= pullback_count else "Mixed",
            "marketRegime": market_regime,
            "momentumScore": pulse_score,
            "repeatedMovers": positive_count + pullback_count,
            "positiveContinuation": positive_count,
            "pullbackNames": pullback_count,
            "repeatedAlphaNames": repeated_alpha,
            "confirmedMomentumCount": len(confirmed),
            "avgUpsideMovePct": _round(avg_upside, 2),
            "topTheme": top_theme,
            "topCatalyst": alpha_memory.get("topCatalyst") or "Mixed",
            "summary": (
                "Repeated market movers and AI opportunity memory are aligning."
                if confirmed
                else "Market momentum is being tracked across movers and AI opportunity sessions."
            ),
        },

        "topAISetup": top_ai_setup,
        "topLeader": top_leader,
        "confirmedMomentum": confirmed,
        "continuousMovers": momentum_movers,
        "historicalAlphaMomentum": alpha_memory.get("historicalAlphaMomentum") or [],
        "pullbackWatch": pullback_watch,

        "momentumMovers": momentum_movers,
        "aiWatchlistMomentum": alpha_memory.get("historicalAlphaMomentum") or [],
    }

    return payload


def save_market_momentum_screen(
    lookback_snapshots: int = LOOKBACK_SNAPSHOTS_DEFAULT,
    window_days: int | None = None,  # backward compatibility
) -> Dict[str, Any]:
    db = get_db()

    if window_days is not None:
        lookback_snapshots = window_days

    payload = build_market_momentum_payload(
        lookback_snapshots=lookback_snapshots,
    )

    ref = (
        db.collection(COL_ROOT)
        .document("screen_cache")
        .collection("market")
        .document("momentum")
    )

    ref.set(payload)
    return payload


def get_market_momentum_screen() -> Dict[str, Any]:
    db = get_db()

    ref = (
        db.collection(COL_ROOT)
        .document("screen_cache")
        .collection("market")
        .document("momentum")
    )

    snap = ref.get()

    if not snap.exists:
        payload = save_market_momentum_screen()
        payload["generated_on_demand"] = True
        return payload

    data = snap.to_dict() or {}
    data["status"] = data.get("status") or "ok"
    return data