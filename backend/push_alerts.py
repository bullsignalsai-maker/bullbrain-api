# backend/push_alerts.py

import datetime
import requests
from typing import Dict, Any

from firebase_admin import firestore

from backend.watchlist_snapshot import build_watchlist_snapshot


EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _send_expo_push(token: str, title: str, body: str, data: Dict[str, Any] | None = None):
    if not token:
        return {"success": False, "error": "missing token"}

    payload = {
        "to": token,
        "sound": "default",
        "title": title,
        "body": body,
        "data": data or {},
    }

    try:
        res = requests.post(
            EXPO_PUSH_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )

        return {
            "success": res.status_code in (200, 201),
            "status_code": res.status_code,
            "response": res.json() if res.text else None,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _is_meaningful_signal_change(old_signal: str | None, new_signal: str | None) -> bool:
    old_signal = (old_signal or "").upper()
    new_signal = (new_signal or "").upper()

    if not old_signal or not new_signal:
        return False

    if old_signal == new_signal:
        return False

    # Only alert for meaningful signal movement
    meaningful = {
        ("HOLD", "BUY"),
        ("HOLD", "SELL"),
        ("BUY", "HOLD"),
        ("SELL", "HOLD"),
        ("BUY", "SELL"),
        ("SELL", "BUY"),
    }

    return (old_signal, new_signal) in meaningful


def _confidence_text(confidence):
    try:
        return f"{float(confidence):.0f}%"
    except Exception:
        return "N/A"


def run_watchlist_push_alerts(max_users: int = 200) -> Dict[str, Any]:
    """
    Checks all users with Expo push tokens and compares their watchlist signal state.

    First run for each symbol creates baseline only.
    Later runs send notification only when signal changes.
    """

    db = firestore.client()

    users = list(db.collection("users").limit(max_users).stream())

    checked_users = 0
    checked_symbols = 0
    sent = 0
    baselined = 0
    skipped = 0
    errors = []

    for user_doc in users:
        user_id = user_doc.id
        user_data = user_doc.to_dict() or {}

        token = user_data.get("expoPushToken") or user_data.get("expo_push_token")

        if not token:
            skipped += 1
            continue

        checked_users += 1

        try:
            snapshot = build_watchlist_snapshot(user_id) or {}
            items = snapshot.get("items", []) or []
        except Exception as e:
            errors.append({"user_id": user_id, "stage": "snapshot", "error": str(e)})
            continue

        for item in items:
            symbol = (item.get("symbol") or "").upper()
            if not symbol:
                continue

            checked_symbols += 1

            bull = item.get("bullbrain") or {}
            new_signal = (
                item.get("hybridSignal")
                or bull.get("signal")
                or "HOLD"
            )
            confidence = (
                item.get("hybridScore")
                or bull.get("confidence")
                or None
            )
            quote = item.get("quote") or {}
            change_pct = quote.get("changePct")

            if change_pct is None:
                change_pct = item.get("changePct")

            try:
                change_pct_num = float(change_pct)
            except Exception:
                change_pct_num = None

            pattern = item.get("pattern") or {}
            pattern_name = pattern.get("name") if isinstance(pattern, dict) else None

            state_ref = (
                db.collection("users")
                  .document(user_id)
                  .collection("alert_state")
                  .document(symbol)
            )

            state_doc = state_ref.get()
            previous = state_doc.to_dict() if state_doc.exists else {}

            old_signal = previous.get("lastSignal")

            # First time: baseline only, do not notify
            if not old_signal:
                state_ref.set(
                    {
                        "symbol": symbol,
                        "lastSignal": new_signal,
                        "lastConfidence": confidence,
                        "lastPattern": pattern_name,
                        "lastCheckedAt": _now_iso(),
                        "lastAlertedAt": None,
                    },
                    merge=True,
                )
                baselined += 1
                continue

            if _is_meaningful_signal_change(old_signal, new_signal):
                title = "AlphaWise Signal Alert"
                body = f"{symbol} changed from {old_signal} → {new_signal}. Confidence: {_confidence_text(confidence)}."

                result = _send_expo_push(
                    token=token,
                    title=title,
                    body=body,
                    data={
                        "type": "watchlist_signal_change",
                        "symbol": symbol,
                        "oldSignal": old_signal,
                        "newSignal": new_signal,
                        "confidence": confidence,
                    },
                )

                if result.get("success"):
                    sent += 1
                    last_alerted_at = _now_iso()
                else:
                    errors.append(
                        {
                            "user_id": user_id,
                            "symbol": symbol,
                            "stage": "send_push",
                            "error": result,
                        }
                    )
                    last_alerted_at = previous.get("lastAlertedAt")

                state_ref.set(
                    {
                        "symbol": symbol,
                        "lastSignal": new_signal,
                        "lastConfidence": confidence,
                        "lastPattern": pattern_name,
                        "lastCheckedAt": _now_iso(),
                        "lastAlertedAt": last_alerted_at,
                        "previousSignal": old_signal,
                    },
                    merge=True,
                )
            else:
                state_ref.set(
                    {
                        "symbol": symbol,
                        "lastSignal": new_signal,
                        "lastConfidence": confidence,
                        "lastPattern": pattern_name,
                        "lastCheckedAt": _now_iso(),
                    },
                    merge=True,
                )
            # ---------------------------------------------------------
            # Big Move Alert
            # Max once per symbol per day per direction
            # ---------------------------------------------------------
            if change_pct_num is not None and abs(change_pct_num) >= 3.0:
                today = datetime.datetime.utcnow().date().isoformat()
                move_direction = "up" if change_pct_num > 0 else "down"

                last_big_move_date = previous.get("lastBigMoveDate")
                last_big_move_direction = previous.get("lastBigMoveDirection")

                already_alerted_today = (
                    last_big_move_date == today
                    and last_big_move_direction == move_direction
                )

                if not already_alerted_today:
                    move_word = "jumped" if move_direction == "up" else "dropped"
                    move_arrow = "▲" if move_direction == "up" else "▼"

                    title = "AlphaWise Watchlist Alert"
                    body = (
                        f"{symbol} {move_word} {move_arrow} "
                        f"{abs(change_pct_num):.2f}% today. Check signal and risk context."
                    )

                    result = _send_expo_push(
                        token=token,
                        title=title,
                        body=body,
                        data={
                            "type": "watchlist_big_move",
                            "symbol": symbol,
                            "changePct": change_pct_num,
                            "direction": move_direction,
                        },
                    )

                    if result.get("success"):
                        sent += 1
                        state_ref.set(
                            {
                                "symbol": symbol,
                                "lastBigMoveDate": today,
                                "lastBigMoveDirection": move_direction,
                                "lastBigMovePct": change_pct_num,
                                "lastBigMoveAlertedAt": _now_iso(),
                                "lastCheckedAt": _now_iso(),
                            },
                            merge=True,
                        )
                    else:
                        errors.append(
                            {
                                "user_id": user_id,
                                "symbol": symbol,
                                "stage": "big_move_push",
                                "error": result,
                            }
                        )
    return {
        "checked_users": checked_users,
        "checked_symbols": checked_symbols,
        "sent": sent,
        "baselined": baselined,
        "skipped_users_without_token": skipped,
        "errors": errors[:10],
        "finished_at": _now_iso(),  
    }

def run_portfolio_push_alerts(max_users: int = 200) -> Dict[str, Any]:
    """
    Portfolio-level push alerts.

    v1 supports:
    - Concentration risk alert
    - Big position gain/loss alert

    Future-ready for:
    - Portfolio day performance
    - Allocation shift
    - AI rebalancing
    - Concentration risk + loss combo
    """

    db = firestore.client()

    users = list(db.collection("users").limit(max_users).stream())

    checked_users = 0
    checked_positions = 0
    sent = 0
    skipped = 0
    errors = []

    today = datetime.datetime.utcnow().date().isoformat()

    for user_doc in users:
        user_id = user_doc.id
        user_data = user_doc.to_dict() or {}

        token = user_data.get("expoPushToken") or user_data.get("expo_push_token")

        if not token:
            skipped += 1
            continue

        checked_users += 1

        try:
            positions_snap = (
                db.collection("users")
                  .document(user_id)
                  .collection("portfolio")
                  .stream()
            )

            positions = []
            for doc in positions_snap:
                p = doc.to_dict() or {}
                symbol = (p.get("symbol") or doc.id or "").upper()
                shares = float(p.get("shares") or 0)
                avg_cost = float(p.get("avgCost") or p.get("avg_cost") or 0)

                if not symbol or shares <= 0:
                    continue

                stock_doc = (
                    db.collection("bullsignals_ai")
                      .document("stocks")
                      .collection("symbols")
                      .document(symbol)
                      .get()
                )

                if not stock_doc.exists:
                    continue

                stock = stock_doc.to_dict() or {}
                quote = stock.get("quote") or {}

                price = quote.get("price")
                prev_close = quote.get("prevClose")
                change_pct = quote.get("changePct")

                try:
                    price = float(price)
                except Exception:
                    price = avg_cost

                try:
                    prev_close = float(prev_close)
                except Exception:
                    prev_close = avg_cost

                try:
                    change_pct = float(change_pct)
                except Exception:
                    change_pct = 0.0

                curr_value = shares * price
                cost = shares * avg_cost
                gain = curr_value - cost
                gain_pct = (gain / cost * 100.0) if cost > 0 else 0.0
                today_gain = shares * (price - prev_close)

                positions.append({
                    "symbol": symbol,
                    "shares": shares,
                    "avg_cost": avg_cost,
                    "price": price,
                    "prev_close": prev_close,
                    "change_pct": change_pct,
                    "curr_value": curr_value,
                    "cost": cost,
                    "gain": gain,
                    "gain_pct": gain_pct,
                    "today_gain": today_gain,
                })

            if not positions:
                continue

            total_value = sum(p["curr_value"] for p in positions)

            if total_value <= 0:
                continue

            for p in positions:
                p["allocation_pct"] = (p["curr_value"] / total_value) * 100.0

            positions.sort(key=lambda x: x["allocation_pct"], reverse=True)

            largest = positions[0]
            checked_positions += len(positions)

            state_ref = (
                db.collection("users")
                  .document(user_id)
                  .collection("alert_state")
                  .document("_portfolio")
            )

            state_doc = state_ref.get()
            state = state_doc.to_dict() if state_doc.exists else {}

            # ---------------------------------------------------------
            # 1) Concentration Risk Alert
            # Max once per day
            # ---------------------------------------------------------
            largest_symbol = largest["symbol"]
            largest_alloc = largest["allocation_pct"]

            last_risk_date = state.get("lastConcentrationRiskDate")
            last_risk_symbol = state.get("lastConcentrationRiskSymbol")

            if largest_alloc >= 40.0:
                already_sent_risk = (
                    last_risk_date == today
                    and last_risk_symbol == largest_symbol
                )

                if not already_sent_risk:
                    title = "AlphaWise Portfolio Alert"
                    body = (
                        f"{largest_symbol} now makes up "
                        f"{largest_alloc:.0f}% of your portfolio. "
                        f"Concentration risk is elevated."
                    )

                    result = _send_expo_push(
                        token=token,
                        title=title,
                        body=body,
                        data={
                            "type": "portfolio_concentration_risk",
                            "symbol": largest_symbol,
                            "allocationPct": largest_alloc,
                        },
                    )

                    if result.get("success"):
                        sent += 1
                        state_ref.set({
                            "lastConcentrationRiskDate": today,
                            "lastConcentrationRiskSymbol": largest_symbol,
                            "lastConcentrationRiskPct": largest_alloc,
                            "lastConcentrationRiskAlertedAt": _now_iso(),
                            "lastCheckedAt": _now_iso(),
                        }, merge=True)
                    else:
                        errors.append({
                            "user_id": user_id,
                            "stage": "portfolio_concentration_push",
                            "error": result,
                        })

            # ---------------------------------------------------------
            # 2) Big Gain/Loss Alert per holding
            # Max once per symbol per day per direction
            # ---------------------------------------------------------
            for p in positions:
                symbol = p["symbol"]
                move_pct = p["change_pct"]

                if abs(move_pct) < 3.0:
                    continue

                pos_state_ref = (
                    db.collection("users")
                      .document(user_id)
                      .collection("alert_state")
                      .document(f"portfolio_{symbol}")
                )

                pos_state_doc = pos_state_ref.get()
                pos_state = pos_state_doc.to_dict() if pos_state_doc.exists else {}

                direction = "up" if move_pct > 0 else "down"

                already_sent_move = (
                    pos_state.get("lastPortfolioMoveDate") == today
                    and pos_state.get("lastPortfolioMoveDirection") == direction
                )

                if already_sent_move:
                    continue

                word = "up" if direction == "up" else "down"
                arrow = "▲" if direction == "up" else "▼"
                today_gain = p["today_gain"]

                title = "AlphaWise Portfolio Alert"
                body = (
                    f"{symbol} is {word} {arrow} {abs(move_pct):.2f}% today. "
                    f"Your position impact is "
                    f"{'+' if today_gain >= 0 else '-'}${abs(today_gain):,.2f}."
                )

                result = _send_expo_push(
                    token=token,
                    title=title,
                    body=body,
                    data={
                        "type": "portfolio_position_big_move",
                        "symbol": symbol,
                        "changePct": move_pct,
                        "todayGain": today_gain,
                        "direction": direction,
                    },
                )

                if result.get("success"):
                    sent += 1
                    pos_state_ref.set({
                        "symbol": symbol,
                        "lastPortfolioMoveDate": today,
                        "lastPortfolioMoveDirection": direction,
                        "lastPortfolioMovePct": move_pct,
                        "lastPortfolioMoveImpact": today_gain,
                        "lastPortfolioMoveAlertedAt": _now_iso(),
                        "lastCheckedAt": _now_iso(),
                    }, merge=True)
                else:
                    errors.append({
                        "user_id": user_id,
                        "symbol": symbol,
                        "stage": "portfolio_position_move_push",
                        "error": result,
                    })

            # Always update portfolio checked state
            state_ref.set({
                "lastCheckedAt": _now_iso(),
                "totalValue": total_value,
                "largestHolding": largest_symbol,
                "largestAllocationPct": largest_alloc,
                "positionCount": len(positions),
            }, merge=True)

        except Exception as e:
            errors.append({
                "user_id": user_id,
                "stage": "portfolio_alerts",
                "error": str(e),
            })

    return {
        "checked_users": checked_users,
        "checked_positions": checked_positions,
        "sent": sent,
        "skipped_users_without_token": skipped,
        "errors": errors[:10],
        "finished_at": _now_iso(),
    }