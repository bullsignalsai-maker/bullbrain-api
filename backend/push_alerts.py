# backend/push_alerts.py

import datetime
import os
import requests
from typing import Dict, Any
from firebase_admin import firestore

from backend.watchlist_snapshot import build_watchlist_snapshot


EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
API_BASE_URL = os.getenv("API_BASE_URL", "https://bullbrain-api.onrender.com")


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


def _get_notification_prefs(db, user_id: str) -> Dict[str, bool]:
    """
    Reads user notification preferences.
    Defaults to ON if preferences doc does not exist.
    """
    try:
        snap = (
            db.collection("users")
              .document(user_id)
              .collection("preferences")
              .document("notifications")
              .get()
        )

        data = snap.to_dict() if snap.exists else {}

        return {
            "enabled": data.get("enabled", True),
            "watchlist": data.get("watchlist", True),
            "portfolio": data.get("portfolio", True),
            "crypto": data.get("crypto", True),
        }
    except Exception:
        return {
            "enabled": True,
            "watchlist": True,
            "portfolio": True,
            "crypto": True,
        }
    
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
        prefs = _get_notification_prefs(db, user_id)

        if not prefs.get("enabled", True):
            skipped += 1
            continue

        if not prefs.get("watchlist", True):
            skipped += 1
            continue
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

    Supports:
    1) Concentration risk alert
    2) Big position gain/loss alert
    3) Portfolio daily performance alert
    4) Allocation shift alert
    5) AI rebalancing alert
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
        prefs = _get_notification_prefs(db, user_id)

        if not prefs.get("enabled", True):
            skipped += 1
            continue

        if not prefs.get("portfolio", True):
            skipped += 1
            continue

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

                try:
                    shares = float(p.get("shares") or 0)
                    avg_cost = float(p.get("avgCost") or p.get("avg_cost") or 0)
                except Exception:
                    continue

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

                try:
                    price = float(quote.get("price") or avg_cost)
                except Exception:
                    price = avg_cost

                try:
                    prev_close = float(quote.get("prevClose") or avg_cost)
                except Exception:
                    prev_close = avg_cost

                try:
                    change_pct = float(quote.get("changePct") or 0)
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

            total_today_gain = sum(p["today_gain"] for p in positions)
            portfolio_day_pct = (total_today_gain / total_value) * 100.0

            for p in positions:
                p["allocation_pct"] = (p["curr_value"] / total_value) * 100.0

            positions.sort(key=lambda x: x["allocation_pct"], reverse=True)

            largest = positions[0]
            largest_symbol = largest["symbol"]
            largest_alloc = largest["allocation_pct"]

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
            if largest_alloc >= 40.0:
                already_sent_risk = (
                    state.get("lastConcentrationRiskDate") == today
                    and state.get("lastConcentrationRiskSymbol") == largest_symbol
                )

                if not already_sent_risk:
                    result = _send_expo_push(
                        token=token,
                        title="AlphaWise Portfolio Alert",
                        body=(
                            f"{largest_symbol} now makes up "
                            f"{largest_alloc:.0f}% of your portfolio. "
                            f"Concentration risk is elevated."
                        ),
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
            # 6) Concentration Risk + Loss Combo Alert
            # Largest holding is heavily concentrated AND down today
            # Max once per day per symbol
            # ---------------------------------------------------------
            try:
                largest_change_pct = float(largest.get("change_pct") or 0)

                already_sent_combo = (
                    state.get("lastRiskLossComboDate") == today
                    and state.get("lastRiskLossComboSymbol") == largest_symbol
                )

                if (
                    largest_alloc >= 40.0
                    and largest_change_pct <= -2.0
                    and not already_sent_combo
                ):
                    result = _send_expo_push(
                        token=token,
                        title="AlphaWise Risk Alert",
                        body=(
                            f"{largest_symbol} is {largest_alloc:.0f}% of your portfolio "
                            f"and down ▼ {abs(largest_change_pct):.2f}% today. "
                            f"Risk is elevated."
                        ),
                        data={
                            "type": "portfolio_risk_loss_combo",
                            "symbol": largest_symbol,
                            "allocationPct": largest_alloc,
                            "changePct": largest_change_pct,
                        },
                    )

                    if result.get("success"):
                        sent += 1
                        state_ref.set({
                            "lastRiskLossComboDate": today,
                            "lastRiskLossComboSymbol": largest_symbol,
                            "lastRiskLossComboAllocationPct": largest_alloc,
                            "lastRiskLossComboChangePct": largest_change_pct,
                            "lastRiskLossComboAlertedAt": _now_iso(),
                            "lastCheckedAt": _now_iso(),
                        }, merge=True)
                    else:
                        errors.append({
                            "user_id": user_id,
                            "symbol": largest_symbol,
                            "stage": "portfolio_risk_loss_combo_push",
                            "error": result,
                        })

            except Exception as e:
                errors.append({
                    "user_id": user_id,
                    "stage": "portfolio_risk_loss_combo",
                    "error": str(e),
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

                result = _send_expo_push(
                    token=token,
                    title="AlphaWise Portfolio Alert",
                    body=(
                        f"{symbol} is {word} {arrow} {abs(move_pct):.2f}% today. "
                        f"Your position impact is "
                        f"{'+' if today_gain >= 0 else '-'}${abs(today_gain):,.2f}."
                    ),
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

            # ---------------------------------------------------------
            # 3) Portfolio Daily Performance Alert
            # Max once per day per direction
            # ---------------------------------------------------------
            if abs(portfolio_day_pct) >= 2.0:
                direction = "up" if portfolio_day_pct > 0 else "down"

                already_sent_day_alert = (
                    state.get("lastPortfolioDayAlertDate") == today
                    and state.get("lastPortfolioDayDirection") == direction
                )

                if not already_sent_day_alert:
                    body = (
                        f"Your portfolio is {'up ▲' if direction == 'up' else 'down ▼'} "
                        f"{abs(portfolio_day_pct):.2f}% today "
                        f"({'+' if total_today_gain >= 0 else '-'}${abs(total_today_gain):,.2f})."
                    )

                    result = _send_expo_push(
                        token=token,
                        title="AlphaWise Portfolio Update",
                        body=body,
                        data={
                            "type": "portfolio_daily_performance",
                            "portfolioDayPct": portfolio_day_pct,
                            "todayGain": total_today_gain,
                            "direction": direction,
                        },
                    )

                    if result.get("success"):
                        sent += 1
                        state_ref.set({
                            "lastPortfolioDayAlertDate": today,
                            "lastPortfolioDayDirection": direction,
                            "lastPortfolioDayPct": portfolio_day_pct,
                            "lastPortfolioDayGain": total_today_gain,
                            "lastPortfolioDayAlertedAt": _now_iso(),
                            "lastCheckedAt": _now_iso(),
                        }, merge=True)
                    else:
                        errors.append({
                            "user_id": user_id,
                            "stage": "portfolio_daily_performance_push",
                            "error": result,
                        })

            # ---------------------------------------------------------
            # 4) Allocation Shift Alert
            # Max once per symbol per day
            # ---------------------------------------------------------
            for p in positions:
                symbol = p["symbol"]
                current_alloc = p["allocation_pct"]

                alloc_state_ref = (
                    db.collection("users")
                      .document(user_id)
                      .collection("alert_state")
                      .document(f"allocation_{symbol}")
                )

                alloc_state_doc = alloc_state_ref.get()
                alloc_state = alloc_state_doc.to_dict() if alloc_state_doc.exists else {}

                previous_alloc = alloc_state.get("lastAllocationPct")

                if previous_alloc is None:
                    alloc_state_ref.set({
                        "symbol": symbol,
                        "lastAllocationPct": current_alloc,
                        "lastCheckedAt": _now_iso(),
                    }, merge=True)
                    continue

                try:
                    previous_alloc = float(previous_alloc)
                except Exception:
                    previous_alloc = current_alloc

                alloc_change = current_alloc - previous_alloc

                already_sent_alloc_alert = (
                    alloc_state.get("lastAllocationAlertDate") == today
                )

                if abs(alloc_change) >= 7.0 and not already_sent_alloc_alert:
                    direction_word = "increased" if alloc_change > 0 else "decreased"

                    result = _send_expo_push(
                        token=token,
                        title="AlphaWise Allocation Alert",
                        body=(
                            f"{symbol} allocation {direction_word} from "
                            f"{previous_alloc:.0f}% to {current_alloc:.0f}% of your portfolio."
                        ),
                        data={
                            "type": "portfolio_allocation_shift",
                            "symbol": symbol,
                            "previousAllocationPct": previous_alloc,
                            "currentAllocationPct": current_alloc,
                            "changePctPoints": alloc_change,
                        },
                    )

                    if result.get("success"):
                        sent += 1
                        alloc_state_ref.set({
                            "symbol": symbol,
                            "lastAllocationPct": current_alloc,
                            "previousAllocationPct": previous_alloc,
                            "lastAllocationChangePctPoints": alloc_change,
                            "lastAllocationAlertDate": today,
                            "lastAllocationAlertedAt": _now_iso(),
                            "lastCheckedAt": _now_iso(),
                        }, merge=True)
                    else:
                        errors.append({
                            "user_id": user_id,
                            "symbol": symbol,
                            "stage": "portfolio_allocation_shift_push",
                            "error": result,
                        })
                else:
                    alloc_state_ref.set({
                        "symbol": symbol,
                        "lastAllocationPct": current_alloc,
                        "lastCheckedAt": _now_iso(),
                    }, merge=True)

            # ---------------------------------------------------------
            # 5) AI Rebalancing Alert
            # Max once per day
            # ---------------------------------------------------------
            try:
                last_ai_alert_date = state.get("lastAIRebalanceAlertDate")

                if last_ai_alert_date != today:
                    ai_url = (
                        f"{API_BASE_URL}/portfolio-ai-insight/{largest_symbol}"
                        f"?allocation_pct={largest_alloc}"
                        f"&portfolio_total_value={total_value}"
                    )

                    ai_res = requests.get(ai_url, timeout=8)

                    if ai_res.ok:
                        ai_json = ai_res.json() or {}

                        risk = (ai_json.get("risk") or "").lower()
                        rebalancing = (ai_json.get("rebalancing") or "").lower()

                        should_alert = (
                            "high" in risk
                            or "rebalance" in rebalancing
                            or "reduce" in rebalancing
                            or "concentration" in rebalancing
                        )

                        if should_alert:
                            result = _send_expo_push(
                                token=token,
                                title="AlphaWise AI Insight",
                                body=(
                                    "Your portfolio shows elevated risk. "
                                    "AI suggests reviewing your allocation."
                                ),
                                data={
                                    "type": "portfolio_ai_rebalance",
                                    "symbol": largest_symbol,
                                    "risk": risk,
                                },
                            )

                            if result.get("success"):
                                sent += 1
                                state_ref.set({
                                    "lastAIRebalanceAlertDate": today,
                                    "lastAIRebalanceSymbol": largest_symbol,
                                    "lastAIRebalanceRisk": risk,
                                    "lastAIRebalanceAlertedAt": _now_iso(),
                                    "lastCheckedAt": _now_iso(),
                                }, merge=True)
                            else:
                                errors.append({
                                    "user_id": user_id,
                                    "stage": "portfolio_ai_rebalance_push",
                                    "error": result,
                                })

            except Exception as e:
                errors.append({
                    "user_id": user_id,
                    "stage": "portfolio_ai_rebalance",
                    "error": str(e),
                })

            # Always update portfolio checked state
            state_ref.set({
                "lastCheckedAt": _now_iso(),
                "totalValue": total_value,
                "totalTodayGain": total_today_gain,
                "portfolioDayPct": portfolio_day_pct,
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

def run_crypto_market_alerts(max_users: int = 200) -> Dict[str, Any]:
    """
    Crypto market movement alerts.

    Tracks major crypto movers from homescreen_snapshot carousel:
    BTC, ETH, SOL, XRP, DOGE

    Alert rules:
    - BTC / ETH: +/- 3%
    - SOL / XRP / DOGE: +/- 5%

    Anti-spam:
    - Max once per crypto per direction per day.
    """

    db = firestore.client()

    users = list(db.collection("users").limit(max_users).stream())

    checked_users = 0
    checked_crypto = 0
    sent = 0
    skipped = 0
    errors = []

    today = datetime.datetime.utcnow().date().isoformat()

    try:
        snap = (
            db.collection("bullsignals_ai")
              .document("homescreen_snapshot")
              .get()
        )

        if not snap.exists:
            return {
                "checked_users": 0,
                "checked_crypto": 0,
                "sent": 0,
                "skipped_users_without_token": 0,
                "errors": [{"stage": "crypto_snapshot", "error": "homescreen_snapshot not found"}],
                "finished_at": _now_iso(),
            }

        data = snap.to_dict() or {}
        carousel = data.get("carousel") or []

        crypto_card = None
        for card in carousel:
            if isinstance(card, dict) and card.get("id") == "crypto":
                crypto_card = card
                break

        if not crypto_card:
            return {
                "checked_users": 0,
                "checked_crypto": 0,
                "sent": 0,
                "skipped_users_without_token": 0,
                "errors": [{"stage": "crypto_card", "error": "crypto card not found"}],
                "finished_at": _now_iso(),
            }

        crypto_items = crypto_card.get("items") or []

    except Exception as e:
        return {
            "checked_users": 0,
            "checked_crypto": 0,
            "sent": 0,
            "skipped_users_without_token": 0,
            "errors": [{"stage": "crypto_load", "error": str(e)}],
            "finished_at": _now_iso(),
        }

    def parse_crypto_symbol(label: str) -> str:
        label = (label or "").upper()

        if "BTC" in label or "BITCOIN" in label:
            return "BTC"
        if "ETH" in label or "ETHEREUM" in label:
            return "ETH"
        if "SOL" in label or "SOLANA" in label:
            return "SOL"
        if "XRP" in label:
            return "XRP"
        if "DOGE" in label or "DOGECOIN" in label:
            return "DOGE"

        return label.strip()

    def parse_pct(value) -> float | None:
        if value is None:
            return None

        try:
            if isinstance(value, (int, float)):
                return float(value)

            s = str(value).replace("%", "").replace("+", "").strip()
            return float(s)
        except Exception:
            return None

    crypto_alerts = []

    for item in crypto_items:
        if not isinstance(item, dict):
            continue

        label = item.get("label") or item.get("symbol") or ""
        symbol = parse_crypto_symbol(label)

        if symbol not in {"BTC", "ETH", "SOL", "XRP", "DOGE"}:
            continue

        pct = (
            item.get("changePct")
            or item.get("change_pct")
            or item.get("pct")
            or item.get("value")
        )

        change_pct = parse_pct(pct)

        if change_pct is None:
            continue

        threshold = 3.0 if symbol in {"BTC", "ETH"} else 5.0

        if abs(change_pct) < threshold:
            continue

        direction = "up" if change_pct > 0 else "down"

        crypto_alerts.append({
            "symbol": symbol,
            "change_pct": change_pct,
            "direction": direction,
            "threshold": threshold,
        })

    if not crypto_alerts:
        return {
            "checked_users": 0,
            "checked_crypto": len(crypto_items),
            "sent": 0,
            "skipped_users_without_token": 0,
            "errors": [],
            "finished_at": _now_iso(),
        }

    for user_doc in users:
        user_id = user_doc.id
        user_data = user_doc.to_dict() or {}

        token = user_data.get("expoPushToken") or user_data.get("expo_push_token")

        if not token:
            skipped += 1
            continue

        checked_users += 1
        prefs = _get_notification_prefs(db, user_id)

        if not prefs.get("enabled", True):
            skipped += 1
            continue

        if not prefs.get("crypto", True):
            skipped += 1
            continue
        for alert in crypto_alerts:
            symbol = alert["symbol"]
            change_pct = alert["change_pct"]
            direction = alert["direction"]

            checked_crypto += 1

            state_ref = (
                db.collection("users")
                  .document(user_id)
                  .collection("alert_state")
                  .document(f"crypto_{symbol}")
            )

            state_doc = state_ref.get()
            state = state_doc.to_dict() if state_doc.exists else {}

            already_sent = (
                state.get("lastCryptoAlertDate") == today
                and state.get("lastCryptoAlertDirection") == direction
            )

            if already_sent:
                continue

            word = "rising" if direction == "up" else "dropping"
            arrow = "▲" if direction == "up" else "▼"

            result = _send_expo_push(
                token=token,
                title="AlphaWise Crypto Alert",
                body=(
                    f"{symbol} is {word} {arrow} {abs(change_pct):.2f}% today. "
                    f"Crypto market movement is active."
                ),
                data={
                    "type": "crypto_market_move",
                    "symbol": symbol,
                    "changePct": change_pct,
                    "direction": direction,
                },
            )

            if result.get("success"):
                sent += 1
                state_ref.set({
                    "symbol": symbol,
                    "lastCryptoAlertDate": today,
                    "lastCryptoAlertDirection": direction,
                    "lastCryptoAlertPct": change_pct,
                    "lastCryptoAlertedAt": _now_iso(),
                    "lastCheckedAt": _now_iso(),
                }, merge=True)
            else:
                errors.append({
                    "user_id": user_id,
                    "symbol": symbol,
                    "stage": "crypto_market_push",
                    "error": result,
                })

    return {
        "checked_users": checked_users,
        "checked_crypto": checked_crypto,
        "sent": sent,
        "skipped_users_without_token": skipped,
        "errors": errors[:10],
        "finished_at": _now_iso(),
    }    