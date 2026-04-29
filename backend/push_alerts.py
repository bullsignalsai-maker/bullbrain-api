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


def run_watchlist_signal_alerts(max_users: int = 200) -> Dict[str, Any]:
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

    return {
        "checked_users": checked_users,
        "checked_symbols": checked_symbols,
        "sent": sent,
        "baselined": baselined,
        "skipped_users_without_token": skipped,
        "errors": errors[:10],
        "finished_at": _now_iso(),
    }