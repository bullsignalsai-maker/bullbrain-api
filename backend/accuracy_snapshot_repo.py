# backend/accuracy_snapshot_repo.py
# ---------------------------------------------------------
# Accuracy Snapshot Repository (Firestore)
#
# Daily win-rate/mean-return snapshots for the accuracy trend chart. Each
# doc also carries a `full_report` field -- the complete build_accuracy_
# report() output (all subgroup breakdowns, factor scores, both horizons),
# added 2026-09-10 so /alphaclara-accuracy-report and /alphaclara-
# historical-edge can serve it directly instead of recomputing the same
# ~90-day pick_tracking scan on every request (measured 7.6-8.0s live vs
# a cheap doc read). Free to add: market_cron's win-rate snapshot block
# already builds the full report and was previously discarding everything
# but the thin rollup. Doc ID is a fixed UTC date string, so a same-day
# rewrite (market_cron's final_close_intelligence window can tick more
# than once) is a natural idempotent overwrite -- no separate
# once-per-day state doc needed.
# ---------------------------------------------------------

import datetime
from typing import Optional, Dict, Any, List

import firebase_admin
from firebase_admin import firestore  # type: ignore

COL_ROOT = "bullsignals_ai"
SNAPSHOTS_DOC = "accuracy_snapshots"


def _db():
    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    return firestore.client()


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _snapshots_collection(db=None):
    return (
        (db or _db())
        .collection(COL_ROOT)
        .document(SNAPSHOTS_DOC)
        .collection("snapshots")
    )


def save_accuracy_snapshot(date_key: str, snapshot: Dict[str, Any], db=None) -> None:
    snapshot.setdefault("generated_at", _now_iso())
    _snapshots_collection(db).document(date_key).set(snapshot, merge=True)


def get_accuracy_snapshot(date_key: str, db=None) -> Optional[Dict[str, Any]]:
    doc = _snapshots_collection(db).document(date_key).get()
    return doc.to_dict() if doc.exists else None


def get_accuracy_snapshots_since(since_date: str, db=None) -> List[Dict[str, Any]]:
    query = (
        _snapshots_collection(db)
        .where("date", ">=", since_date)
        .order_by("date")
    )
    return [doc.to_dict() for doc in query.stream()]


def get_latest_accuracy_snapshot(db=None) -> Optional[Dict[str, Any]]:
    """
    The single most recently recorded daily snapshot, regardless of date --
    a proper order_by+limit(1) query, not get_accuracy_snapshots_since()
    with the caller fetching a whole date range and discarding every row
    but the last. Used by /alphaclara-accuracy-report and /alphaclara-
    historical-edge to read the precomputed `full_report` field instead of
    recomputing it live.
    """
    query = (
        _snapshots_collection(db)
        .order_by("date", direction=firestore.Query.DESCENDING)
        .limit(1)
    )
    docs = list(query.stream())
    return docs[0].to_dict() if docs else None
