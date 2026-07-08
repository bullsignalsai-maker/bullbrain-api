import firebase_admin
from firebase_admin import credentials, firestore
from collections import defaultdict
import datetime

if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

TARGET_DOC_PATH = ("bullsignals_ai", "watchlist_symbols")

DRY_RUN = True  # first run True, then change to False


def _utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def backfill_watchlist_symbols():
    counts = defaultdict(int)
    scanned = 0

    for doc in db.collection_group("watchlist").stream():
        scanned += 1

        sym = (doc.id or "").upper().strip()
        if not sym:
            continue

        counts[sym] += 1

    print(f"Scanned {scanned} per-user watchlist docs")
    print(f"Distinct symbols: {len(counts)}")

    for sym, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {sym}: {count}")

    payload = {
        "symbols": {sym: {"count": count} for sym, count in counts.items()},
        "updated_at": _utc_now_iso(),
    }

    if DRY_RUN:
        print("\nDRY_RUN=True — not writing. Set DRY_RUN=False to persist.")
        return

    doc_root, doc_id = TARGET_DOC_PATH
    db.collection(doc_root).document(doc_id).set(payload)  # full overwrite, not merge
    print(f"\nWrote aggregate to {doc_root}/{doc_id}")


if __name__ == "__main__":
    backfill_watchlist_symbols()
