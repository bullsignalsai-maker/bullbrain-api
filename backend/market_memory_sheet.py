import os
from typing import List, Dict, Any

import gspread
from google.oauth2.service_account import Credentials


SHEET_NAME = "Alphaclara Market Memory"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICE_ACCOUNT_FILE = os.path.join(
    BASE_DIR,
    "backend",
    "credentials",
    "google_service_account.json",
)

def latest_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []

    latest_ts = max(str(r.get("generated_at", "")) for r in rows)
    return [r for r in rows if str(r.get("generated_at", "")) == latest_ts]

def get_sheet_client():
    import json

    # Prefer dedicated Google Sheets service account if added later.
    service_json = (
        os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        or os.getenv("FIREBASE_ADMIN_JSON")
    )

    if service_json:
        info = json.loads(service_json)
        creds = Credentials.from_service_account_info(
            info,
            scopes=SCOPES,
        )
    else:
        creds = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=SCOPES,
        )

    return gspread.authorize(creds)

def read_tab(tab_name: str) -> List[Dict[str, Any]]:
    client = get_sheet_client()
    sheet = client.open(SHEET_NAME)
    worksheet = sheet.worksheet(tab_name)
    return worksheet.get_all_records()


def get_premarket_gainers() -> List[Dict[str, Any]]:
    return read_tab("PremarketGainers")


def get_premarket_losers() -> List[Dict[str, Any]]:
    return read_tab("PremarketLosers")


def get_alpha_opportunities() -> List[Dict[str, Any]]:
    return read_tab("AlphaOpportunities")


def get_all_market_memory_candidates() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "premarket_gainers": latest_rows(get_premarket_gainers()),
        "premarket_losers": latest_rows(get_premarket_losers()),
        "alpha_opportunities": latest_rows(get_alpha_opportunities()),
    }

if __name__ == "__main__":
    data = get_all_market_memory_candidates()

    print("Premarket gainers:", len(data["premarket_gainers"]))
    print("Premarket losers:", len(data["premarket_losers"]))
    print("Alpha opportunities:", len(data["alpha_opportunities"]))

    print("\nSample gainer:")
    print(data["premarket_gainers"][0] if data["premarket_gainers"] else "None")