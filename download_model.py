# download_model.py
import requests
import os

GOOGLE_DRIVE_FILE_ID = "1lLQUjV0dB2S6eaHSJWWGWLyFIkwjKDd2"
OUTPUT_PATH = "models/bullbrain_latest.pkl"

def download_model():
    url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
    print(f"Downloading model from: {url}")

    r = requests.get(url)
    if r.status_code != 200:
        raise Exception(f"Failed to download model: HTTP {r.status_code}")

    os.makedirs("models", exist_ok=True)

    with open(OUTPUT_PATH, "wb") as f:
        f.write(r.content)

    print(f"✅ Download complete → {OUTPUT_PATH}")

if __name__ == "__main__":
    download_model()
