# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

BullSignalsAI backend: a Python/FastAPI service that computes stock market intelligence (BullBrain ML signal, technicals, news, watchlists, portfolio AI, push alerts) and serves it to a mobile app (Expo/React Native — see the CORS setup in `main.py`). Data is persisted in Firestore; quotes/candles come from Finnhub, Polygon, FMP, CoinGecko; narrative/chat answers come from xAI's Grok API. Deployed on Render as a web service plus a background worker and scheduled cron jobs.

## Commands

There is no build step, linter config, or test suite in this repo — verification is done by running the FastAPI app and cron/worker scripts directly.

```bash
pip install -r requirements.txt

# Run the API locally (Render starts this the same way in production)
uvicorn main:app --reload --port 10000

# One-off cron / worker scripts (each imports `main` for shared state, e.g. the loaded BullBrain model)
python market_cron.py        # */15 * * * 0-5 (Sun-Fri; Sunday included on purpose — see get_cron_mode()'s
                              #   is_sunday_overnight carve-out, a real 20:30-20:44 ET warm scan ahead of
                              #   Monday's open, not dead code or drift) — refreshes stock intelligence,
                              #   movers, homescreen carousel
python homescreen_cron.py    # lightweight snapshot cron; calls the running API's /internal/homescreen/compute
python quote_worker.py       # long-running Render background worker; refreshes live quotes with market-hours-aware throttling

# Maintenance / backfill scripts (one-off, run manually)
python scripts/backfill_daily_alpha_intelligence.py
python scripts/migrate_daily_movers.py
```

Required env vars (set on Render, read via `os.getenv` — see `backend/config.py`): `FINNHUB_KEY`, `XAI_API_KEY`, `FMP_API_KEY`, `NEWS_API_KEY`, `POLYGON_API_KEY`, `FIREBASE_ADMIN_JSON` (JSON string of the service account, parsed in `backend/firestore_utils.py`/`backend/firestore_paths.py`).

## Architecture

### Entry points
- `main.py` — the monolithic FastAPI app. Defines nearly all HTTP routes (`/predict`, `/candles`, `/quote`, `/stockdetail/*`, `/watchlist/*`, `/astra-chat`, `/market-*`, `/push/*`, etc.), loads the BullBrain XGBoost model on startup from `models/bullbrain_v2_48f.json` (downloaded from Google Drive via `gdown` if missing), and holds shared module-level state (`bullbrain_model`, `cache`) that other modules import via `import main as backend`.
- `market_cron.py`, `homescreen_cron.py` — scheduled jobs (Render cron) that refresh Firestore-cached intelligence outside the request path.
- `quote_worker.py` — long-running Render worker that keeps live quotes fresh with market-hours/weekend/holiday-aware throttling.
- `symbols_clean.py` — canonical `REAL_TICKERS` (S&P 500 + extras) and `COMPANY_NAMES` used everywhere symbol universes are needed; `symbols_clean_test.py` is an unrelated tiny fixture (6 tickers), not a pytest suite.

### Data layer: Firestore-backed caching, everywhere
Nearly every subsystem follows the same shape: a `*_repo.py` or `*_store.py` module wraps a Firestore collection with `get_*`/`save_*`/`is_*_fresh` helpers and a TTL constant, so callers never talk to Firestore directly:
- `backend/stock_repo.py` — canonical BullBrain intelligence at `/bullsignals_ai/stocks/symbols/{SYMBOL}`.
- `backend/candle_store.py` — OHLCV candle cache with delta-fetching and rate-limit protection; the single source of truth for all candle access.
- `backend/quote_repo.py` — live quote cache, 30s TTL.
- `backend/watchlist_snapshot.py` — precomputed per-user watchlist summaries.
- `backend/firestore_paths.py` / `backend/schema_versions.py` — collection names are versioned (`stockdetail_cache_{version}`, `grok_cache_{version}`); bump the version constant to do a safe schema migration instead of mutating existing docs in place.
- `backend/firestore_utils.py` and `backend/firestore_paths.py` both implement Firebase Admin init + `get_db()` singletons independently — check which one a module already imports before adding a third variant.

Fetchers degrade quietly: helpers like `safe_json` and the quote/candle providers return `None`/`{}` on failure rather than raising, so downstream code always checks for empty/missing data instead of catching exceptions.

### ML signal (BullBrain)
`BULLBRAIN_FEATURES` in `main.py` defines the exact ordered feature vector (returns, volatility, SMAs, RSI, MACD, Bollinger-style bands, volume/OBV stats, candle shape) fed to the XGBoost booster. `backend/bullbrain.py` computes features from candles and runs inference; `backend/stock_bootstrap.py` is the on-demand path (candles → features → signal → save to `stock_repo`) used when a symbol isn't cached yet, mirroring the batch logic in `market_cron.py` — the two are meant to stay in sync.

### Astra/Clara chat + AI narrative layer
- `backend/astra_intent_router.py` → `backend/astra_context_builder.py` → `backend/astra_engine.py` → `backend/astra_chat.py` is the pipeline for the in-app assistant: detect intent from user text, build context (portfolio/symbols/market data), then generate cards/answers.
- `backend/grok_ai.py` is the only module allowed to call the Grok/xAI API directly (by design, per its own docstring) — it deliberately has no FastAPI, Firestore, or candle-fetching code; other modules call its exported functions (`astra_llm_answer`, `grok_prob_up`, etc.) instead of hitting the API themselves.
- **Signal-wording convention**: user-facing copy never says raw `BUY`/`SELL`/`HOLD`. `backend/astra_engine.py`'s `sanitize_clara_answer`/`clara_signal_label` rewrite these to "Bullish Setup"/"Risk Alert"/"Neutral Setup" everywhere the assistant ("Clara") or push alerts (`backend/push_alerts.py`) speak to the user. Internal signal values themselves stay `BUY`/`SELL`/`HOLD` — only the display layer translates them. Match this convention in any new user-facing text.
- `backend/explain/` holds the narrative/indicator explanation system (`indicator_library.py`, `indicator_states.py`, `narrative_engine.py`, `reason_map.py`) that turns raw technicals into human-readable explanations for stock detail screens and watchlist summaries.

### News, market screens, alerts
- `backend/news/` — news fetching, source list, and cleaning independent of the ML/AI layers.
- `backend/market_*_logic.py` files (`market_pulse_logic.py`, `market_overview_logic.py`, `market_hotlist_logic.py`, `market_bearwatch_logic.py`, etc.) each build one homescreen/market card; `market_cron.py` is what actually invokes and persists them on a schedule.
- `backend/push_alerts.py` sends Expo push notifications (watchlist, price, portfolio, crypto alerts) and reuses `watchlist_snapshot.py` for content — go through the sanitize helpers above rather than formatting raw signals into notification text.

### Credentials note
`serviceAccountKey.json` at the repo root is untracked (not gitignored) — treat it as a live credential, not a fixture; don't add it to git or read it into logs/output.
