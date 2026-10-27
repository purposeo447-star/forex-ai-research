# FX Research Lab — live-read-only prototype

A research-first starter for a 5-minute Forex next-candle direction model. It can request current 5-minute candles from Twelve Data and generate a read-only research estimate once you configure your own data-provider API key. It does not log into Quotex, read your Quotex account, or place trades.

## Important
- Do not send anyone your Quotex password or session cookies. The Quotex link you shared is an account page, not an integration credential.
- Quotex's published Rules of Trading Operations restrict automated mechanisms/software that allow operations without direct user participation. This project therefore provides independent read-only research signals only; it does not automate trades.
- Predictions are uncertain. Provider quotes may differ from Quotex quotes. The holdout accuracy displayed is a quick diagnostic, not proof of a profitable strategy.
- The current endpoint refits a model from the latest provider history each request. It is a prototype, not production infrastructure; API limits and request latency apply.
- The API key belongs only in the backend environment, never in HTML/JavaScript or a public repository.

## Requirements
Python 3.10+ recommended.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Set the provider key in your shell before starting the API (get a key directly from the provider and verify plan/history limits):

```bash
export TWELVE_DATA_API_KEY='YOUR_KEY_HERE'  # macOS/Linux
uvicorn main:app --host 127.0.0.1 --port 8000
```

On Windows PowerShell:

```powershell
$env:TWELVE_DATA_API_KEY='YOUR_KEY_HERE'
uvicorn main:app --host 127.0.0.1 --port 8000
```

Then open `index.html` in a browser and click **Fetch live research signal**. The browser calls `http://127.0.0.1:8000`; if your browser blocks local requests from a file URL, serve the dashboard from a local HTTP server in another terminal, e.g. `python -m http.server 8080`, then open `http://127.0.0.1:8080`.

API health: `http://127.0.0.1:8000/health`
Interactive API docs: `http://127.0.0.1:8000/docs`

## Historical CSV training
Required columns (case-insensitive): `timestamp,open,high,low,close`. Optional `volume`. One symbol and timeframe per CSV. Use the API docs to POST a CSV to `/data/upload`, then call `/model/train` and `/patterns/study`.

## Data limitations
The live endpoint requests up to 5,000 5-minute candles, not eight full years. Eight years of intraday coverage must be sourced separately, and actual history depth depends on provider plan, instrument, and API limits. Do not describe the model as having analyzed eight years until a complete, quality-checked dataset has actually been loaded.

## Research limitations
- The label is whether the next candle closes above its open; it is not a guaranteed trading outcome.
- Use walk-forward testing, a final untouched test period, calibration, baseline comparisons, uncertainty intervals, and payout/cost assumptions before judging efficacy.
- Signals should be paper-traded and logged before being relied on.
- Telegram notification settings are not implemented yet; no token should ever be placed in frontend code.


## Online deployment (Render)
1. Put this project in a private GitHub repository (do not commit `.env` or API keys).
2. In Render, create a new Blueprint from that repository; Render reads `render.yaml`.
3. In the service Environment settings, add `TWELVE_DATA_API_KEY`, `TELEGRAM_BOT_TOKEN`, and `TELEGRAM_CHAT_ID`. Keep all values private and server-side.
4. Deploy and open the generated HTTPS service URL. The dashboard and API are served from the same host.
5. Test `/health`, then click **Test Telegram** in the dashboard. Enable **Send qualifying signals to Telegram** to send CALL/PUT research estimates (duplicate symbol/candle alerts are suppressed in the running process).

### Telegram setup
Create a bot with Telegram's official BotFather, send the bot a message from your Telegram account, then obtain the chat ID using Telegram's bot API or a trusted bot-management tool. Store the bot token and chat ID only in the hosting environment variables. Never paste these secrets into this chat, HTML, or a public repository.

### Deployment status and limitations
This repository includes deployment configuration; it is not automatically deployed just by downloading it. Deployment requires connecting your own GitHub and Render accounts and setting the secrets in Render. A free or low-tier host may sleep or restart, which can interrupt alerting. Current candle history is up to 5,000 bars, not eight full years. Telegram duplicate suppression is in-memory and resets when the service restarts.
