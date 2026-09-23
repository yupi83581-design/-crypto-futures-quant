# -crypto-futures-quant
Crypto Futures quantitative research and paper-trading engine

## Read-only snapshot runtime

The repository now includes a persistent-capable runtime composition at
`src/production/runtime.py`.

It connects the existing components without adding a second quant engine:

```text
Binance USDⓈ-M Futures public klines
        ↓
BinancePublicMarketDataAdapter
        ↓
real closed-candle training window
        ↓
BaselineRSIPipeline / LogisticRegressionBaseline
        ↓
ProductionInferenceEngine
        ↓
QuantCommandCenterSnapshot / snapshot_bridge
        ↓
GET /snapshot
```

The runtime is research/paper only. It never places exchange orders.

### Start

From the repository root:

```bash
python -m src.production.runtime
```

Default environment:

- `QUANT_SYMBOL=BTCUSDT`
- `QUANT_TIMEFRAME=5m`
- `QUANT_LOOKBACK_CANDLES=240`
- `QUANT_TRAINING_FRACTION=0.8`
- `QUANT_REFRESH_SECONDS=300`
- `QUANT_SNAPSHOT_HOST=0.0.0.0`
- `QUANT_SNAPSHOT_PORT=8080`

The service exposes only:

```text
GET /snapshot
```

If Binance data cannot be reached or validated, the service stays available
but publishes `status=ERROR` and `market_data.status=OFFLINE`; it does not
replace missing values with zero or synthetic data.

### Android / Termux

The runtime itself is already Android/Termux-friendly: it uses the Python
standard library for HTTP/networking and the repository's existing pure-Python
baseline model. The only declared third-party dependency is `jsonschema`.
Current `jsonschema` releases pull in `rpds-py`, which is a Rust-backed native
module. On Termux, install Termux's native `python-rpds-py` package first so
pip does not attempt an on-device Rust build.

Run the setup script from the repository root:

```bash
bash scripts/termux_setup.sh
```

The script checks for Android ARM64, installs the minimal Termux tools,
installs `python-rpds-py` from the Termux package repository, installs the
repository requirements, and verifies the imports. It does not install any
exchange SDK, trading component, API key, or cloud service.

Then start the existing runtime unchanged:

```bash
python -m src.production.runtime
```

The default listener is `0.0.0.0:8080`. No Binance API key is required because
the runtime uses the existing public Binance Futures market-data endpoint.

If `python-rpds-py` is not available from the configured Termux repository,
the setup script stops rather than installing Rust or silently changing the
project dependency. Refresh/switch to an official current Termux repository
and rerun the script.

### Verify locally

In a second Termux session:

```bash
curl -sS http://127.0.0.1:8080/snapshot
```

A successful live refresh returns JSON with `market_data.status` equal to
`LIVE`, `exchange` equal to `BINANCE`, `market_type` equal to `FUTURES`, and
`provenance` equal to `BINANCE_PUBLIC_FUTURES_KLINES`. If public Binance data
is unavailable, the same endpoint remains available but reports `ERROR` /
`OFFLINE` rather than fabricating a snapshot.

### HTTPS tunnel for testing

For a temporary development/demo URL, Cloudflare Quick Tunnel can expose the
local HTTP service without changing the runtime:

```bash
cloudflared tunnel --url http://localhost:8080
```

Cloudflare prints a temporary `https://*.trycloudflare.com` URL. Quick Tunnels
are for testing/development, not a permanent production endpoint. The project
itself does not depend on `cloudflared`.

### Minimal public deployment path

Run the command above on a persistent host that permits inbound TCP access to
the configured port. Configure the host's firewall/security group to expose
that port to the intended dashboard client. Then verify:

```bash
curl -sS http://HOST:8080/snapshot
```

Do not use a GitHub Actions runner as the public runtime: Actions jobs are
ephemeral. The repository does not contain hosting credentials or a public
runtime URL.

### Evidence boundary

A successful software test proves software behavior only. It does not prove
trading profitability, live-market edge, or the Final Validation Gate.
The snapshot intentionally remains `validation_status=INSUFFICIENT_EVIDENCE`
until the required paper-trading evidence exists.
