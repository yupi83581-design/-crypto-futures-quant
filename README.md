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
