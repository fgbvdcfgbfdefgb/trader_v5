# trader_v5

Offline multi-agent reinforcement-learning trader for **BTC / ETH / LTC**, with a
CPU-resident LLM news advisor.

Everything the training run needs — market data, model weights, news — is
committed to this repository. Once cloned, **no network access is required**,
which is what makes it runnable inside a Snowflake workspace that can only
`pip install` and clone this repo.

---

## What's in here

| Path | Contents |
|---|---|
| `data/market/{BTC,ETH,LTC}USDT/` | **14.07M** 1-minute bars, 2017-08-17 → 2026-08-31, one Parquet per year (zstd) |
| `data/news/` | **126,746** dated news articles, 2011 → 2026, one Parquet per year + `daily_index.parquet` |
| `models/` | 3 CPU sentiment encoders, weights split into <90 MB chunks |
| `src/` | dataset, advisor, networks, environment, trainer, plotting |
| `scripts/` | `bootstrap.py` (run once offline), `smoke_test.py` |
| `tools/` | the online scrapers used to build the datasets (not needed at train time) |

### Market data
Binance public monthly klines, normalised to a single schema. Timestamps are
epoch-ms (Binance's 2025 switch to microseconds is auto-detected and
corrected). **3,178 days** have complete coverage across all three assets —
that is the epoch sampling pool.

### News corpus
| Source | What it gives |
|---|---|
| `edaschau/bitcoin_news` (HF) | dated BTC headlines + bodies, 2011-2025 |
| `SahandNZ/cryptonews-articles` (HF) | cryptonews.com, 2021-2023 |
| Cointelegraph sitemaps | dated article URLs |
| cryptoslate / newsbtc / bitcoinist / ambcrypto sitemaps | 2014-2026, fills the recent window |

Tagged per asset: BTC 59,957 · MKT 49,435 · ETH 15,260 · LTC 2,094.

**Coverage: 3,183 / 3,183 tradable days (100%)** have news behind them —
median 253 articles in the trailing 7-day window, worst case 34.

> **De-spiking.** Sitemap `<lastmod>` is a *modification* time. When an outlet
> bulk-re-touches its archive, thousands of old articles collapse onto one
> date (Cointelegraph dumped 14,787 rows onto 2026-07-07). Those carry false
> timestamps and would inject phantom news, so `despike()` drops any
> `(domain, date)` bucket above 250 articles — **55,847 rows removed**.
>
> LTC coverage is genuinely thin (2,094 rows); the advisor leans on the
> BTC/MKT channel for LTC.

### Models (CPU advisor)
| Dir | Upstream | Params | Chunks |
|---|---|---|---|
| `models/finbert` | `ProsusAI/finbert` | 110M | 5 × 90 MB |
| `models/cryptobert` | `ElKulako/cryptobert` | 125M | 6 × 90 MB |
| `models/fin_distilroberta` | `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` | 82M | 4 × 90 MB |

No Git LFS is used — `models/assemble.py` concatenates the chunks and verifies
SHA-256 against `manifest.json`.

---

## The three agents

They train **concurrently in separate processes** and feed each other:

```
[producer]  random day -> features -> CPU LLM advisor (point-in-time)
     |
     +--> [price_predictor]  GPU A   causal TCN, 4-horizon returns + sigma
     +--> [market_analyzer]  GPU B   regime + realised-vol, FUSED with news
     +--> [trader / PPO]     GPU C   consumes both agents' live latents
```

The predictor and analyzer publish fresh weights every `--sync-every` epochs;
the trader hot-reloads them mid-run, so the policy always acts on the current
beliefs of the other two rather than a frozen snapshot.

**Observation** given to the policy at each decision:
market features (40) · advisor vector (20) · predictor μ/σ + latent (56) ·
analyzer regime/vol + latent (44) · portfolio state (7).

**Episode = one randomly chosen calendar day.** Start **$20**, decisions every
5 minutes (288 steps), 4 bp fees + 1 bp slippage, gross exposure ≤ 1.0,
long **and** short. Reward is log-growth, minus a drawdown penalty and a
turnover penalty, plus a terminal bonus for finishing at **≥ $30**.

### Point-in-time news — no lookahead
At minute *t* of day *D*, the advisor is only ever shown articles with
`datetime_utc <= t`, plus a 7-day trailing window. Nothing published later
that day, and nothing from the future, can reach the model. This is enforced
in `advisor.NewsStore.window()` with a `searchsorted` upper bound.

---

## Running it

### Offline (Snowflake)

```bash
pip install -r requirements.txt
python scripts/bootstrap.py          # reassemble weights + verify data
python src/train.py --epochs 2000
```

`bootstrap.py` sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`; every
`from_pretrained` call uses `local_files_only=True`.

### Hardware auto-sizing
`src/resources.py` inspects GPUs/CPU/RAM and picks widths, depth, batch size
and device placement. It needs no flags.

| GPUs found | Placement | Width |
|---|---|---|
| ≥3 | predictor→cuda:0, analyzer→cuda:1, trader→cuda:2 | full tier |
| 2 | predictor+analyzer→cuda:0, trader→cuda:1 | full tier |
| 1 | all three share cuda:0 | **auto-shrunk** ~40% narrower, 1 layer fewer, half batch |
| 0 | CPU | tiny tier |

Tier comes from the *smallest* GPU's free VRAM: ≥38 GB → `d_model` 768 /
8 layers; ≥20 GB → 512/6; ≥10 GB → 320/4 (a T4 lands here); ≥6 GB → 192/3.

The LLM advisor always runs on **CPU**, in the producer process, with its
thread count capped so it never starves the GPU trainers.

Useful flags:
```
--single-gpu          pretend there is only one GPU
--cpu                 no CUDA at all
--day-lo / --day-hi   restrict the sampling pool (e.g. train/test split)
--no-advisor          ablate the news channel
--require-news N      only sample days with >=N articles in the trailing 7d
                      (default 3; stops the agent training on silent stretches)
--png-every N         save a diagnostic PNG every N epochs
```

---

## Outputs

Every `--png-every` epochs, `outputs/epochs/epoch_000123_2024-03-07.png` is
written containing 13 panels — equity, rebased prices, weights, exposure and
turnover, step and cumulative reward, drawdown, predictor forward-return
track, analyzer regime belief, advisor sentiment track, normalised losses,
final equity across all epochs, per-asset contribution — plus **the actual
headlines the advisor saw that day with their sentiment scores**.

Also written: `outputs/metrics.jsonl` (one row per epoch), `outputs/summary.png`,
and checkpoints in `outputs/checkpoints/`.

---

## On the "$20 → $30 per day" objective

That is a **+50% intraday return**, and it is encoded exactly as asked: the
terminal bonus fires at ≥ $30 and `hit_rate` is tracked in `metrics.jsonl`.
Be aware this is an extremely aggressive target — reaching it consistently on
real minute data with fees and a gross-exposure cap of 1.0 is not something
any honest backtest will produce reliably. Treat the hit rate as a shaping
signal and an upper-bound diagnostic, not an expectation. To make it more
attainable you can raise `--max-gross` (leverage), which the environment
supports directly.

## Rebuilding the datasets (online only)

```bash
python tools/fetch_market.py    # Binance monthly klines -> Parquet
python tools/fetch_news.py      # HF datasets + Cointelegraph -> Parquet
python tools/fetch_models.py    # HF weights -> chunked
```
