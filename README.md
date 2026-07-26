# Forecasting the UK equity premium with transformer news sentiment

Does sentiment extracted from UK financial news improve **out-of-sample**
forecasts of the UK equity premium, beyond traditional predictors?

Built for an MSc Finance Analytics dissertation. 198,178 Guardian business and
money articles scored with FinBERT, 300 months of UK data, 179 out-of-sample
forecasts, evaluated statistically and economically.

**Answer: a qualified no.** FinBERT sentiment improves three of five model
specifications, best case +0.69% out-of-sample R², but not significantly at
conventional levels. The transformer does clearly beat a Loughran-McDonald
dictionary on the same corpus, which is the more robust finding.

---

## What is interesting here, engineering-wise

**A look-ahead bug worth 5.8 percentage points.** An earlier version of the
forecast loop trained on `df.iloc[:i+1]`. Row `i` holds the premium of month
*t+1*, the forecast target, so the model was reading the answer out of its own
training data. Corrected to `df.iloc[:i]`. `src/quantify_leak.py` runs both
timings on identical data to measure the damage:

| model | R²_OS with the bug | corrected | overstated by |
|---|---|---|---|
| Random forest | **+4.89%** | −0.88% | 5.77 pp |
| Combination | +3.73% | −0.89% | 4.61 pp |
| OLS | −5.82% | −14.02% | 8.21 pp |

The uncorrected code produced a large, plausible, entirely fictional result.

**A regression test that could not fail, until it could.** The original
look-ahead test truncated the sample but kept one extra month past the cut,
which preserved exactly the leak it was meant to detect, so it passed whether or
not the code was correct. The extra month is genuinely needed so the final
forecast has a realised target, so that final forecast is now excluded from the
comparison instead.

**Version invariance as a test, not a hope.** pandas changed the monthly
resample alias from `M` to `ME` between 2.1 and 2.2, and code relying on either
breaks on the other. This uses neither: all monthly aggregation goes through
`PeriodIndex(freq="M")`. Verified to produce identical results under pandas
**2.1.4, 2.3.3 and 3.0.2** with `FutureWarning` and `DeprecationWarning`
escalated to errors. Agreement is to 1.6e-13 relative, i.e. float64 rounding.

**Resumable GPU scoring.** 1.8 million passages through FinBERT is one to two
hours on a Colab T4, and a free session will usually disconnect at least once.
The scorer checkpoints every 2,000 articles, writes each part to a temp file
before renaming so an interrupted write cannot leave a truncated part a later
run would trust, and skips completed parts on restart. Verified idempotent and
invariant to chunk size.

**Clark-West without statsmodels.** The MSPE-adjusted t-statistic is computed
directly, including the Bartlett-kernel HAC variant, in about fifteen lines.
Agrees with `statsmodels` OLS and HAC to ten decimal places, and drops a heavy
dependency.

**Loud failure over silent wrongness.** `build_panel` cross-checks the daily and
monthly index series against each other, then audits the finished panel against
plausible ranges and against the sign of four months whose direction is not in
doubt (Oct 2008, Mar 2020, Apr 2009, Sep 2022). A unit error or a sign error
raises instead of quietly producing a wrong dissertation. Inflation is lagged
one month because ONS publishes month *t* CPI in mid-month *t+1*, so the
contemporaneous figure is not in the month-*t* information set.

---

## Layout

```
src/
  ukep_core.py                      library: loaders, panel, sentiment, forecasts, evaluation
  run_analysis.py                   CLI driver, produces every table and figure
  UK_Equity_Premium_Sentiment.ipynb self-contained notebook, library inline as cells
  quantify_leak.py                  measures the look-ahead bug described above
  lm_words.json                     Loughran-McDonald word lists, 354 pos / 2,355 neg
output/                             tables and figures from the final run
```

The notebook and `ukep_core.py` are generated from one source, so they cannot
drift apart.

## Method

- **Target**: monthly FTSE All-Share total return minus SONIA, fixed at the
  start of the month rather than within it
- **Baseline**: dividend yield, term spread, short rate, inflation, realised
  volatility
- **Augmented**: baseline plus a monthly FinBERT sentiment index. Strictly
  nested, so any difference is attributable to sentiment alone
- **Models**: OLS, elastic net, random forest, gradient boosting, plus their
  equal-weighted combination. OLS is kept deliberately, to separate the
  contribution of the *predictors* from the contribution of *model flexibility*
- **Design**: expanding window, re-estimated monthly, hyper-parameters re-tuned
  every 12 months, tuning and scaling strictly inside the training window
- **Statistical**: Campbell-Thompson out-of-sample R², Clark-West test for
  nested models
- **Economic**: mean-variance timing, weights clipped to [0, 1.5], 10 bps on
  the traded amount net of weight drift, certainty equivalent and Sharpe

## Running it

```bash
pip install -r requirements.txt

export UKEP_DATA=path/to/data
export UKEP_OUT=path/to/output

python src/run_analysis.py sentiment --finbert      # once, on a GPU
python src/run_analysis.py sentiment --dictionary   # once, anywhere, ~2 min
python src/run_analysis.py all                      # ~18 min
```

`--fast`, `--no-ct`, `--oos-start`, `--rf`, `--baseline-only` and `--no-cpi-lag`
switch the robustness variants. Nothing is downloaded at run time and no API
keys are needed.

## Data

Not included, and not redistributable. The pipeline expects six files in
`UKEP_DATA`:

| File | Source |
|---|---|
| FTSE All-Share total return, daily | Investing.com export |
| `FTSE_dividend_yield.csv` | derived from the price and total-return indices |
| `Bank Rates.csv` | Bank of England, series IUDSOIA and IUDBEDR |
| `10-year gilt yield.csv` | FRED, series IRLTLT01GBM156N |
| `CPI.csv` | ONS, series D7BT |
| `News Articles.csv` | Guardian Open Platform API, Business and Money sections |

The corpus is 694MB and its text is the Guardian's, so it is not in this repo.
The Guardian API is open and the section filters are documented above, so the
extraction is reproducible.

## Notes

The Loughran-McDonald word lists in `src/lm_words.json` are from
[Loughran and McDonald (2011)](https://sraf.nd.edu/loughranmcdonald-master-dictionary/),
redistributed here for reproducibility. FinBERT is
[ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert), Araci (2019).
