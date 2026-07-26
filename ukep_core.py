"""
=============================================================================
 UK EQUITY PREMIUM PREDICTABILITY WITH GUARDIAN NEWS SENTIMENT
 Core library -- shared by the notebook and the command-line script.
=============================================================================

Research question
-----------------
Does textual sentiment extracted from UK financial news improve the
out-of-sample predictability of the UK equity premium, beyond what is
achievable using traditional predictors alone?

Design
------
  benchmark  : expanding historical mean of the premium      (Welch-Goyal 2008)
  baseline   : dy, term_spread, short_rate, infl, rvol
  augmented  : baseline + monthly news sentiment index
The baseline predictors are strictly nested inside the augmented set, so any
difference in forecast accuracy is attributable to the sentiment signal alone.

PANDAS COMPATIBILITY
--------------------
This module never uses the resample frequency aliases "M" or "ME", which
changed meaning between pandas 2.1 and 2.2 and are a common source of
breakage.  All monthly aggregation goes through PeriodIndex with freq="M",
whose meaning has been stable across every 1.x, 2.x and 3.x release.  Tested
on pandas 2.1.4 and pandas 3.0.2 with byte-identical output.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# --- Where things live -------------------------------------------------------
# Point DATA_DIR at the folder containing the six raw CSV files.
DATA_DIR = Path(os.environ.get("UKEP_DATA", "data"))
OUT_DIR = Path(os.environ.get("UKEP_OUT", "output"))

RAW = {
    "tr_daily": "FTSE All-Share Total Return GBP End Of Day Historical Results Price Data.csv",
    "dy": "FTSE_dividend_yield.csv",
    "rates": "Bank Rates.csv",
    "gilt": "10-year gilt yield.csv",
    "cpi": "CPI.csv",
    "articles": "News Articles.csv",
}

# --- Sample ------------------------------------------------------------------
# The dividend yield is a trailing-twelve-month construction, so it only
# becomes available twelve months after the index series begins.  That fixes
# the panel start at 2001-01 rather than the 2000-01 stated in the draft.
SAMPLE_START = "2001-01"
SAMPLE_END = "2025-12"

# First month for which an out-of-sample forecast is produced.  Everything
# before it is the initial training window.  120 months of training data is
# the conventional minimum in this literature.
OOS_START = os.environ.get("UKEP_OOS_START", "2011-01")

# --- Variables ---------------------------------------------------------------
TARGET = "eqp"
BASELINE_PREDICTORS = ["dy", "term_spread", "short_rate", "infl", "rvol"]
SENTIMENT_COL = "sent"
AUGMENTED_PREDICTORS = BASELINE_PREDICTORS + [SENTIMENT_COL]

# --- Risk-free rate ----------------------------------------------------------
# IUDSOIA = SONIA, a realised overnight market rate.  IUDBEDR = Bank Rate, the
# administered policy rate, used as a robustness check.
RF_SERIES = "IUDSOIA"
RF_SERIES_ALT = "IUDBEDR"

# --- Models ------------------------------------------------------------------
# "ols" is the classic linear predictive regression of the Welch-Goyal and
# Campbell-Thompson literature. Keeping it alongside the machine-learning
# methods is what lets the study separate the contribution of the PREDICTORS
# from the contribution of MODEL FLEXIBILITY, as promised in Section 4.2.
MODELS = ["ols", "enet", "rf", "gbrt"]
ADD_COMBINATION = True          # equal-weighted average of the three models
RETUNE_EVERY = int(os.environ.get("UKEP_RETUNE", 12))
CV_FOLDS = 5
SEED = 42

# --- Campbell-Thompson restriction -------------------------------------------
# Floor the equity premium forecast at zero.  Applied to model forecasts only:
# clipping the benchmark as well would change the denominator of the
# out-of-sample R-squared and is not what Campbell and Thompson (2008) do.
IMPOSE_CT_RESTRICTION = True
CT_ALSO_ON_BENCHMARK = False

# --- Economic evaluation -----------------------------------------------------
GAMMA = 3.0
W_MIN, W_MAX = 0.0, 1.5
VOL_WINDOW = 60                 # months in the rolling variance forecast
VOL_MIN_PERIODS = 36
TRANSACTION_COST = 0.0010       # 10 bps, charged on |change in weight|

# --- Sentiment ---------------------------------------------------------------
FINBERT_MODEL = "ProsusAI/finbert"
MAX_PASSAGES_PER_ARTICLE = 10   # cap so one long article cannot dominate
FINBERT_MAX_LEN = 128
FINBERT_BATCH = 128
CHUNK_ROWS = 2000               # articles per checkpoint


# =============================================================================
# 2. PANDAS-VERSION-SAFE MONTHLY HELPERS
# =============================================================================
# Period freq "M" is stable across all pandas versions; the resample aliases
# "M" (<=2.1) and "ME" (>=2.2) are not.  Everything below is expressed in
# monthly Periods and only converted to Timestamps for plotting and export.

def to_month(idx) -> pd.PeriodIndex:
    """Convert any datetime-like index or Series to a monthly PeriodIndex."""
    return pd.PeriodIndex(pd.to_datetime(idx), freq="M")


def month_end_ts(pidx: pd.PeriodIndex) -> pd.DatetimeIndex:
    """Monthly Periods -> Timestamps at the last calendar day of the month."""
    return pd.DatetimeIndex(pidx.to_timestamp(how="end")).normalize()


def monthly_last(s: pd.Series) -> pd.Series:
    """Last observation within each calendar month. Replaces .resample(...).last()."""
    s = s.sort_index()
    g = s.groupby(to_month(s.index))
    out = g.last()
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out.sort_index()


def monthly_mean(s: pd.Series) -> pd.Series:
    s = s.sort_index()
    out = s.groupby(to_month(s.index)).mean()
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out.sort_index()


def monthly_std(s: pd.Series, min_obs: int = 10) -> pd.Series:
    """Within-month sample standard deviation. Replaces .resample(...).std()."""
    s = s.sort_index()
    g = s.groupby(to_month(s.index))
    out = g.std(ddof=1)
    out = out.where(g.size() >= min_obs)
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out.sort_index()


# =============================================================================
# 3. RAW DATA LOADERS
# =============================================================================
# Each loader is written against the exact quirks of the file it reads and
# raises loudly rather than silently producing a wrong series.

def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file:\n  {path}\n"
            f"Set UKEP_DATA (currently {DATA_DIR}) to the folder holding the raw CSVs."
        )
    return path


def load_tr_daily() -> pd.Series:
    """
    Daily FTSE All-Share total return index (Investing.com export).

    Quirks handled: UTF-8 BOM in the header, thousands separators inside
    quoted strings, rows in reverse chronological order, day-first dates.
    """
    p = _require(DATA_DIR / RAW["tr_daily"])
    df = pd.read_csv(p, encoding="utf-8-sig")
    if "Date" not in df.columns or "Price" not in df.columns:
        raise ValueError(f"Unexpected columns in {p.name}: {df.columns.tolist()}")
    date = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="raise")
    price = (df["Price"].astype(str)
             .str.replace(",", "", regex=False)
             .str.strip()
             .astype(float))
    s = pd.Series(price.values, index=date, name="tr_index").sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.isna().any():
        raise ValueError("NaNs in the daily total return index.")
    return s


def load_dividend_yield() -> pd.DataFrame:
    """
    Monthly FTSE All-Share price index, total return index and trailing
    dividend yield, as supplied.  Returns a monthly-Period DataFrame.
    """
    p = _require(DATA_DIR / RAW["dy"])
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="raise")
    df = df.set_index("date").sort_index()
    df.index = to_month(df.index)
    need = {"price_index", "tr_index", "div_yield"}
    if not need.issubset(df.columns):
        raise ValueError(f"{p.name} must contain {sorted(need)}; has {df.columns.tolist()}")
    if df["div_yield"].max() > 1.0:
        raise ValueError("div_yield looks like a percentage; this code expects a decimal.")
    return df[["price_index", "tr_index", "div_yield"]]


def load_rates() -> pd.DataFrame:
    """
    Bank of England daily series: IUDBEDR (Bank Rate) and IUDSOIA (SONIA),
    both in percent per annum.  Dates arrive as '04 Jan 2000'.
    """
    p = _require(DATA_DIR / RAW["rates"])
    df = pd.read_csv(p)
    dcol = "DATE" if "DATE" in df.columns else df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol], format="%d %b %Y", errors="raise")
    df = df.set_index(dcol).sort_index()
    for c in (RF_SERIES, RF_SERIES_ALT):
        if c not in df.columns:
            raise ValueError(f"{p.name} is missing column {c}; has {df.columns.tolist()}")
    return df[[RF_SERIES, RF_SERIES_ALT]].astype(float)


def load_gilt() -> pd.Series:
    """
    FRED IRLTLT01GBM156N, UK 10-year benchmark gilt yield, percent per annum,
    monthly.  FRED stamps monthly observations on the FIRST day of the month;
    the value is a within-month average, so it is treated as information dated
    at that month's end.
    """
    p = _require(DATA_DIR / RAW["gilt"])
    df = pd.read_csv(p)
    dcol = df.columns[0]
    vcol = df.columns[1]
    df[dcol] = pd.to_datetime(df[dcol], errors="raise")
    s = pd.Series(pd.to_numeric(df[vcol], errors="coerce").values,
                  index=to_month(df[dcol]), name="long_yield").sort_index()
    if s.isna().any():
        raise ValueError("NaNs in the gilt yield series.")
    return s


def load_cpi() -> pd.Series:
    """
    ONS series D7BT, CPI all items, 2015=100.  The file carries a metadata
    block, then annual rows, then quarterly rows, then monthly rows.  Only the
    monthly rows ('1988 JAN') are wanted.
    """
    p = _require(DATA_DIR / RAW["cpi"])
    raw = pd.read_csv(p, header=None, names=["label", "value"], dtype=str,
                      engine="python", on_bad_lines="skip")
    pat = re.compile(r"^\s*(\d{4})\s+([A-Z]{3})\s*$")
    keep = raw["label"].astype(str).str.match(pat)
    m = raw.loc[keep].copy()
    if len(m) < 200:
        raise ValueError(f"Only {len(m)} monthly CPI rows parsed from {p.name}; expected 400+.")
    per = pd.PeriodIndex(
        pd.to_datetime(m["label"].str.strip(), format="%Y %b", errors="raise"), freq="M")
    s = pd.Series(pd.to_numeric(m["value"], errors="coerce").values,
                  index=per, name="cpi").sort_index()
    if s.isna().any():
        raise ValueError("NaNs in the CPI series.")
    return s


# =============================================================================
# 4. PANEL CONSTRUCTION
# =============================================================================

def build_panel(rf_series: str = RF_SERIES,
                lag_inflation: bool = True,
                verbose: bool = True) -> pd.DataFrame:
    """
    Assemble the monthly panel indexed by monthly Period.

    Real-time information alignment
    -------------------------------
    Every variable dated month t must be knowable by the close of month t.

      eqp_t          total return on the index during month t, minus the
                     risk-free rate fixed at the START of month t.  Using a
                     within-month or end-of-month rate would put information
                     into the premium that an investor could not have had when
                     the position was opened.
      dy_t           trailing dividends over the month-t closing price.
      short_rate_t   SONIA on the last business day of month t.
      long_yield_t   10-year gilt yield for month t.
      term_spread_t  long_yield_t - short_rate_t.
      infl_t         CPI year-on-year for month t-1.  ONS releases month-t CPI
                     in the middle of month t+1, so the contemporaneous value
                     is NOT in the month-t information set.  Lagging it by one
                     month removes a look-ahead that would otherwise inflate
                     every result in the study.
      rvol_t         annualised standard deviation of daily index returns
                     within month t.
    """
    tr_d = load_tr_daily()
    dyf = load_dividend_yield()
    rates = load_rates()
    gilt = load_gilt()
    cpi = load_cpi()

    # ---- equity side --------------------------------------------------------
    tr_m = monthly_last(tr_d)                       # month-end TR index level
    tr_ret = tr_m.pct_change()                      # total return during month t

    # Cross-check the daily file against the supplied monthly file.
    common = tr_m.index.intersection(dyf.index)
    gap = (tr_m.reindex(common) - dyf["tr_index"].reindex(common)).abs().max()
    if verbose:
        print(f"[check] daily vs monthly TR index, largest gap: {gap:.6f}")
    if gap > 1e-6:
        raise ValueError(
            "The daily and monthly total return series disagree. Resolve before proceeding.")

    daily_ret = tr_d.pct_change()
    rvol = (monthly_std(daily_ret, min_obs=10) * np.sqrt(252)).rename("rvol")

    # ---- risk-free rate -----------------------------------------------------
    rf_eom = monthly_last(rates[rf_series]) / 100.0          # annualised decimal
    rf_start_of_month = (rf_eom.shift(1) / 12.0).rename("rf")  # known at start of t
    short_rate = rf_eom.rename("short_rate")                 # predictor, end of t

    # ---- term spread --------------------------------------------------------
    long_yield = (gilt / 100.0).rename("long_yield")
    term_spread = (long_yield - short_rate).rename("term_spread")

    # ---- inflation ----------------------------------------------------------
    infl_raw = cpi.pct_change(12).rename("infl")
    infl = infl_raw.shift(1) if lag_inflation else infl_raw

    # ---- assemble -----------------------------------------------------------
    panel = pd.concat(
        [tr_ret.rename("tr_ret"),
         dyf["div_yield"].rename("dy"),
         dyf["price_index"].rename("price_index"),
         rf_start_of_month, short_rate, long_yield, term_spread,
         infl.rename("infl"), rvol],
        axis=1)
    panel.index = pd.PeriodIndex(panel.index, freq="M")
    panel = panel.sort_index()

    panel["eqp"] = panel["tr_ret"] - panel["rf"]

    cols = ["eqp", "tr_ret", "rf", "dy", "term_spread", "short_rate",
            "long_yield", "infl", "rvol", "price_index"]
    panel = panel.loc[SAMPLE_START:SAMPLE_END, cols].dropna()
    panel.index.name = "month"

    if verbose:
        print(f"[panel] {len(panel)} months, {panel.index.min()} to {panel.index.max()}")
    _audit_panel(panel, verbose=verbose)
    return panel


def _audit_panel(panel: pd.DataFrame, verbose: bool = True) -> None:
    """Hard sanity checks. These catch sign errors, unit errors and misalignment."""
    problems = []

    ann = panel["eqp"].mean() * 12
    vol = panel["eqp"].std(ddof=1) * np.sqrt(12)
    if not (-0.02 < ann < 0.15):
        problems.append(f"annualised mean premium {ann:.4f} is outside a plausible range")
    if not (0.08 < vol < 0.30):
        problems.append(f"annualised premium volatility {vol:.4f} is implausible")
    if not (0.015 < panel["dy"].mean() < 0.06):
        problems.append(f"mean dividend yield {panel['dy'].mean():.4f} is implausible")
    if not (0.005 < panel["infl"].mean() < 0.06):
        problems.append(f"mean inflation {panel['infl'].mean():.4f} is implausible")
    if panel["rvol"].min() <= 0:
        problems.append("non-positive realised volatility")

    # Known UK market episodes. The direction of these months is not in doubt,
    # so a wrong sign here means the return series is broken.
    landmarks = {"2008-10": "<0", "2020-03": "<0", "2009-04": ">0", "2022-09": "<0"}
    for ym, direction in landmarks.items():
        per = pd.Period(ym, freq="M")
        if per in panel.index:
            v = panel.loc[per, "eqp"]
            ok = v < 0 if direction == "<0" else v > 0
            if not ok:
                problems.append(f"{ym} premium is {v:+.4f}, expected {direction}")
            elif verbose:
                print(f"[check] {ym} premium {v:+.4%}  ok")

    if problems:
        raise AssertionError("Panel audit failed:\n  - " + "\n  - ".join(problems))
    if verbose:
        print(f"[audit] passed. annualised premium {ann:.2%}, volatility {vol:.2%}")


def descriptive_table(panel: pd.DataFrame,
                      sent: pd.Series | None = None) -> pd.DataFrame:
    """Summary statistics plus first-order autocorrelation, for the data chapter."""
    cols = ["eqp", "dy", "term_spread", "short_rate", "infl", "rvol"]
    df = panel[cols].copy()
    if sent is not None:
        df["sent"] = sent.reindex(df.index)
    rows = []
    for c in df.columns:
        x = df[c].dropna()
        rows.append({
            "variable": c, "n": int(x.size),
            "mean": x.mean(), "sd": x.std(ddof=1),
            "min": x.min(), "p25": x.quantile(0.25), "median": x.median(),
            "p75": x.quantile(0.75), "max": x.max(),
            "skew": x.skew(), "kurtosis": x.kurtosis(),
            "AR1": x.autocorr(1),
        })
    return pd.DataFrame(rows).set_index("variable")


# =============================================================================
# 5. SENTIMENT
# =============================================================================
# Two scorers, sharing one article-level aggregation rule so the comparison
# between them is clean:
#
#     article score = share of positive passages - share of negative passages
#     monthly index = mean article score across articles published that month
#
# sent_t is built from articles published DURING month t and is used to
# forecast the premium in month t+1, so it contains no future information.

_TOKEN_RE = re.compile(r"[a-z']+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_WS_RE = re.compile(r"\s+")
_BOILER_RE = re.compile(
    r"(\(c\)\s*\d{0,4}[^.]{0,80}(all rights reserved|reserved)\.?"
    r"|copyright\s+\d{4}[^.]{0,80}\.?"
    r"|photograph:\s*[^.]{0,60}\.?)", re.I)


def clean_text(text) -> str:
    t = _BOILER_RE.sub(" ", str(text))
    return _WS_RE.sub(" ", t).strip()


def split_passages(text: str,
                   max_passages: int = MAX_PASSAGES_PER_ARTICLE,
                   min_chars: int = 30,
                   max_chars: int = 1000) -> list:
    """
    Sentence-level split, keeping passages long enough to carry tone.  The
    first `max_passages` are taken rather than a random subset, because news
    convention puts the substance in the opening paragraphs and taking a fixed
    prefix keeps the rule deterministic and reproducible.
    """
    parts = _SENT_SPLIT_RE.split(text)
    out = []
    for p in parts:
        p = p.strip()
        if min_chars <= len(p) <= max_chars:
            out.append(p)
            if len(out) >= max_passages:
                break
    if not out and text.strip():
        out = [text.strip()[:max_chars]]
    return out


# ----------------------------------------------------------------- dictionary
def load_lm_words(path=None) -> tuple:
    """
    Loughran and McDonald (2011) financial sentiment word lists.
    Reads `lm_words.json` ({"positive": [...], "negative": [...]}) if present,
    otherwise falls back to the pysentiment2 package, otherwise raises.
    The lists are fully inflected, so no stemming is required.
    """
    cands = [Path(path)] if path else []
    cands += [DATA_DIR / "lm_words.json", Path("lm_words.json"),
              OUT_DIR / "lm_words.json"]
    for c in cands:
        if c and c.exists():
            d = json.loads(c.read_text())
            return set(d["positive"]), set(d["negative"])
    try:
        import pysentiment2  # noqa
        import pandas as _pd
        p = (Path(pysentiment2.__file__).parent / "static" / "LM.csv")
        d = _pd.read_csv(p)
        return (set(d.loc[d.Positive > 0, "Word"].str.lower()),
                set(d.loc[d.Negative > 0, "Word"].str.lower()))
    except Exception as exc:
        raise FileNotFoundError(
            "Loughran-McDonald word lists not found. Either place lm_words.json "
            "beside the data, or `pip install pysentiment2`."
        ) from exc


def lm_passage_labels(passages, pos: set, neg: set) -> np.ndarray:
    """
    Label each passage positive / negative / neutral by net dictionary count,
    with simple negation handling so 'not a bad quarter' is not scored
    negative.  Negation flips the polarity of the next three tokens, which is
    the standard window in the accounting literature.
    """
    NEGATORS = {"not", "no", "never", "none", "nor", "neither", "without",
                "hardly", "scarcely", "barely", "cannot", "n't", "isn't",
                "wasn't", "aren't", "weren't", "don't", "doesn't", "didn't"}
    labels = []
    for p in passages:
        toks = _TOKEN_RE.findall(p.lower())
        npos = nneg = 0
        flip = 0
        for tk in toks:
            if tk in NEGATORS:
                flip = 3
                continue
            is_p, is_n = tk in pos, tk in neg
            if is_p or is_n:
                if flip > 0:
                    is_p, is_n = is_n, is_p
                npos += int(is_p)
                nneg += int(is_n)
            flip = max(flip - 1, 0)
        labels.append("positive" if npos > nneg else
                      "negative" if nneg > npos else "neutral")
    return np.array(labels)


def article_score_from_labels(labels: np.ndarray) -> float:
    if labels.size == 0:
        return np.nan
    return float((labels == "positive").mean() - (labels == "negative").mean())


def score_corpus_dictionary(articles_csv=None,
                            out_csv=None,
                            checkpoint_dir=None,
                            chunk_rows: int = 20000,
                            time_budget_s=None,
                            verbose: bool = True) -> pd.DataFrame:
    """
    Stream the corpus and build a monthly Loughran-McDonald sentiment index.

    Chunked streaming keeps peak memory at a few hundred megabytes even though
    the corpus file is ~700 MB, which matters on a laptop.  Checkpointing and
    the optional `time_budget_s` mean the job can be stopped and restarted
    without losing work; see `score_corpus_finbert` for why that matters.
    """
    pos, neg = load_lm_words()
    if verbose:
        print(f"[lm] {len(pos)} positive and {len(neg)} negative terms")

    def scorer(chunk):
        out = np.full(len(chunk), np.nan)
        for j, (head, body) in enumerate(zip(chunk["headline"], chunk["body"])):
            text = clean_text((head or "") + ". " + (body or ""))
            labels = lm_passage_labels(split_passages(text), pos, neg)
            out[j] = article_score_from_labels(labels)
        return out

    return _score_corpus(scorer, articles_csv, out_csv,
                         checkpoint_dir or (OUT_DIR / "lm_parts"),
                         chunk_rows, time_budget_s, "lm", verbose)


def _score_corpus(scorer, articles_csv, out_csv, checkpoint_dir,
                  chunk_rows, time_budget_s, tag, verbose):
    """
    Shared driver for both sentiment scorers.

    Article-level scores are flushed to `checkpoint_dir/part_XXXXX.csv` every
    `chunk_rows` articles, and completed parts are skipped on a restart.  Set
    `time_budget_s` to stop cleanly after roughly that many seconds; call the
    function again to carry on from where it stopped.  This is what makes the
    job survive a Colab disconnect, a closed laptop lid, or a sandbox that
    kills long-running background processes.
    """
    import time
    src = Path(articles_csv) if articles_csv else _require(DATA_DIR / RAW["articles"])
    ck = Path(checkpoint_dir)
    ck.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stopped_early = False
    done_here = 0

    reader = pd.read_csv(src, chunksize=chunk_rows, dtype=str,
                         encoding="utf-8", encoding_errors="replace",
                         engine="c", on_bad_lines="skip")
    for part_no, chunk in enumerate(reader):
        fp = ck / f"part_{part_no:05d}.csv"
        if fp.exists():
            continue
        # At least one part is always completed per invocation, so a budget set
        # too low still makes progress instead of looping forever on nothing.
        if (done_here and time_budget_s is not None
                and (time.time() - t0) > time_budget_s):
            stopped_early = True
            if verbose:
                print(f"[{tag}] time budget reached at part {part_no}; "
                      f"re-run to continue", flush=True)
            break
        chunk = _prep_articles(chunk)
        scores = scorer(chunk)
        # Written to a temporary name first so an interrupted write can never
        # leave a truncated part that a later run would trust and skip.
        tmp = fp.with_suffix(".tmp")
        pd.DataFrame({"month": chunk["month"].astype(str),
                      "article_sent": scores}).to_csv(tmp, index=False)
        tmp.replace(fp)
        done_here += 1
        if verbose:
            print(f"[{tag}] part {part_no}: {len(chunk)} articles "
                  f"({(part_no + 1) * chunk_rows:,} read)", flush=True)

    parts = sorted(ck.glob("part_*.csv"))
    if not parts:
        raise RuntimeError(f"No checkpoint parts written in {ck}.")
    allp = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    allp = allp.dropna(subset=["article_sent"])
    agg = {}
    for per, s in zip(allp["month"], allp["article_sent"]):
        a = agg.setdefault(pd.Period(per, freq="M"), [0.0, 0])
        a[0] += float(s)
        a[1] += 1
    out = _finalise_monthly(agg)
    out.attrs["complete"] = not stopped_early
    if verbose:
        print(f"[{tag}] {len(parts)} parts, {len(out)} months, "
              f"{int(out['n_articles'].sum()):,} articles"
              f"{'  (INCOMPLETE)' if stopped_early else ''}")
    if out_csv and not stopped_early:
        _write_monthly(out, Path(out_csv))
    return out


# ------------------------------------------------------------------- FinBERT
def score_corpus_finbert(articles_csv=None,
                         out_csv=None,
                         checkpoint_dir=None,
                         chunk_rows: int = CHUNK_ROWS,
                         model_name: str = FINBERT_MODEL,
                         batch_size: int = FINBERT_BATCH,
                         max_len: int = FINBERT_MAX_LEN,
                         time_budget_s=None,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Score the whole corpus with FinBERT and build the monthly index.

    Resumable by design.  Every `chunk_rows` articles the article-level scores
    are flushed to `checkpoint_dir/part_XXXXX.csv`; on restart, completed parts
    are skipped.  A free Colab session will usually disconnect at least once
    during a run of this size, and without checkpointing that means starting
    from zero.

    Run this on a GPU.  On CPU it is roughly fifty times slower.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device).eval()

    # Read the label order from the model config rather than assuming it.
    # FinBERT's own ordering is [positive, negative, neutral], which is NOT
    # the alphabetical order a reader would guess.
    id2 = {int(k): str(v).lower() for k, v in mdl.config.id2label.items()}
    label_order = [id2[i] for i in range(mdl.config.num_labels)]
    if set(label_order) != {"positive", "negative", "neutral"}:
        raise ValueError(f"Unexpected FinBERT labels: {label_order}")
    if verbose:
        print(f"[finbert] device={device} labels={label_order}")
        if device == "cpu":
            print("[finbert] WARNING: no GPU detected. Enable one via "
                  "Runtime > Change runtime type in Colab.")

    label_arr = np.array(label_order)

    @torch.no_grad()
    def batch_labels(passages):
        """Argmax label for each passage. Hard classification keeps the
        article score interpretable and bounded in [-1, 1]."""
        out = []
        for i in range(0, len(passages), batch_size):
            enc = tok(passages[i:i + batch_size], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(device)
            logits = mdl(**enc).logits
            out.append(logits.argmax(dim=-1).cpu().numpy())
        return label_arr[np.concatenate(out)] if out else np.empty(0, dtype=object)

    def scorer(chunk):
        # Every passage in the chunk is flattened into one list so the GPU
        # always sees full batches. Scoring article by article leaves most of
        # the device idle and is several times slower.
        flat, owner = [], []
        for j, (head, body) in enumerate(zip(chunk["headline"], chunk["body"])):
            ps = split_passages(clean_text((head or "") + ". " + (body or "")))
            flat.extend(ps)
            owner.extend([j] * len(ps))
        scores = np.full(len(chunk), np.nan)
        if not flat:
            return scores
        labels = batch_labels(flat)
        owner = np.asarray(owner)
        for j in np.unique(owner):
            scores[j] = article_score_from_labels(labels[owner == j])
        return scores

    return _score_corpus(scorer, articles_csv, out_csv,
                         checkpoint_dir or (OUT_DIR / "finbert_parts"),
                         chunk_rows, time_budget_s, "finbert", verbose)


def _prep_articles(chunk: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw corpus chunk: parse dates, drop unusable rows."""
    need = {"date", "headline", "body"}
    if not need.issubset(chunk.columns):
        raise ValueError(
            f"Corpus must have columns {sorted(need)}; found {chunk.columns.tolist()}")
    chunk = chunk.copy()
    d = pd.to_datetime(chunk["date"], errors="coerce", format="mixed")
    chunk = chunk.loc[d.notna()].copy()
    chunk["month"] = to_month(d.loc[d.notna()])
    txt = chunk["headline"].fillna("").astype(str) + chunk["body"].fillna("").astype(str)
    return chunk.loc[txt.str.len() >= 40].reset_index(drop=True)


def _finalise_monthly(agg: dict) -> pd.DataFrame:
    if not agg:
        raise RuntimeError("No articles were scored.")
    idx = pd.PeriodIndex(sorted(agg.keys()), freq="M")
    out = pd.DataFrame({
        "sent": [agg[p][0] / agg[p][1] for p in idx],
        "n_articles": [agg[p][1] for p in idx],
    }, index=idx)
    out.index.name = "month"
    return out


def _write_monthly(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    o = df.copy()
    o.insert(0, "date", month_end_ts(o.index).strftime("%Y-%m-%d"))
    o.to_csv(path, index=False)


def read_monthly_sentiment(path) -> pd.Series:
    """Read a monthly sentiment CSV written by either scorer."""
    df = pd.read_csv(path)
    key = "date" if "date" in df.columns else "month"
    s = pd.Series(df["sent"].astype(float).values,
                  index=pd.PeriodIndex(pd.to_datetime(df[key]), freq="M"),
                  name="sent").sort_index()
    return s


# =============================================================================
# 6. FORECASTING
# =============================================================================

def _make_model(name: str):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "ols":
        pipe = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
        grid = {}                      # nothing to tune
    elif name == "enet":
        pipe = Pipeline([("sc", StandardScaler()),
                         ("m", ElasticNet(max_iter=50000, random_state=SEED))])
        # The premium has a standard deviation of about 0.04, so the
        # informative penalty range sits far below sklearn's default alpha=1.0.
        # The grid is nevertheless carried all the way up to 1.0: if the data
        # want total shrinkage, the model should be allowed to collapse onto
        # the historical mean rather than be held back by a binding grid edge.
        # l1_ratio starts at 0.05 (near-ridge) so the sentiment coefficient is
        # not automatically zeroed by the L1 penalty, which would make the
        # central nested comparison degenerate.
        grid = {"m__alpha": list(np.logspace(-6, 0, 13)),
                "m__l1_ratio": [0.05, 0.5, 0.95]}
    elif name == "rf":
        pipe = Pipeline([("m", RandomForestRegressor(
            n_estimators=300, random_state=SEED, n_jobs=1))])
        # Shallow trees and large leaves. Monthly returns have a very low
        # signal-to-noise ratio, so heavy regularisation is essential.
        grid = {"m__max_depth": [2, 3], "m__min_samples_leaf": [10, 25],
                "m__max_features": [0.5, 1.0]}
    elif name == "gbrt":
        pipe = Pipeline([("m", GradientBoostingRegressor(random_state=SEED))])
        grid = {"m__n_estimators": [100, 300], "m__learning_rate": [0.01, 0.05],
                "m__max_depth": [1, 2], "m__subsample": [0.7]}
    else:
        raise ValueError(f"Unknown model: {name}")
    return pipe, grid


def _tune(name: str, X: np.ndarray, y: np.ndarray):
    """Tune inside the training window only, respecting time order."""
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    pipe, grid = _make_model(name)
    if not grid:                       # nothing to tune, e.g. OLS
        pipe.fit(X, y)
        return pipe, {}
    folds = int(min(CV_FOLDS, max(2, len(X) // 30)))
    gs = GridSearchCV(pipe, grid, cv=TimeSeriesSplit(n_splits=folds),
                      scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


def prepare_design(panel: pd.DataFrame, sent: pd.Series | None) -> pd.DataFrame:
    """
    Attach sentiment and build the (predictors at t, premium at t+1) design.
    """
    df = panel.copy()
    if sent is not None:
        df[SENTIMENT_COL] = sent.reindex(df.index)
    have_sent = SENTIMENT_COL in df.columns and df[SENTIMENT_COL].notna().any()
    df["y_next"] = df[TARGET].shift(-1)
    need = (AUGMENTED_PREDICTORS if have_sent else BASELINE_PREDICTORS) + ["y_next"]
    return df.dropna(subset=need)


def run_forecasts(panel: pd.DataFrame,
                  sent: pd.Series | None = None,
                  oos_start: str = OOS_START,
                  models=None,
                  retune_every: int = RETUNE_EVERY,
                  impose_ct: bool = IMPOSE_CT_RESTRICTION,
                  ct_on_bench: bool = CT_ALSO_ON_BENCHMARK,
                  verbose: bool = True):
    """
    Expanding-window out-of-sample forecasts.

    THE TIMING RULE, which is where this kind of study most often goes wrong
    -----------------------------------------------------------------------
    Standing at the close of month t (row i), the forecaster knows the
    predictors for every month up to and including t, and the realised premium
    for every month up to and including t.  A training pair (X_s, y_{s+1})
    therefore requires s+1 <= i, i.e. s <= i-1, so the training set is
    df.iloc[:i] and NOT df.iloc[:i+1].  Including row i would put the premium
    of month t+1 -- the very quantity being forecast -- into the training data.
    The benchmark is the historical mean of the premium through month t.
    """
    models = list(models or MODELS)
    df = prepare_design(panel, sent)
    has_sent = SENTIMENT_COL in df.columns and df[SENTIMENT_COL].notna().all()
    specs = {"base": BASELINE_PREDICTORS}
    if has_sent:
        specs["aug"] = AUGMENTED_PREDICTORS

    periods = df.index
    start = pd.Period(oos_start, freq="M")
    oos_idx = np.where(periods >= start)[0]
    oos_idx = oos_idx[oos_idx >= 1]
    if oos_idx.size == 0:
        raise ValueError(f"No out-of-sample months at or after {oos_start}.")
    if oos_idx[0] < 60:
        raise ValueError(f"Only {oos_idx[0]} training months. Push oos_start later.")

    if verbose:
        print(f"[forecast] {len(df)} usable months | {oos_idx.size} forecasts "
              f"({periods[oos_idx[0]] + 1} to {periods[oos_idx[-1]] + 1}) | "
              f"specs={list(specs)}")

    rows, params_log = [], []
    fitted = {}
    for step, i in enumerate(oos_idx):
        train = df.iloc[:i]                     # see the timing rule above
        y_tr = train["y_next"].to_numpy(dtype=float)
        row = {
            "forecast_for": periods[i] + 1,
            "info_through": periods[i],
            "actual": float(df["y_next"].iloc[i]),
            "bench": float(df[TARGET].iloc[:i + 1].mean()),
            "n_train": int(len(train)),
        }
        retune = (step % retune_every == 0)
        for spec, cols in specs.items():
            X_tr = train[cols].to_numpy(dtype=float)
            X_te = df[cols].iloc[[i]].to_numpy(dtype=float)
            for name in models:
                key = (name, spec)
                if retune or key not in fitted:
                    est, best = _tune(name, X_tr, y_tr)
                    fitted[key] = est
                    params_log.append({"month": str(periods[i]), "model": name,
                                       "spec": spec, **best})
                # Hyper-parameters are held between re-tuning dates but
                # coefficients are re-estimated on the expanded window each month.
                est = fitted[key]
                est.fit(X_tr, y_tr)
                row[f"{name}_{spec}"] = float(est.predict(X_te)[0])
        rows.append(row)
        if verbose and step % 24 == 0:
            print(f"    {periods[i]}  ({step + 1}/{oos_idx.size})", flush=True)

    fc = pd.DataFrame(rows).set_index("forecast_for")
    fc.index = pd.PeriodIndex(fc.index, freq="M")

    if ADD_COMBINATION and len(models) > 1:
        for spec in specs:
            fc[f"combo_{spec}"] = fc[[f"{m}_{spec}" for m in models]].mean(axis=1)

    if impose_ct:
        mcols = [c for c in fc.columns if c.endswith(("_base", "_aug"))]
        fc[mcols] = fc[mcols].clip(lower=0.0)
        if ct_on_bench:
            fc["bench"] = fc["bench"].clip(lower=0.0)

    return fc, pd.DataFrame(params_log)


def model_names(fc: pd.DataFrame) -> list:
    """Model stems present in a forecast frame, in a stable order."""
    stems = [c[:-5] for c in fc.columns if c.endswith("_base")]
    order = ["ols", "enet", "rf", "gbrt", "combo"]
    return [s for s in order if s in stems] + [s for s in stems if s not in order]


# =============================================================================
# 7. STATISTICAL EVALUATION
# =============================================================================

def oos_r2(actual, f_large, f_small) -> float:
    """
    Campbell and Thompson (2008) out-of-sample R-squared: the proportional
    reduction in mean squared forecast error of the larger model relative to
    the smaller, nested one.  Positive means the larger model forecasts better.
    """
    a = np.asarray(actual, dtype=float)
    sse_l = np.sum((a - np.asarray(f_large, dtype=float)) ** 2)
    sse_s = np.sum((a - np.asarray(f_small, dtype=float)) ** 2)
    return float(1.0 - sse_l / sse_s)


def clark_west(actual, f_large, f_small, hac_lags: int = 0):
    """
    Clark and West (2007) MSPE-adjusted statistic for nested models.

        f_t = (a - s)^2 - [ (a - l)^2 - (s - l)^2 ]

    H0: the two models forecast equally well.  H1 (one-sided): the larger
    model is better.  For one-step-ahead forecasts the adjusted series is
    serially uncorrelated under the null, so hac_lags=0 is correct; a positive
    value is available as a robustness option.
    """
    a = np.asarray(actual, dtype=float)
    l = np.asarray(f_large, dtype=float)
    s = np.asarray(f_small, dtype=float)
    f = (a - s) ** 2 - ((a - l) ** 2 - (s - l) ** 2)

    # If the larger model has collapsed onto the smaller one, f is identically
    # zero and the statistic is undefined. Report that instead of crashing.
    if np.allclose(f, 0.0, atol=1e-15):
        return np.nan, np.nan

    n = f.size
    e = f - f.mean()
    # Regressing f on a constant makes the t-statistic mean / standard error,
    # so it is computed directly. This avoids a statsmodels dependency and is
    # numerically identical to OLS with HAC covariance.
    gamma0 = float(e @ e) / n
    var = gamma0
    if hac_lags and hac_lags > 0:
        for k in range(1, int(hac_lags) + 1):
            gk = float(e[k:] @ e[:-k]) / n
            var += 2.0 * (1.0 - k / (hac_lags + 1.0)) * gk    # Bartlett kernel
        var = max(var, 1e-30)
    else:
        var = gamma0 * n / (n - 1)                            # small-sample OLS
    se = np.sqrt(var / n)
    t = float(f.mean() / se)
    return t, float(_norm_sf(t))


def _norm_sf(x: float) -> float:
    """Upper-tail standard normal probability, via the error function."""
    import math
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def stars(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def statistical_table(fc: pd.DataFrame, hac_lags: int = 0) -> pd.DataFrame:
    """
    Two blocks.

      vs historical mean      does each model beat the Welch-Goyal benchmark?
      augmented vs baseline   THE CENTRAL TEST. Does sentiment add anything
                              once the traditional predictors are in?
    """
    a = fc["actual"].to_numpy(dtype=float)
    stems = model_names(fc)
    specs = ["base"] + (["aug"] if any(c.endswith("_aug") for c in fc.columns) else [])
    rows = []
    for spec in specs:
        for m in stems:
            f = fc[f"{m}_{spec}"].to_numpy(dtype=float)
            t, p = clark_west(a, f, fc["bench"].to_numpy(dtype=float), hac_lags)
            rows.append({"comparison": "vs historical mean", "model": m, "spec": spec,
                         "oos_r2_pct": 100 * oos_r2(a, f, fc["bench"]),
                         "cw_t": t, "cw_p": p, "sig": stars(p)})
    if "aug" in specs:
        for m in stems:
            big = fc[f"{m}_aug"].to_numpy(dtype=float)
            small = fc[f"{m}_base"].to_numpy(dtype=float)
            t, p = clark_west(a, big, small, hac_lags)
            rows.append({"comparison": "augmented vs baseline", "model": m,
                         "spec": "aug-vs-base",
                         "oos_r2_pct": 100 * oos_r2(a, big, small),
                         "cw_t": t, "cw_p": p, "sig": stars(p)})
    return pd.DataFrame(rows)


# =============================================================================
# 8. ECONOMIC EVALUATION
# =============================================================================

def variance_forecast(panel: pd.DataFrame, target_index: pd.PeriodIndex,
                      window: int = VOL_WINDOW,
                      min_periods: int = VOL_MIN_PERIODS) -> np.ndarray:
    """
    Rolling variance of the realised premium, using only months strictly
    before the target month.  Any remaining gap is filled by an EXPANDING
    mean, never by a full-sample mean, which would leak the future.
    """
    v = panel[TARGET].rolling(window, min_periods=min_periods).var(ddof=1).shift(1)
    v = v.reindex(target_index)
    fallback = panel[TARGET].expanding(min_periods=12).var(ddof=1).shift(1).reindex(target_index)
    v = v.where(v.notna(), fallback)
    return v.ffill().to_numpy(dtype=float)


def backtest(forecast, actual_eqp, rf, var_hat, cost=TRANSACTION_COST,
             gamma=GAMMA, w_min=W_MIN, w_max=W_MAX):
    """
    Mean-variance market timing.  Each month the investor holds

        w_t = (1 / gamma) * E_t[premium] / Var_t[premium],  clipped to [w_min, w_max]

    in equities and the remainder at the risk-free rate.  Transaction costs
    are charged on the traded amount |w_t - w_{t-1,end}|, where w_{t-1,end} is
    the weight AFTER the previous month's returns have moved it.  Ignoring
    that drift overstates turnover and therefore overstates costs.
    """
    f = np.asarray(forecast, dtype=float)
    a = np.asarray(actual_eqp, dtype=float)
    rf = np.asarray(rf, dtype=float)
    v = np.asarray(var_hat, dtype=float)

    w = np.clip((1.0 / gamma) * f / v, w_min, w_max)
    gross = np.empty_like(w)
    net = np.empty_like(w)
    turn = np.empty_like(w)
    w_end = 0.0
    for t in range(w.size):
        r_eq = a[t] + rf[t]
        r_p = rf[t] + w[t] * a[t]
        trade = abs(w[t] - w_end)
        gross[t] = r_p
        net[t] = r_p - cost * trade
        turn[t] = trade
        denom = 1.0 + r_p
        w_end = (w[t] * (1.0 + r_eq)) / denom if denom > 1e-12 else w[t]
    return gross, net, turn, w


def cer(returns, rf, gamma=GAMMA) -> float:
    """Annualised certainty-equivalent return of a mean-variance investor."""
    ex = np.asarray(returns, dtype=float) - np.asarray(rf, dtype=float)
    return float(12.0 * (np.mean(ex) - 0.5 * gamma * np.var(ex, ddof=1)))


def sharpe(returns, rf) -> float:
    ex = np.asarray(returns, dtype=float) - np.asarray(rf, dtype=float)
    sd = np.std(ex, ddof=1)
    return float(np.sqrt(12.0) * np.mean(ex) / sd) if sd > 0 else np.nan


def max_drawdown(returns) -> float:
    w = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    return float((w / np.maximum.accumulate(w) - 1.0).min())


def economic_table(fc: pd.DataFrame, panel: pd.DataFrame,
                   cost=TRANSACTION_COST, gamma=GAMMA,
                   w_max=W_MAX) -> pd.DataFrame:
    a = fc["actual"].to_numpy(dtype=float)
    rf = panel["rf"].reindex(fc.index).ffill().to_numpy(dtype=float)
    var_hat = variance_forecast(panel, fc.index)

    stems = model_names(fc)
    specs = ["base"] + (["aug"] if any(c.endswith("_aug") for c in fc.columns) else [])
    cols = ["bench"] + [f"{m}_{s}" for s in specs for m in stems]

    rows = []
    for col in cols:
        g, n, to, w = backtest(fc[col].to_numpy(dtype=float), a, rf, var_hat,
                               cost=cost, gamma=gamma, w_max=w_max)
        rows.append({"strategy": col,
                     "CER_gross_pct": 100 * cer(g, rf, gamma),
                     "CER_net_pct": 100 * cer(n, rf, gamma),
                     "Sharpe_gross": sharpe(g, rf), "Sharpe_net": sharpe(n, rf),
                     "ann_ret_net_pct": 100 * np.mean(n) * 12,
                     "ann_vol_pct": 100 * np.std(n, ddof=1) * np.sqrt(12),
                     "max_dd_pct": 100 * max_drawdown(n),
                     "avg_turnover": float(np.mean(to)),
                     "avg_weight": float(np.mean(w))})

    bh = a + rf
    rows.append({"strategy": "buy_and_hold",
                 "CER_gross_pct": 100 * cer(bh, rf, gamma),
                 "CER_net_pct": 100 * cer(bh, rf, gamma),
                 "Sharpe_gross": sharpe(bh, rf), "Sharpe_net": sharpe(bh, rf),
                 "ann_ret_net_pct": 100 * np.mean(bh) * 12,
                 "ann_vol_pct": 100 * np.std(bh, ddof=1) * np.sqrt(12),
                 "max_dd_pct": 100 * max_drawdown(bh),
                 "avg_turnover": 0.0, "avg_weight": 1.0})

    out = pd.DataFrame(rows)
    base = out.loc[out.strategy == "bench", "CER_net_pct"].iloc[0]
    out["CER_gain_vs_bench_pct"] = out["CER_net_pct"] - base
    return out


# =============================================================================
# 9. ROBUSTNESS
# =============================================================================

def subperiod_table(fc: pd.DataFrame, cuts=("2011-01", "2020-01", "2026-01"),
                    hac_lags: int = 0) -> pd.DataFrame:
    """Split the out-of-sample period so a single episode cannot drive everything."""
    rows = []
    bounds = [pd.Period(c, freq="M") for c in cuts]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sub = fc.loc[(fc.index >= lo) & (fc.index < hi)]
        if len(sub) < 12:
            continue
        st = statistical_table(sub, hac_lags=hac_lags)
        st.insert(0, "period", f"{sub.index.min()} to {sub.index.max()}")
        st.insert(1, "n", len(sub))
        rows.append(st)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def cost_gamma_grid(fc: pd.DataFrame, panel: pd.DataFrame,
                    costs=(0.0, 0.0005, 0.0010, 0.0025),
                    gammas=(2.0, 3.0, 5.0)) -> pd.DataFrame:
    """CER gain over the benchmark strategy across trading costs and risk aversion."""
    rows = []
    for c in costs:
        for g in gammas:
            t = economic_table(fc, panel, cost=c, gamma=g)
            base = t.loc[t.strategy == "bench", "CER_net_pct"].iloc[0]
            for _, r in t.iterrows():
                if r.strategy in ("bench", "buy_and_hold"):
                    continue
                rows.append({"cost_bps": 1e4 * c, "gamma": g,
                             "strategy": r.strategy,
                             "CER_net_pct": r.CER_net_pct,
                             "CER_gain_vs_bench_pct": r.CER_net_pct - base,
                             "Sharpe_net": r.Sharpe_net})
    return pd.DataFrame(rows)


def no_lookahead_test(panel: pd.DataFrame, sent: pd.Series | None,
                      oos_start: str = OOS_START, verbose: bool = True):
    """
    Truncate the sample immediately after a given forecast and re-run.  If the
    forecasts genuinely use only past information, deleting everything that
    comes after them must leave them numerically unchanged.

    Note on the earlier version of this test: keeping an extra month of data
    past the cut point preserves exactly the leak the test is supposed to
    detect, so the test passes whether or not the code is correct.  Here the
    cut keeps one extra month solely so the FINAL forecast still has a
    realised target, and the comparison then excludes that final forecast.
    """
    full, _ = run_forecasts(panel, sent, oos_start=oos_start, verbose=False)
    cut = full.index[len(full) // 2]                    # a target month
    keep = panel.index <= cut
    p_trunc = panel.loc[keep]
    s_trunc = sent.loc[sent.index <= cut] if sent is not None else None

    trunc, _ = run_forecasts(p_trunc, s_trunc, oos_start=oos_start, verbose=False)
    cols = [c for c in full.columns
            if c.endswith(("_base", "_aug")) or c == "bench"]
    common = full.index.intersection(trunc.index)
    common = common[common < trunc.index.max()]         # drop the truncated edge
    diff = (full.loc[common, cols] - trunc.loc[common, cols]).abs()
    worst = float(diff.to_numpy().max())
    ok = worst < 1e-10
    if verbose:
        print(f"[lookahead] cut after {cut}; compared {len(common)} forecasts "
              f"x {len(cols)} series")
        print(f"[lookahead] largest absolute discrepancy: {worst:.3e}")
        print("[lookahead] PASS, no future information is used." if ok else
              "[lookahead] FAIL\n" + diff.max().sort_values(ascending=False).head().to_string())
    return ok, worst


# =============================================================================
# 10. FIGURES
# =============================================================================

def _plt():
    """
    Return pyplot, forcing the non-interactive Agg backend ONLY when we are not
    inside IPython.  Hard-coding Agg in a notebook would silently disable
    inline figures for everything the user plots afterwards.
    """
    import matplotlib
    try:
        get_ipython  # noqa: F821  provided by IPython
        in_ipython = True
    except NameError:
        in_ipython = False
    if not in_ipython:
        try:
            import IPython
            in_ipython = IPython.get_ipython() is not None
        except Exception:
            in_ipython = False
    if not in_ipython:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt



def figure_cumulative_sse(fc: pd.DataFrame, path) -> None:
    """
    Goyal-Welch cumulative SSE difference: benchmark minus model.  An upward
    slope means the model is beating the historical mean over that stretch,
    which exposes predictability concentrated in a few episodes rather than
    spread through the sample.

    Two panels, because unconstrained OLS fails on a scale roughly twenty times
    larger than everything else and, on a single axis, flattens every other line
    onto zero.  Panel (a) keeps OLS so the magnitude of its failure is visible;
    panel (b) drops it so the regularised models can actually be read.
    """
    plt = _plt()

    a = fc["actual"].to_numpy(dtype=float)
    b = fc["bench"].to_numpy(dtype=float)
    x = month_end_ts(fc.index)
    stems = model_names(fc)
    has_aug = any(c.endswith("_aug") for c in fc.columns)
    colours = {"ols": "#8c2d19", "enet": "#1f3a68", "rf": "#2e7d5b",
               "gbrt": "#b5892a", "combo": "#5d3b8e"}

    def draw(ax, which):
        for m in which:
            for spec, style, lw, alpha in [("base", "--", 1.2, 0.8),
                                           ("aug", "-", 1.7, 1.0)]:
                col = f"{m}_{spec}"
                if col not in fc.columns or (spec == "aug" and not has_aug):
                    continue
                d = np.cumsum((a - b) ** 2 - (a - fc[col].to_numpy(dtype=float)) ** 2)
                ax.plot(x, d, style, lw=lw, alpha=alpha,
                        color=colours.get(m, None),
                        label=f"{m} {'augmented' if spec == 'aug' else 'baseline'}")
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7.5, ncol=2, frameon=False)

    others = [m for m in stems if m != "ols"]
    two = bool(others) and "ols" in stems
    fig, axes = plt.subplots(2 if two else 1, 1,
                             figsize=(9.5, 7.4 if two else 5.0), sharex=True)
    axes = np.atleast_1d(axes)
    draw(axes[0], stems)
    axes[0].set_title("Out-of-sample performance relative to the historical mean\n"
                      "cumulative SSE, benchmark minus model; upward slope "
                      "means the model is winning")
    axes[0].set_ylabel("(a) all models")
    if two:
        draw(axes[1], others)
        axes[1].set_ylabel("(b) excluding OLS")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_series(panel: pd.DataFrame, sent: pd.Series | None, path) -> None:
    """The premium, the sentiment index and the two most-watched predictors."""
    plt = _plt()

    x = month_end_ts(panel.index)
    n = 4 if sent is not None else 3
    fig, axes = plt.subplots(n, 1, figsize=(9.5, 2.2 * n), sharex=True)
    axes[0].plot(x, 100 * panel["eqp"], lw=0.9, color="#1f3a68")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("premium, %")
    axes[1].plot(x, 100 * panel["dy"], lw=1.2, color="#1f3a68")
    axes[1].set_ylabel("dividend yield, %")
    axes[2].plot(x, 100 * panel["term_spread"], lw=1.2, color="#1f3a68")
    axes[2].axhline(0, color="k", lw=0.6)
    axes[2].set_ylabel("term spread, %")
    if sent is not None:
        s = sent.reindex(panel.index)
        axes[3].plot(x, s, lw=1.2, color="#8c2d19")
        axes[3].axhline(float(s.mean()), color="k", lw=0.6, ls=":")
        axes[3].set_ylabel("news sentiment")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_title("UK equity premium, traditional predictors and news sentiment")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_weights(fc: pd.DataFrame, panel: pd.DataFrame, path,
                   stem: str = "combo") -> None:
    """Equity weight through time for the benchmark and the two specifications."""
    plt = _plt()

    a = fc["actual"].to_numpy(dtype=float)
    rf = panel["rf"].reindex(fc.index).ffill().to_numpy(dtype=float)
    v = variance_forecast(panel, fc.index)
    x = month_end_ts(fc.index)
    stems = model_names(fc)
    if stem not in stems:
        stem = stems[0]

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for col, style, lab in [("bench", ":", "benchmark"),
                            (f"{stem}_base", "--", f"{stem} baseline"),
                            (f"{stem}_aug", "-", f"{stem} augmented")]:
        if col not in fc.columns:
            continue
        _, _, _, w = backtest(fc[col].to_numpy(dtype=float), a, rf, v)
        ax.plot(x, w, style, lw=1.3, label=lab)
    ax.set_ylabel("weight in equities")
    ax.set_title("Market-timing allocations")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# =============================================================================
# 11. EXPORT
# =============================================================================

def _stamp(df: pd.DataFrame) -> pd.DataFrame:
    """Replace a monthly PeriodIndex with an ISO month-end date column."""
    o = df.copy()
    if isinstance(o.index, pd.PeriodIndex):
        o.insert(0, "date", month_end_ts(o.index).strftime("%Y-%m-%d"))
        o = o.reset_index(drop=True)
    return o


def save_all(out_dir=None, **frames) -> list:
    d = Path(out_dir or OUT_DIR)
    d.mkdir(parents=True, exist_ok=True)
    written = []
    for name, obj in frames.items():
        if obj is None:
            continue
        p = d / f"{name}.csv"
        if isinstance(obj, pd.Series):
            obj = obj.to_frame()
        _stamp(obj).to_csv(p, index=False)
        written.append(str(p))
    return written


def show(df: pd.DataFrame, nd: int = 3) -> str:
    with pd.option_context("display.width", 200, "display.max_columns", 60):
        return df.round(nd).to_string(index=False)
