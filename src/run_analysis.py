"""
=============================================================================
 UK EQUITY PREMIUM PREDICTABILITY WITH NEWS SENTIMENT
 Command-line driver. Produces every table and figure for Chapter 5.
=============================================================================

Usage
-----
    # 0. one-off: score the news corpus. Do this on a GPU (Colab).
    python run_analysis.py sentiment --finbert

    #    the dictionary version is fast and runs anywhere; it is the
    #    Loughran-McDonald robustness check
    python run_analysis.py sentiment --dictionary

    # 1. the main run
    python run_analysis.py all

    # useful variants
    python run_analysis.py all --sentiment output/sentiment_monthly_lm.csv
    python run_analysis.py all --oos-start 2015-01 --retune 24
    python run_analysis.py all --no-ct           # Campbell-Thompson off
    python run_analysis.py all --baseline-only   # no sentiment at all

Environment
-----------
    UKEP_DATA   folder holding the raw CSV files          (default: data)
    UKEP_OUT    folder for tables and figures             (default: output)

Everything written to UKEP_OUT is regenerated from scratch on each run, so the
folder is safe to delete.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

import ukep_core as U


# -----------------------------------------------------------------------------
def cmd_sentiment(args) -> None:
    out = Path(U.OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    if args.finbert:
        target = out / "sentiment_monthly_finbert.csv"
        res = U.score_corpus_finbert(articles_csv=args.articles,
                                     out_csv=target,
                                     time_budget_s=args.time_budget)
    else:
        target = out / "sentiment_monthly_lm.csv"
        res = U.score_corpus_dictionary(articles_csv=args.articles,
                                        out_csv=target,
                                        time_budget_s=args.time_budget)
    if not res.attrs.get("complete", True):
        print("\nStopped before the end of the corpus. Run the same command "
              "again to continue; finished chunks are not repeated.")
        sys.exit(0)
    print(f"\nWrote {target}")
    print(res.describe().round(4).to_string())


# -----------------------------------------------------------------------------
def _pick_sentiment(explicit) -> tuple:
    """Prefer FinBERT, fall back to the dictionary, else run without sentiment."""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Sentiment file not found: {p}")
        return U.read_monthly_sentiment(p), p.name
    for cand, tag in [(U.OUT_DIR / "sentiment_monthly_finbert.csv", "FinBERT"),
                      (U.DATA_DIR / "sentiment_monthly_finbert.csv", "FinBERT"),
                      (U.OUT_DIR / "sentiment_monthly_lm.csv", "Loughran-McDonald"),
                      (U.DATA_DIR / "sentiment_monthly_lm.csv", "Loughran-McDonald")]:
        if Path(cand).exists():
            return U.read_monthly_sentiment(cand), tag
    return None, "none"


def cmd_all(args) -> None:
    t0 = time.time()
    out = Path(U.OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("1. PANEL")
    print("=" * 78)
    panel = U.build_panel(rf_series=args.rf, lag_inflation=not args.no_cpi_lag)

    sent, tag = (None, "none") if args.baseline_only else _pick_sentiment(args.sentiment)
    print(f"\n[sentiment] source: {tag}")
    if sent is not None:
        cover = sent.reindex(panel.index).notna().mean()
        print(f"[sentiment] covers {cover:.1%} of panel months")
        if cover < 0.95:
            print("[sentiment] WARNING: coverage below 95%, months without a "
                  "sentiment value are dropped from the augmented model")

    desc = U.descriptive_table(panel, sent)
    print("\n" + "=" * 78)
    print("TABLE 5.1  Descriptive statistics")
    print("=" * 78)
    print(U.show(desc.reset_index(), 4))

    print("\n" + "=" * 78)
    print("2. OUT-OF-SAMPLE FORECASTS")
    print("=" * 78)
    fc, params = U.run_forecasts(panel, sent,
                                oos_start=args.oos_start,
                                retune_every=args.retune,
                                impose_ct=not args.no_ct)

    stat = U.statistical_table(fc, hac_lags=args.hac_lags)
    econ = U.economic_table(fc, panel, cost=args.cost, gamma=args.gamma)
    print("\n" + "=" * 78)
    print("TABLE 5.2  Statistical significance")
    print("=" * 78)
    print(U.show(stat))
    print("\n" + "=" * 78)
    print("TABLE 5.3  Economic significance")
    print("=" * 78)
    print(U.show(econ))

    print("\n" + "=" * 78)
    print("3. ROBUSTNESS")
    print("=" * 78)
    sub = U.subperiod_table(fc, hac_lags=args.hac_lags)
    if len(sub):
        print("\nTABLE 5.4  Sub-periods")
        print(U.show(sub))
    grid = U.cost_gamma_grid(fc, panel)
    print("\nTABLE 5.5  Trading costs and risk aversion (CER gain over benchmark, %)")
    piv = grid.pivot_table(index="strategy", columns=["gamma", "cost_bps"],
                           values="CER_gain_vs_bench_pct")
    print(piv.round(2).to_string())

    stat_ct_off = None
    if not args.no_ct and not args.fast:
        print("\n[robustness] re-running with the Campbell-Thompson floor removed")
        fc_off, _ = U.run_forecasts(panel, sent, oos_start=args.oos_start,
                                    retune_every=args.retune, impose_ct=False,
                                    verbose=False)
        stat_ct_off = U.statistical_table(fc_off, hac_lags=args.hac_lags)
        print("\nTABLE 5.6  No Campbell-Thompson restriction")
        print(U.show(stat_ct_off))

    ok = worst = None
    if not args.fast:
        print("\n[robustness] no-look-ahead test")
        ok, worst = U.no_lookahead_test(panel, sent, oos_start=args.oos_start)

    print("\n" + "=" * 78)
    print("4. FIGURES AND FILES")
    print("=" * 78)
    U.figure_series(panel, sent, out / "fig_5_1_series.png")
    U.figure_cumulative_sse(fc, out / "fig_5_2_cumulative_sse.png")
    U.figure_weights(fc, panel, out / "fig_5_3_weights.png")

    written = U.save_all(
        out,
        panel=panel,
        table_5_1_descriptives=desc.reset_index(),
        forecasts=fc,
        table_5_2_statistical=stat,
        table_5_3_economic=econ,
        table_5_4_subperiods=sub,
        table_5_5_cost_gamma=grid,
        table_5_6_no_ct=stat_ct_off,
        chosen_hyperparams=params,
    )
    for w in written:
        print("  wrote", w)
    for f in ["fig_5_1_series.png", "fig_5_2_cumulative_sse.png",
              "fig_5_3_weights.png"]:
        print("  wrote", out / f)

    # A one-line summary of the central result, so the headline number cannot
    # be misread off the wrong row of the wrong table.
    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    b = stat[(stat.comparison == "vs historical mean") & (stat.spec == "base")]
    print("baseline vs historical mean, best R2_OS: "
          f"{b.oos_r2_pct.max():+.3f}% ({b.loc[b.oos_r2_pct.idxmax(), 'model']})")
    a = stat[stat.comparison == "augmented vs baseline"]
    if len(a):
        best = a.loc[a.oos_r2_pct.idxmax()]
        print(f"sentiment vs baseline, best R2_OS: {best.oos_r2_pct:+.3f}% "
              f"({best.model}), Clark-West t = {best.cw_t:.2f}, "
              f"p = {best.cw_p:.3f} {best.sig}")
        print(f"number of positive aug-vs-base R2_OS: "
              f"{int((a.oos_r2_pct > 0).sum())} of {len(a)}")
    e = econ[econ.strategy != "buy_and_hold"]
    print("best net CER gain over benchmark: "
          f"{e.CER_gain_vs_bench_pct.max():+.3f}% per year "
          f"({e.loc[e.CER_gain_vs_bench_pct.idxmax(), 'strategy']})")
    if ok is not None:
        print(f"no-look-ahead test: {'PASS' if ok else 'FAIL'} (max diff {worst:.1e})")
    print(f"\ntotal runtime {time.time() - t0:.0f}s")


# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sentiment", help="score the news corpus")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--finbert", action="store_true")
    g.add_argument("--dictionary", action="store_true")
    s.add_argument("--articles", default=None)
    s.add_argument("--time-budget", type=float, default=None,
                   help="stop cleanly after this many seconds; re-run to continue")
    s.set_defaults(func=cmd_sentiment)

    a = sub.add_parser("all", help="run the full analysis")
    a.add_argument("--sentiment", default=None, help="path to a monthly sentiment CSV")
    a.add_argument("--baseline-only", action="store_true")
    a.add_argument("--oos-start", default=U.OOS_START)
    a.add_argument("--retune", type=int, default=U.RETUNE_EVERY)
    a.add_argument("--rf", default=U.RF_SERIES, choices=[U.RF_SERIES, U.RF_SERIES_ALT])
    a.add_argument("--cost", type=float, default=U.TRANSACTION_COST)
    a.add_argument("--gamma", type=float, default=U.GAMMA)
    a.add_argument("--hac-lags", type=int, default=0)
    a.add_argument("--no-ct", action="store_true",
                   help="drop the Campbell-Thompson non-negativity floor")
    a.add_argument("--no-cpi-lag", action="store_true",
                   help="use contemporaneous CPI (introduces publication look-ahead)")
    a.add_argument("--fast", action="store_true",
                   help="skip the two extra full re-runs")
    a.set_defaults(func=cmd_all)

    args = ap.parse_args()
    pd.set_option("display.width", 200)
    args.func(args)


if __name__ == "__main__":
    main()
