"""
Measure how much the look-ahead bug in the original forecast.py was worth.

The original loop used
    train = df.iloc[:i + 1]          and   bench = train["y_next"].mean()
Row i of y_next holds the premium of month t+1, which is the forecast target,
so both the training data and the benchmark contained the answer.

This reruns the identical pipeline with the original timing and with the
corrected timing, and reports the difference in out-of-sample R-squared.
"""
import numpy as np
import pandas as pd

import ukep_core as U


def run_leaky(df, oos_start, retune_every=12):
    """The original timing, reproduced exactly."""
    specs = {"base": U.BASELINE_PREDICTORS, "aug": U.AUGMENTED_PREDICTORS}
    periods = df.index
    oos = np.where(periods >= pd.Period(oos_start, freq="M"))[0]
    rows, fitted = [], {}
    for step, i in enumerate(oos):
        train = df.iloc[:i + 1]                      # <-- the bug
        y_tr = train["y_next"].to_numpy(dtype=float)
        row = {"forecast_for": periods[i] + 1,
               "actual": float(df["y_next"].iloc[i]),
               "bench": float(y_tr.mean())}          # <-- and here
        retune = (step % retune_every == 0)
        for spec, cols in specs.items():
            X_tr = train[cols].to_numpy(dtype=float)
            X_te = df[cols].iloc[[i]].to_numpy(dtype=float)
            for name in U.MODELS:
                key = (name, spec)
                if retune or key not in fitted:
                    fitted[key] = U._tune(name, X_tr, y_tr)[0]
                est = fitted[key]
                est.fit(X_tr, y_tr)
                row[f"{name}_{spec}"] = float(est.predict(X_te)[0])
        rows.append(row)
    fc = pd.DataFrame(rows).set_index("forecast_for")
    fc.index = pd.PeriodIndex(fc.index, freq="M")
    for spec in specs:
        fc[f"combo_{spec}"] = fc[[f"{m}_{spec}" for m in U.MODELS]].mean(axis=1)
    mcols = [c for c in fc.columns if c.endswith(("_base", "_aug"))] + ["bench"]
    fc[mcols] = fc[mcols].clip(lower=0.0)            # original clipped bench too
    return fc


OOS = "2011-01"
panel = U.build_panel(verbose=False)
sent = U.read_monthly_sentiment("data/sentiment_monthly_lm.csv")
df = U.prepare_design(panel, sent)

print("running corrected timing ...", flush=True)
fixed, _ = U.run_forecasts(panel, sent, oos_start=OOS, verbose=False)
print("running original timing ...", flush=True)
leaky = run_leaky(df, OOS)

sf = U.statistical_table(fixed).set_index(["comparison", "model", "spec"])
sl = U.statistical_table(leaky).set_index(["comparison", "model", "spec"])
cmp = pd.DataFrame({"R2_original_pct": sl["oos_r2_pct"],
                    "R2_corrected_pct": sf["oos_r2_pct"]})
cmp["overstatement_pp"] = cmp["R2_original_pct"] - cmp["R2_corrected_pct"]
print("\n" + "=" * 78)
print("EFFECT OF THE LOOK-AHEAD BUG ON OUT-OF-SAMPLE R-SQUARED (percentage points)")
print("=" * 78)
print(cmp.round(3).to_string())

ef = U.economic_table(fixed, panel).set_index("strategy")["CER_net_pct"]
el = U.economic_table(leaky, panel).set_index("strategy")["CER_net_pct"]
ec = pd.DataFrame({"CER_original_pct": el, "CER_corrected_pct": ef})
ec["overstatement_pp"] = ec["CER_original_pct"] - ec["CER_corrected_pct"]
print("\nNET CERTAINTY EQUIVALENT (percent per year)")
print(ec.round(3).to_string())
cmp.to_csv("output/leak_effect_statistical.csv")
ec.to_csv("output/leak_effect_economic.csv")
print("\nwrote output/leak_effect_*.csv")
