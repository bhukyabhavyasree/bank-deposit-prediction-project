#!/usr/bin/env python3
"""
Bank Marketing (UCI, Moro et al. 2014) - term-deposit subscription.

Usage (from the repository root):
    python analysis/run_analysis.py data/bank-additional-full.csv docs/data/results.json

Everything the website shows is produced here. Declared design (fixed before any model
was compared; the class balance of each block WAS looked at first, which is how the drift was
noticed, and the split was deliberately not changed afterwards):
  * Rows are in calendar order (May 2008 - Nov 2010). Primary split = chronological
    60% train / 20% validation / 20% test, by row position.
  * Selection metric on the validation block: average precision (PR-AUC).
  * Final models are refit on train+validation with the chosen configuration and scored
    on the test block exactly once.
  * `duration` and `campaign` are never used by the main models (see site, "Framing").
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
DATA = sys.argv[1] if len(sys.argv) > 1 else "data/bank-additional-full.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/data/results.json"
T0 = time.time()

# ----------------------------------------------------------------------------- data
raw = pd.read_csv(DATA, sep=";")
N = len(raw)
df = raw.copy()
df["y_bin"] = (df["y"] == "yes").astype(int)
df["pdays_never"] = (df["pdays"] == 999).astype(int)
df["pdays_days"] = np.where(df["pdays"] == 999, 0, df["pdays"])

# calendar year is derived ONLY for description (the file has no year column).
# The month sequence wraps backwards (dec -> mar) at each new year.
MONTH_NUM = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
             "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
mnum = df["month"].map(MONTH_NUM).to_numpy()
year = 2008 + np.concatenate([[0], np.cumsum(mnum[1:] < mnum[:-1])])
df["year"] = year

CATS = ["job", "marital", "education", "default", "housing", "loan", "contact",
        "month", "day_of_week", "poutcome"]
MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
BASE_NUM = ["age", "previous", "pdays_days", "pdays_never"]
FEATURE_SETS = {
    "pre_contact": BASE_NUM + MACRO,            # declared main set
    "pre_contact_no_macro": BASE_NUM,           # candidate chosen/rejected on validation
    "plus_campaign": BASE_NUM + MACRO + ["campaign"],                # diagnostic only
    "plus_campaign_duration": BASE_NUM + MACRO + ["campaign", "duration"],  # oracle, diagnostic
}
SELECTABLE = ["pre_contact", "pre_contact_no_macro"]

i1, i2 = int(0.6 * N), int(0.8 * N)
TR, VA, TE = np.arange(0, i1), np.arange(i1, i2), np.arange(i2, N)
TRVA = np.arange(0, i2)


def span(idx):
    s = df.iloc[idx]
    a, b = s.iloc[0], s.iloc[-1]
    return f"{a['month']} {a['year']} - {b['month']} {b['year']}"


# ------------------------------------------------------------------------- modelling
def build(model_name, params, fs):
    num = FEATURE_SETS[fs]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATS),
    ])
    if model_name == "logreg":
        clf = LogisticRegression(C=params["C"], max_iter=3000)
    else:
        clf = HistGradientBoostingClassifier(
            learning_rate=params["lr"], max_leaf_nodes=params["leaves"],
            l2_regularization=params["l2"], max_iter=params["iters"],
            min_samples_leaf=40, early_stopping=False, random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_predict(model_name, params, fs, fit_idx, eval_idx):
    """Preprocessing (scaling, one-hot categories) is fit inside the pipeline on fit_idx only."""
    m = build(model_name, params, fs)
    m.fit(df.iloc[fit_idx], df["y_bin"].iloc[fit_idx])
    return m.predict_proba(df.iloc[eval_idx])[:, 1]


def ap(y, s):
    return float(average_precision_score(y, s))


def auc(y, s):
    return float(roc_auc_score(y, s))


GRIDS = {
    "logreg": [{"C": c} for c in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]],
    "hgb": [{"lr": lr, "leaves": lv, "l2": l2, "iters": it}
            for lr in [0.05, 0.1] for lv in [8, 16, 31] for l2 in [0.0, 1.0] for it in [100, 200]],
}


def tune(fit_idx, val_idx, tag=""):
    """Grid search on the validation block ONLY. Returns table + best config per model."""
    yv = df["y_bin"].iloc[val_idx].to_numpy()
    table, best = [], {}
    for mn, grid in GRIDS.items():
        for fs in SELECTABLE:
            for p in grid:
                s = fit_predict(mn, p, fs, fit_idx, val_idx)
                row = {"model": mn, "feature_set": fs, "params": p,
                       "val_ap": ap(yv, s), "val_auc": auc(yv, s)}
                table.append(row)
                if mn not in best or row["val_ap"] > best[mn]["val_ap"]:
                    best[mn] = row
    print(f"  tuned{tag} in {time.time() - T0:.0f}s", flush=True)
    return table, best


def rule_scores(idx):
    return (df["poutcome"].iloc[idx] == "success").astype(float).to_numpy()


def tiebreak(s, seed=SEED):
    return s + np.random.default_rng(seed).uniform(0, 1e-7, len(s))


def topk(y, s, ks=(0.05, 0.10, 0.20, 0.30, 0.50)):
    s = tiebreak(s)
    order = np.argsort(-s)
    prev, out = y.mean(), []
    for k in ks:
        m = int(round(k * len(y)))
        tp = int(y[order[:m]].sum())
        out.append({"k": k, "calls": m, "hits": tp, "precision": tp / m,
                    "recall": tp / y.sum(), "lift": (tp / m) / prev})
    return out


# ========================================================================= PRIMARY RUN
print("primary (chronological) protocol", flush=True)
yte = df["y_bin"].iloc[TE].to_numpy()
ytr = df["y_bin"].iloc[TR].to_numpy()
yva = df["y_bin"].iloc[VA].to_numpy()

val_table, best = tune(TR, VA, " primary")

# final models: refit on train+validation with chosen config, score test ONCE
final_scores, train_only_test, train_only_val = {}, {}, {}
for mn in ["logreg", "hgb"]:
    b = best[mn]
    final_scores[mn] = fit_predict(mn, b["params"], b["feature_set"], TRVA, TE)
    train_only_val[mn] = fit_predict(mn, b["params"], b["feature_set"], TR, VA)
    train_only_test[mn] = fit_predict(mn, b["params"], b["feature_set"], TR, TE)
final_scores["rule"] = rule_scores(TE)
prior_rate = float(ytr.mean())

# ---- headline test metrics + bootstrap CIs (iid bootstrap; see site for caveat)
rng = np.random.default_rng(SEED)
B = 1000
names = ["rule", "logreg", "hgb"]
boots = {n: {"ap": [], "auc": []} for n in names}
for _ in range(B):
    ix = rng.integers(0, len(yte), len(yte))
    if yte[ix].sum() == 0:
        continue
    for n in names:
        s = tiebreak(final_scores[n][ix], rng.integers(1e9))
        boots[n]["ap"].append(ap(yte[ix], s))
        boots[n]["auc"].append(auc(yte[ix], s))


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def paired(a, b, key):
    # re-run pairing with identical resamples
    return None


# paired differences with identical resamples
rng = np.random.default_rng(SEED + 1)
diff = {"hgb_minus_logreg": [], "logreg_minus_rule": [], "hgb_minus_rule": []}
for _ in range(B):
    ix = rng.integers(0, len(yte), len(yte))
    if yte[ix].sum() == 0:
        continue
    t = rng.integers(1e9)
    a = {n: ap(yte[ix], tiebreak(final_scores[n][ix], t)) for n in names}
    diff["hgb_minus_logreg"].append(a["hgb"] - a["logreg"])
    diff["logreg_minus_rule"].append(a["logreg"] - a["rule"])
    diff["hgb_minus_rule"].append(a["hgb"] - a["rule"])

test_metrics = {}
for n in names:
    s = final_scores[n]
    test_metrics[n] = {
        "ap": ap(yte, tiebreak(s)), "ap_ci": ci(boots[n]["ap"]),
        "auc": auc(yte, tiebreak(s)), "auc_ci": ci(boots[n]["auc"]),
        "brier": float(brier_score_loss(yte, s)) if n != "rule" else None,
        "topk": topk(yte, s),
        "mean_score": float(s.mean()),
    }
test_metrics["prior"] = {"ap": float(yte.mean()), "auc": 0.5, "constant_score": prior_rate,
                         "brier": float(brier_score_loss(yte, np.full(len(yte), prior_rate)))}
paired_diffs = {k: {"mean": float(np.mean(v)), "ci": ci(v),
                    "share_positive": float(np.mean(np.array(v) > 0))} for k, v in diff.items()}

# ---- validation-block metrics of the chosen (train-only) models, for the tuning story
val_metrics = {"prior_ap": float(yva.mean())}
for mn in ["logreg", "hgb"]:
    val_metrics[mn] = {"ap": ap(yva, train_only_val[mn]), "auc": auc(yva, train_only_val[mn])}
val_metrics["rule"] = {"ap": ap(yva, tiebreak(rule_scores(VA))),
                       "auc": auc(yva, tiebreak(rule_scores(VA)))}

# ---- calibration on test
def calib(s, y, bins=10):
    q = np.quantile(s, np.linspace(0, 1, bins + 1))
    q[-1] += 1e-9
    out = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (s >= lo) & (s < hi)
        if m.sum():
            out.append({"pred": float(s[m].mean()), "obs": float(y[m].mean()), "n": int(m.sum())})
    return out


calibration = {mn: calib(final_scores[mn], yte) for mn in ["logreg", "hgb"]}

# ---- frozen-threshold check: threshold chosen on validation (F1-max), train-only model
frozen = {}
for mn in ["logreg", "hgb"]:
    p, r, t = precision_recall_curve(yva, train_only_val[mn])
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    thr = float(t[int(np.argmax(f1))])
    s = train_only_test[mn]
    pred = s >= thr
    tp = int((pred & (yte == 1)).sum())
    prec = tp / max(pred.sum(), 1)
    rec = tp / yte.sum()
    # best-possible F1 on test, for contrast (oracle, not a valid choice)
    p2, r2, t2 = precision_recall_curve(yte, s)
    f2 = 2 * p2[:-1] * r2[:-1] / np.clip(p2[:-1] + r2[:-1], 1e-12, None)
    frozen[mn] = {
        "threshold": thr, "val_f1": float(f1.max()),
        "val_pred_pos_share": float((train_only_val[mn] >= thr).mean()),
        "val_prevalence": float(yva.mean()),
        "test_pred_pos_share": float(pred.mean()), "test_prevalence": float(yte.mean()),
        "test_precision": float(prec), "test_recall": float(rec),
        "test_f1": float(2 * prec * rec / max(prec + rec, 1e-12)),
        "oracle_test_f1": float(f2.max()), "oracle_threshold": float(t2[int(np.argmax(f2))]),
        "test_ap_train_only": ap(yte, train_only_test[mn]),
        "test_auc_train_only": auc(yte, train_only_test[mn]),
    }

# ---- ablations (hyperparameters fixed to the chosen ones; diagnostics, NOT selection)
ablation = []
for fs in ["pre_contact", "pre_contact_no_macro", "plus_campaign", "plus_campaign_duration"]:
    row = {"feature_set": fs}
    for mn in ["logreg", "hgb"]:
        s = fit_predict(mn, best[mn]["params"], fs, TRVA, TE)
        row[mn] = {"ap": ap(yte, s), "auc": auc(yte, s)}
    ablation.append(row)
print(f"  ablations done {time.time() - T0:.0f}s", flush=True)

# ---- rolling-origin (expanding window, next 10% block), fixed chosen configs
rolling = []
for o in [0.5, 0.6, 0.7, 0.8, 0.9]:
    a, b = int(o * N), min(int((o + 0.1) * N), N)
    tr_i, te_i = np.arange(0, a), np.arange(a, b)
    y = df["y_bin"].iloc[te_i].to_numpy()
    row = {"origin": o, "train_n": len(tr_i), "test_n": len(te_i), "span": span(te_i),
           "train_prev": float(df["y_bin"].iloc[tr_i].mean()), "test_prev": float(y.mean())}
    for mn in ["logreg", "hgb"]:
        s = fit_predict(mn, best[mn]["params"], best[mn]["feature_set"], tr_i, te_i)
        row[mn] = {"ap": ap(y, s), "auc": auc(y, s)}
    rs = tiebreak(rule_scores(te_i))
    row["rule"] = {"ap": ap(y, rs), "auc": auc(y, rs)}
    rolling.append(row)

# ---- performance by consecutive chunk of the test block (final models)
chunks = []
for c in np.array_split(np.arange(len(TE)), 4):
    y = yte[c]
    row = {"span": span(TE[c]), "n": len(c), "prevalence": float(y.mean())}
    for n in names:
        s = tiebreak(final_scores[n][c])
        row[n] = {"ap": ap(y, s), "auc": auc(y, s)}
    chunks.append(row)

# ---- random stratified split (what you would get if time were ignored), 3 seeds
random_split = []
for seed in [1, 2, 3]:
    idx = np.arange(N)
    a, tmp = train_test_split(idx, test_size=0.4, stratify=df["y_bin"], random_state=seed)
    v, t = train_test_split(tmp, test_size=0.5, stratify=df["y_bin"].iloc[tmp], random_state=seed)
    a, v, t = np.sort(a), np.sort(v), np.sort(t)
    tbl, bst = tune(a, v, f" random seed {seed}")
    y = df["y_bin"].iloc[t].to_numpy()
    row = {"seed": seed, "test_prev": float(y.mean()),
           "chosen": {mn: {"feature_set": bst[mn]["feature_set"], "params": bst[mn]["params"]}
                      for mn in bst}}
    for mn in ["logreg", "hgb"]:
        s = fit_predict(mn, bst[mn]["params"], bst[mn]["feature_set"], np.concatenate([a, v]), t)
        row[mn] = {"ap": ap(y, s), "auc": auc(y, s)}
    rs = tiebreak(rule_scores(t))
    row["rule"] = {"ap": ap(y, rs), "auc": auc(y, rs)}
    random_split.append(row)

# ========================================================================= EDA (descriptive)
tr = df.iloc[TR]


def rate_table(col, data=tr, order=None, min_n=0):
    g = data.groupby(col)["y_bin"].agg(["size", "mean"]).reset_index()
    g = g[g["size"] >= min_n]
    if order:
        g["o"] = g[col].map({k: i for i, k in enumerate(order)})
        g = g.sort_values("o")
    else:
        g = g.sort_values("mean", ascending=False)
    return [{"label": str(r[col]), "n": int(r["size"]), "rate": float(r["mean"])}
            for _, r in g.iterrows()]


age_bins = [17, 25, 30, 35, 40, 50, 60, 100]
age_lab = ["<=25", "26-30", "31-35", "36-40", "41-50", "51-60", "61+"]
tr_age = tr.assign(age_bin=pd.cut(tr["age"], age_bins, labels=age_lab))
tr_camp = tr.assign(camp_bin=pd.cut(tr["campaign"], [0, 1, 2, 3, 5, 10, 100],
                                    labels=["1", "2", "3", "4-5", "6-10", "11+"]))
tr_dur = tr.assign(dur_bin=pd.cut(tr["duration"], [-1, 60, 120, 240, 480, 960, 10000],
                                  labels=["<1m", "1-2m", "2-4m", "4-8m", "8-16m", "16m+"]))
macro_corr = tr[MACRO].corr().round(3)

# timeline over the whole file (describes the drift; no modelling decision is based on it)
timeline = []
for k, c in enumerate(np.array_split(np.arange(N), 41)):
    s = df.iloc[c]
    timeline.append({"start": int(c[0]), "end": int(c[-1]), "rate": float(s["y_bin"].mean()),
                     "euribor": float(s["euribor3m"].mean()),
                     "label": f"{s['month'].mode().iloc[0]} {int(s['year'].mode().iloc[0])}",
                     "split": "train" if c[-1] < i1 else ("val" if c[-1] < i2 else "test")})

year_rows = {int(y): {"rows": int((df["year"] == y).sum()),
                      "rate": float(df.loc[df["year"] == y, "y_bin"].mean())}
             for y in sorted(df["year"].unique())}
unknowns = {c: int((raw[c] == "unknown").sum()) for c in CATS if (raw[c] == "unknown").any()}
quality = {
    "rows": N, "columns": int(raw.shape[1]), "missing_cells": int(raw.isna().sum().sum()),
    "exact_duplicate_rows": int(raw.duplicated().sum()),
    "unknown_counts": unknowns,
    "pdays_999_share": float((raw["pdays"] == 999).mean()),
    "default_yes": int((raw["default"] == "yes").sum()),
    "campaign_max": int(raw["campaign"].max()), "campaign_p99": float(raw["campaign"].quantile(.99)),
    "previous_nonzero_share": float((raw["previous"] > 0).mean()),
    "age_min": int(raw["age"].min()), "age_max": int(raw["age"].max()),
    "duration_zero": int((raw["duration"] == 0).sum()),
}
split_info = {
    "n": N, "train": {"rows": len(TR), "span": span(TR), "prevalence": float(ytr.mean()),
                      "euribor_mean": float(df.iloc[TR]["euribor3m"].mean())},
    "val": {"rows": len(VA), "span": span(VA), "prevalence": float(yva.mean()),
            "euribor_mean": float(df.iloc[VA]["euribor3m"].mean())},
    "test": {"rows": len(TE), "span": span(TE), "prevalence": float(yte.mean()),
             "euribor_mean": float(df.iloc[TE]["euribor3m"].mean())},
    "train_months": sorted(df.iloc[TR]["month"].unique().tolist(), key=lambda m: MONTH_NUM[m]),
    "test_months": sorted(df.iloc[TE]["month"].unique().tolist(), key=lambda m: MONTH_NUM[m]),
    "index_cuts": [i1, i2],
}
# macro range shift (train vs test)
split_info["euribor_range_train"] = [float(df.iloc[TR]["euribor3m"].min()), float(df.iloc[TR]["euribor3m"].max())]
split_info["euribor_range_test"] = [float(df.iloc[TE]["euribor3m"].min()), float(df.iloc[TE]["euribor3m"].max())]
split_info["test_rows_outside_train_euribor_range"] = float(
    ((df.iloc[TE]["euribor3m"] < split_info["euribor_range_train"][0]) |
     (df.iloc[TE]["euribor3m"] > split_info["euribor_range_train"][1])).mean())

eda = {
    "timeline": timeline, "year_rows": year_rows,
    "train_rates": {
        "job": rate_table("job", min_n=30), "poutcome": rate_table("poutcome"),
        "contact": rate_table("contact"), "marital": rate_table("marital", min_n=30),
        "education": rate_table("education", min_n=30),
        "month": rate_table("month", order=list(MONTH_NUM)),
        "age": rate_table("age_bin", data=tr_age, order=age_lab),
        "campaign": rate_table("camp_bin", data=tr_camp, order=["1", "2", "3", "4-5", "6-10", "11+"]),
        "duration": rate_table("dur_bin", data=tr_dur, order=["<1m", "1-2m", "2-4m", "4-8m", "8-16m", "16m+"]),
    },
    "duration_auc_train": auc(ytr, tr["duration"]),
    "campaign_auc_train_inverted": auc(ytr, -tr["campaign"]),
    "macro_corr": {"labels": MACRO, "matrix": macro_corr.values.tolist()},
    "quality": quality,
    "dictionary": {
        "age": "Client age in years", "job": "Occupation (12 categories incl. unknown)",
        "marital": "Marital status", "education": "Education level",
        "default": "Has credit in default? (mostly 'unknown')",
        "housing": "Has a housing loan?", "loan": "Has a personal loan?",
        "contact": "Contact channel: cellular or telephone", "month": "Month of the last contact",
        "day_of_week": "Weekday of the last contact",
        "duration": "Last call length in seconds - known only after the call",
        "campaign": "Contacts made in this campaign for this client, including the last",
        "pdays": "Days since previous campaign contact (999 = never contacted)",
        "previous": "Contacts before this campaign", "poutcome": "Outcome of the previous campaign",
        "emp.var.rate": "Employment variation rate (quarterly)",
        "cons.price.idx": "Consumer price index (monthly)", "cons.conf.idx": "Consumer confidence index (monthly)",
        "euribor3m": "Euribor 3-month rate (daily)", "nr.employed": "Number employed (quarterly, thousands)",
        "y": "Target: subscribed a term deposit (yes/no)"},
}

# ================================================================== ERROR ANALYSIS
# the error analysis follows the model that validation selected (not the one that looks best on test)
PRIMARY = max(["logreg", "hgb"], key=lambda m: best[m]["val_ap"])
te = df.iloc[TE].reset_index(drop=True)
s_final = tiebreak(final_scores[PRIMARY])
K = 0.20
m_calls = int(round(K * len(te)))
order = np.argsort(-s_final)
called = np.zeros(len(te), bool)
called[order[:m_calls]] = True
rank = np.empty(len(te), int)
rank[order] = np.arange(len(te))
pct = 1 - rank / len(te)  # 1 = highest score


def ex(i):
    r = te.iloc[i]
    return {"row": int(TE[i]), "when": f"{r['month']} {int(r['year'])}", "job": r["job"],
            "age": int(r["age"]), "poutcome": r["poutcome"], "contact": r["contact"],
            "education": r["education"], "euribor": float(r["euribor3m"]),
            "prev_contacts": int(r["previous"]), "score": float(s_final[i]),
            "score_percentile": float(pct[i]) * 100}


fp = [i for i in order if called[i] and yte[i] == 0][:8]
fn = [i for i in order[::-1] if (not called[i]) and yte[i] == 1][:8]
rng2 = np.random.default_rng(SEED)
fn_all = np.where((~called) & (yte == 1))[0]
fp_all = np.where(called & (yte == 0))[0]
typical_fn = rng2.choice(fn_all, 6, replace=False).tolist()
typical_fp = rng2.choice(fp_all, 6, replace=False).tolist()


def seg(col, bins=None):
    d = te.assign(called=called, y=yte)
    out = []
    for k, g in d.groupby(col):
        out.append({"label": str(k), "n": int(len(g)), "subscribers": int(g["y"].sum()),
                    "called": int(g["called"].sum()),
                    "captured": int((g["called"] & (g["y"] == 1)).sum()),
                    "recall": float((g["called"] & (g["y"] == 1)).sum() / max(g["y"].sum(), 1)),
                    "precision": float((g["called"] & (g["y"] == 1)).sum() / max(g["called"].sum(), 1))})
    return sorted(out, key=lambda r: -r["n"])


tp_n = int((called & (yte == 1)).sum())
errors = {
    "operating_point": {"budget": K, "calls": m_calls, "tp": tp_n, "fp": int((called & (yte == 0)).sum()),
                        "fn": int((~called & (yte == 1)).sum()), "tn": int((~called & (yte == 0)).sum()),
                        "precision": tp_n / m_calls, "recall": tp_n / int(yte.sum()),
                        "score_cutoff": float(s_final[order[m_calls - 1]])},
    "worst_false_positives": [ex(i) for i in fp],
    "worst_false_negatives": [ex(i) for i in fn],
    "typical_false_positives": [ex(i) for i in typical_fp],
    "typical_false_negatives": [ex(i) for i in typical_fn],
    "segments": {"poutcome": seg("poutcome"), "contact": seg("contact"), "year": seg("year"),
                 "month": seg("month"), "job": seg("job")[:8]},
    "fn_share_poutcome_nonexistent": float((te.loc[fn_all, "poutcome"] == "nonexistent").mean()),
    "subscribers_nonexistent_share": float((te.loc[yte == 1, "poutcome"] == "nonexistent").mean()),
}

# ============================================================ ROW-LEVEL EXPORT (slider)
def code(col, vals=None):
    vals = vals or sorted(te[col].unique().tolist())
    return vals, te[col].map({v: i for i, v in enumerate(vals)}).astype(int).tolist()


months = [m for m in MONTH_NUM if m in te["month"].unique()]
mv, mc = code("month", months)
jv, jc = code("job")
pv, pc = code("poutcome")
cv, cc = code("contact")
yv, yc = code("year")
rows = {
    "n": len(te), "y": yte.tolist(),
    "logreg": np.round(final_scores["logreg"], 4).tolist(),
    "hgb": np.round(final_scores["hgb"], 4).tolist(),
    "rule": final_scores["rule"].astype(int).tolist(),
    "dims": {"month": {"values": mv, "codes": mc}, "job": {"values": jv, "codes": jc},
             "poutcome": {"values": pv, "codes": pc}, "contact": {"values": cv, "codes": cc},
             "year": {"values": [int(v) for v in yv], "codes": yc}},
}

out = {
    "meta": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
             "seed": SEED, "bootstrap_resamples": B, "runtime_s": round(time.time() - T0)},
    "features": {"categorical": CATS, "macro": MACRO, "base_numeric": BASE_NUM,
                 "sets": {k: v for k, v in FEATURE_SETS.items()}},
    "grids": GRIDS, "split": split_info, "quality": quality, "eda": eda,
    "tuning": {"table": val_table, "best": best, "val_metrics": val_metrics},
    "test": {"metrics": test_metrics, "paired_diffs": paired_diffs, "calibration": calibration,
             "frozen_threshold": frozen, "ablation": ablation, "chunks": chunks},
    "rolling_origin": rolling, "random_split": random_split,
    "primary_model": PRIMARY, "errors": errors, "rows": rows,
}
with open(OUT, "w") as f:
    json.dump(out, f, separators=(",", ":"))
# same payload as a script, so the site also opens from a double-click (file:// blocks fetch)
with open(OUT.replace(".json", ".js"), "w") as f:
    f.write("window.RESULTS=" + json.dumps(out, separators=(",", ":")) + ";\n")
print(f"wrote {OUT} (+ .js) in {time.time() - T0:.0f}s")
