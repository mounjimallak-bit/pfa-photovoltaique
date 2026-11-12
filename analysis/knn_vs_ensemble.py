"""Why does a 10-nearest-neighbour distance on six raw sensors beat the deployed ensemble on test PR-AUC?

Read-only diagnostic (writes no file). Run from the repository root:

    python analysis/knn_vs_ensemble.py

It reuses the split of notebook section 16.1 and the deployed detector (src/detecteur.py).
Every reading of the test set below is a posteriori: it explains a gap, it selects nothing.
Runtime: about 2 to 4 minutes on a laptop CPU.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detecteur import DetecteurPV  # noqa: E402

FEATURES = ["GTI", "Pg", "Va", "Vg", "Ia", "TPV"]
GAP = pd.Timedelta("7min")
pd.set_option("display.width", 200)


# ----------------------------------------------------------------------------- data (as in section 16.1)
def load_split():
    df = pd.read_csv(ROOT / "data/merged_5min_day.csv", parse_dates=["time"], index_col="time")
    df = df[~((df["Va"] == 0) & (df["Vg"] == 0) & (df["Fg"] == 0))]
    end = df.loc[df["Fault"] != 0].index.max() + pd.Timedelta(days=7)
    study, unused = df.loc[:end], df.loc[df.index > end]
    d = study.index.date
    cut_train, cut_val = pd.Timestamp("2022-03-22").date(), pd.Timestamp("2022-04-28").date()
    train, val, test = study[d <= cut_train], study[(d > cut_train) & (d <= cut_val)], study[d > cut_val]
    assert (len(train), len(val), len(test)) == (10_863, 4_219, 2_975)
    return df, train, val, test, unused


# ----------------------------------------------------------------------------- deployed scores (as in section 16.2)
def deployed_scores(det, d):
    d = d[d[FEATURES].notna().all(axis=1)]
    s_ae = pd.Series(det._err_ae(det.sc_ae.transform(d[FEATURES])), index=d.index)
    X, idx = [], []
    for _, b in d.groupby((d.index.to_series().diff() > GAP).cumsum()):
        v = b[FEATURES].values
        for st in range(len(b) - det.seq_len + 1):
            X.append(v[st:st + det.seq_len]); idx.append(b.index[st + det.seq_len - 1])
    X = np.array(X)
    s_l = pd.Series(det._err_lstm(det.sc_lstm.transform(X.reshape(-1, len(FEATURES))).reshape(X.shape)),
                    index=pd.DatetimeIndex(idx))
    ic = s_ae.index.intersection(s_l.index)
    out = pd.DataFrame({"ae": det._rang(s_ae.loc[ic].values, det.ref_ae),
                        "lstm": det._rang(s_l.loc[ic].values, det.ref_lstm)}, index=ic)
    out["fusion"] = (out["ae"] + out["lstm"]) / 2
    out["y"] = (d.loc[ic, "Fault"] != 0).astype(int)
    return out.sort_index()


def knn_score(fit, X, k=10, scaler="standard", agg="mean"):
    sc = (StandardScaler() if scaler == "standard" else MinMaxScaler()).fit(fit)
    dist = NearestNeighbors(n_neighbors=k).fit(sc.transform(fit)).kneighbors(sc.transform(X))[0]
    return dist.mean(axis=1) if agg == "mean" else dist[:, -1]


# ----------------------------------------------------------------------------- tools (as in section 16)
def f2_threshold(s, y):
    p, r, t = precision_recall_curve(y, s); p, r = p[:-1], r[:-1]
    return float(t[np.argmax(5 * p * r / (4 * p + r + 1e-9))])


def blocks(index, mask):
    mask = np.asarray(mask, bool)
    night = np.r_[True, np.diff(index.values) > GAP.to_timedelta64()]
    start = mask & (night | ~np.r_[False, mask[:-1]])
    num = np.cumsum(start)
    return [np.flatnonzero((num == k) & mask) for k in range(1, num.max() + 1)] if mask.any() else []


def summary(s, col, thr):
    a, y = (s[col] >= thr).values, s["y"].values
    tp, fp = int((a & (y == 1)).sum()), int((a & (y == 0)).sum())
    eps = blocks(s.index, y == 1)
    return dict(precision=tp / max(tp + fp, 1), recall=tp / max(y.sum(), 1),
                episodes=f"{sum(a[e].any() for e in eps)}/{len(eps)}", fp=fp)


def events_per_week(s, col, thr):
    return 7 * len(blocks(s.index, (s[col] >= thr).values)) / len(np.unique(s.index.date))


def budget_threshold(s, col, budget):
    """Lowest threshold giving at most `budget` alarm events per week on s (no label used)."""
    grid = np.unique(np.quantile(s[col], np.linspace(0.80, 1, 4001)))
    return float(min(t for t in grid if events_per_week(s, col, t) <= budget))


def partial_auc(y, s, fpr_max):
    f, tp, _ = roc_curve(y, s); keep = f <= fpr_max
    return np.trapezoid(np.r_[tp[keep], np.interp(fpr_max, f, tp)], np.r_[f[keep], fpr_max]) / fpr_max


# ----------------------------------------------------------------------------- main
df, train, val, test, unused = load_split()
det = DetecteurPV(verbeux=False)
S = {name: deployed_scores(det, part) for name, part in [("val", val), ("test", test), ("unused", unused)]}
healthy = train.loc[train["Fault"] == 0, FEATURES].dropna()
assert len(healthy) == 9_618
for name in S:
    S[name]["knn"] = knn_score(healthy, df.loc[S[name].index, FEATURES])
v, t, u = S["val"], S["test"], S["unused"]
METHODS = {"knn": "kNN (k=10)", "ae": "Dense AE x5", "lstm": "LSTM AE x5", "fusion": "Fusion (deployed)"}

print("1. Same points, same labels, same metric")
print(f"   val {len(v)} points / {v.y.sum()} faults | test {len(t)} points / {t.y.sum()} faults")
print("   PR-AUC test:", {METHODS[m]: round(float(average_precision_score(t.y, t[m])), 3) for m in METHODS})
print("\n2. Where does the gap come from? (test)")
days = t.index.normalize(); fault_days = np.isin(days, list(set(days[t.y == 1])))
rows = []
for m in METHODS:
    med = t.loc[t.y == 1, m].median()
    intruders = (~fault_days) & (t[m] > med).values
    rows.append({"method": METHODS[m],
                 "PR-AUC, 3 fault days": average_precision_score(t.y[fault_days], t[m][fault_days]),
                 "PR-AUC, 27 days": average_precision_score(t.y, t[m]),
                 "ROC-AUC, 27 days": roc_auc_score(t.y, t[m]),
                 "healthy points of other days above the fault median": int(intruders.sum()),
                 "on n days": len(set(days[intruders]))})
print(pd.DataFrame(rows).set_index("method").round(3).to_string())

print("\n3. Fragility: PR-AUC after removing the few 'intruders' of one method")
for m in ("fusion", "knn"):
    keep = ~((~fault_days) & (t[m] > t.loc[t.y == 1, m].median()).values)
    print(f"   without the {int((~keep).sum())} intruders of {METHODS[m]}: "
          f"fusion {average_precision_score(t.y[keep], t.fusion[keep]):.3f}, kNN {average_precision_score(t.y[keep], t.knn[keep]):.3f}")

print("\n4. Same selection rule for kNN as for the ensemble (choose on validation PR-AUC)")
rows = []
for scaler in ("standard", "minmax"):
    for agg in ("mean", "kth"):
        for k in (1, 5, 10, 20, 50, 100):
            sv = knn_score(healthy, df.loc[v.index, FEATURES], k, scaler, agg)
            st = knn_score(healthy, df.loc[t.index, FEATURES], k, scaler, agg)
            rows.append(dict(scaler=scaler, agg=agg, k=k, prauc_val=average_precision_score(v.y, sv),
                             prauc_test=average_precision_score(t.y, st)))
K = pd.DataFrame(rows)
best = K.loc[K.prauc_val.idxmax()]
print(f"   selected on validation: {best['scaler']}, {best['agg']}, k={best['k']} -> val {best['prauc_val']:.3f}, test {best['prauc_test']:.3f}")
print(f"   test PR-AUC over the 24 settings: {K.prauc_test.min():.3f} to {K.prauc_test.max():.3f} "
      f"(standard: {K[K.scaler == 'standard'].prauc_test.min():.3f}-{K[K.scaler == 'standard'].prauc_test.max():.3f}, "
      f"minmax: {K[K.scaler == 'minmax'].prauc_test.min():.3f}-{K[K.scaler == 'minmax'].prauc_test.max():.3f})")

print("\n5. Equal false-alarm budget, calibrated on UNLABELLED validation")
for budget in (2, 3):
    for m in ("knn", "ae", "fusion"):
        thr = budget_threshold(v, m, budget); r = summary(t, m, thr)
        print(f"   {budget} events/week  {METHODS[m]:18s} precision {r['precision']:.3f}  recall {r['recall']:.3f}  "
              f"episodes {r['episodes']}  | unused period {events_per_week(u, m, thr):.2f} events/week")

print("\n6. Partial ROC-AUC on test (normalised, low false-positive rates)")
for m in ("knn", "ae", "fusion"):
    print(f"   {METHODS[m]:18s} FPR<=0.5%: {partial_auc(t.y, t[m], 0.005):.3f}  FPR<=1%: {partial_auc(t.y, t[m], 0.01):.3f}")

print("\n7. Paired day-block bootstrap of the difference kNN - fusion (5,000 resamples)")
groups = [np.flatnonzero(days == d) for d in days.unique()]
rng = np.random.default_rng(1); d_pr, d_roc = [], []
for _ in range(5_000):
    idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
    y = t.y.values[idx]
    if y.sum() == 0:
        continue
    d_pr.append(average_precision_score(y, t.knn.values[idx]) - average_precision_score(y, t.fusion.values[idx]))
    d_roc.append(roc_auc_score(y, t.knn.values[idx]) - roc_auc_score(y, t.fusion.values[idx]))
for name, d in (("PR-AUC", np.array(d_pr)), ("ROC-AUC", np.array(d_roc))):
    print(f"   {name}: mean {d.mean():+.3f}, 95% interval [{np.percentile(d, 2.5):+.3f} ; {np.percentile(d, 97.5):+.3f}], "
          f"kNN ahead in {np.mean(d > 0):.0%} of resamples")

print("\n8. Points never scored by the ensemble (first 5 of each day): effect on single-point methods")
d = test[test[FEATURES].notna().all(axis=1)]
y_all = (d["Fault"] != 0).astype(int).values
knn_all = knn_score(healthy, d[FEATURES])
ae_all = det._rang(det._err_ae(det.sc_ae.transform(d[FEATURES])), det.ref_ae)
common = d.index.isin(t.index)
print(f"   {int((~common).sum())} points excluded, {int(y_all[~common].sum())} of them faulty")
for name, sc in (("kNN (k=10)", knn_all), ("Dense AE x5", ae_all)):
    print(f"   {name:12s} PR-AUC common points {average_precision_score(y_all[common], sc[common]):.3f} | "
          f"all {len(d)} points {average_precision_score(y_all, sc):.3f}")
