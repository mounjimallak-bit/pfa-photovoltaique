"""Experimental design and sample size: the numbers behind docs/experimental_design.md.

Read-only (writes no file). Run from the repository root:

    python analysis/sample_size.py

Inputs: the deployed detector (model_final/), the 5-minute cache (data/merged_5min_day.csv) and the
leave-one-incident-out score cache (data/loio_scores.csv.gz). Nothing is retrained.
Simulations use fixed seeds. Runtime: about 3 to 6 minutes on a laptop CPU.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detecteur import DetecteurPV  # noqa: E402

FEATURES = ["GTI", "Pg", "Va", "Vg", "Ia", "TPV"]
GAP = pd.Timedelta("7min")
GROUPS = {0: "G1", 1: "G1", 2: "G1", 3: "G2", 4: "G2", 5: "G2", 6: "G3", 7: "G4", 8: "G5", 9: "G6",
          10: "G7", 11: "G8", 12: "G9", 13: "G9", 14: "G10", 15: "G11", 16: "G12"}
pd.set_option("display.width", 200)


# ----------------------------------------------------------------------------- data and deployed scores
def load_split():
    df = pd.read_csv(ROOT / "data/merged_5min_day.csv", parse_dates=["time"], index_col="time")
    df = df[~((df["Va"] == 0) & (df["Vg"] == 0) & (df["Fg"] == 0))]
    end = df.loc[df["Fault"] != 0].index.max() + pd.Timedelta(days=7)
    study, unused = df.loc[:end], df.loc[df.index > end]
    d = study.index.date
    cut_train, cut_val = pd.Timestamp("2022-03-22").date(), pd.Timestamp("2022-04-28").date()
    train, val, test = study[d <= cut_train], study[(d > cut_train) & (d <= cut_val)], study[d > cut_val]
    assert (len(train), len(val), len(test)) == (10_863, 4_219, 2_975)
    return val, test, unused


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
    out = pd.DataFrame({"score": (det._rang(s_ae.loc[ic].values, det.ref_ae)
                                  + det._rang(s_l.loc[ic].values, det.ref_lstm)) / 2}, index=ic)
    out["y"] = (d.loc[ic, "Fault"] != 0).astype(int)
    out["fault"] = d.loc[ic, "Fault"]
    return out.sort_index()


def blocks(index, mask):
    mask = np.asarray(mask, bool)
    night = np.r_[True, np.diff(index.values) > GAP.to_timedelta64()]
    start = mask & (night | ~np.r_[False, mask[:-1]])
    num = np.cumsum(start)
    return [np.flatnonzero((num == k) & mask) for k in range(1, num.max() + 1)] if mask.any() else []


val, test, unused = load_split()
det = DetecteurPV(verbeux=False)
S = {name: deployed_scores(det, part) for name, part in [("val", val), ("test", test), ("unused", unused)]}

# ----------------------------------------------------------------------------- 1. episodes
print("1. Results per episode (deployed system, threshold %.4f)" % det.seuil)
for name in ("val", "test"):
    s = S[name]
    for e in blocks(s.index, s["y"].values == 1):
        a = s["score"].values[e] >= det.seuil
        day = s[s.index.normalize() == s.index[e[0]].normalize()]
        print(f"   {name:4s} {str(s.index[e[0]])[:16]}  F{s['fault'].values[e[0]]:.1f}  {len(e):3d} points  "
              f"detected={bool(a.any())}  recall={a.mean():.3f}  max score={s['score'].values[e].max():.3f}  "
              f"within-day ROC-AUC={roc_auc_score(day['y'], day['score']):.3f}")

# ----------------------------------------------------------------------------- 2. leave-one-incident-out cache
L = pd.read_csv(ROOT / "data/loio_scores.csv.gz", parse_dates=["time", "inc_start", "inc_end"])
METHODS = list(L["method"].unique())
rows, alarms = [], []
for fold, f in L.groupby("fold"):
    te, ca = f[f["part"] == "test"], f[f["part"] == "cal"]
    t_common = set.intersection(*[set(te.loc[te["method"] == m, "time"]) for m in METHODS])
    c_common = set.intersection(*[set(ca.loc[ca["method"] == m, "time"]) for m in METHODS])
    for m in METHODS:
        t = te[(te["method"] == m) & te["time"].isin(t_common)].sort_values("time")
        c = ca[(ca["method"] == m) & ca["time"].isin(c_common)]
        y, sc = t["label"].to_numpy(), t["score"].to_numpy()
        a = sc >= np.quantile(c["score"], 0.99)
        inside = ((t["time"] >= t["inc_start"].iloc[0]) & (t["time"] <= t["inc_end"].iloc[0])).to_numpy() & (y == 1)
        rows.append(dict(fold=fold, group=GROUPS[fold], fault=t["fault"].iloc[0], method=m,
                         roc=roc_auc_score(y, sc), recall=a[y == 1].mean(), detected=bool(a[inside].any())))
        if m == "Fusion (deployed design)":
            alarms.append(pd.DataFrame({"incident": fold, "a": a[y == 1].astype(float)}))
R = pd.DataFrame(rows)

print("\n2. Leave-one-incident-out (17 incidents, 12 campaigns), threshold = 99th percentile of healthy calibration")
for m in ("Fusion (deployed design)", "kNN", "Dense AE x5"):
    r = R[R["method"] == m]
    print(f"   {m:26s} detected {int(r['detected'].sum())}/17 | ROC-AUC mean {r['roc'].mean():.3f} (sd {r['roc'].std():.3f}) "
          f"| recall mean {r['recall'].mean():.3f} (sd {r['recall'].std():.3f})")
missed = R[(R["method"] == "Fusion (deployed design)") & ~R["detected"]]
print(f"   incidents missed by the fusion: {missed[['fold', 'fault']].to_dict('records')}")

# intraclass correlation of alarms among the faulty points of each incident (one-way ANOVA estimator)
P = pd.concat(alarms)
k = P.groupby("incident").size()
m0 = (k.sum() - (k ** 2).sum() / k.sum()) / (len(k) - 1)
msb = (k * (P.groupby("incident")["a"].mean() - P["a"].mean()) ** 2).sum() / (len(k) - 1)
msw = ((P["a"] - P.groupby("incident")["a"].transform("mean")) ** 2).sum() / (len(P) - len(k))
icc = (msb - msw) / (msb + (m0 - 1) * msw)
deff = 1 + (k.mean() - 1) * icc
print(f"   ICC of alarms within incidents = {icc:.2f}; mean {k.mean():.0f} faulty points per incident; "
      f"design effect {deff:.0f}; {len(P)} faulty points ~ {len(P) / deff:.0f} independent observations")
r_sain = [np.corrcoef(b["score"].values[:-1], b["score"].values[1:])[0, 1]
          for _, b in S["unused"].groupby((S["unused"].index.to_series().diff() > GAP).cumsum()) if len(b) > 20]
print(f"   lag-1 autocorrelation of healthy scores within a day (median) = {np.nanmedian(r_sain):.2f}")

G = R.groupby(["method", "group"])[["roc", "recall"]].mean().reset_index()
d_roc = (G[G["method"] == "Fusion (deployed design)"].set_index("group")["roc"]
         - G[G["method"] == "kNN"].set_index("group")["roc"]).to_numpy()
d_rec = (G[G["method"] == "Fusion (deployed design)"].set_index("group")["recall"]
         - G[G["method"] == "kNN"].set_index("group")["recall"]).to_numpy()
print(f"   per-campaign difference fusion - kNN: ROC-AUC mean {d_roc.mean():+.4f} (sd {d_roc.std(ddof=1):.4f}), "
      f"recall mean {d_rec.mean():+.3f} (sd {d_rec.std(ddof=1):.3f})")

# false alarms per day on the unused (fault-free) period
h = S["unused"]
days = sorted(set(h.index.normalize()))
ev = pd.Series([len(blocks(b.index, (b["score"] >= det.seuil).values))
                for b in (h[h.index.normalize() == d] for d in days)], index=pd.DatetimeIndex(days))
winter = ev.index.month.isin([6, 7, 8, 9])
print(f"   false-alarm events: {7 * ev.mean():.2f}/week over {len(ev)} days; variance/mean = {ev.var() / ev.mean():.1f}; "
      f"lag-1 autocorrelation {ev.autocorr(1):.2f}; June-September {7 * ev[winter].mean():.2f}/week, "
      f"October-April {7 * ev[~winter].mean():.2f}/week")

# ----------------------------------------------------------------------------- 3. sample size, independent episodes
def exact_power(n, p0, p1, alpha=0.05):
    k = int(stats.binom.isf(alpha, n, p0)) + 1
    while k > 0 and stats.binom.sf(k - 2, n, p0) <= alpha:
        k -= 1
    return stats.binom.sf(k - 1, n, p1)


print("\n3. Episode detection rate, independent episodes (exact one-sided test, alpha = 0.05)")
for p0, p1 in [(0.5, 0.8), (0.7, 0.9), (0.8, 0.95), (0.9, 0.99)]:
    n80 = next(n for n in range(3, 2000) if all(exact_power(m, p0, p1) >= 0.8 for m in range(n, n + 30)))
    n90 = next(n for n in range(3, 2000) if all(exact_power(m, p0, p1) >= 0.9 for m in range(n, n + 30)))
    print(f"   H0 p <= {p0:.2f} vs p = {p1:.2f}: n = {n80} (80% power), {n90} (90% power); power with n = 3: "
          f"{exact_power(3, p0, p1):.2f}")
for lb in (0.8, 0.9, 0.95):
    print(f"   all detected (k/k): lower 95% bound >= {lb} from n = {int(np.ceil(np.log(0.025) / np.log(lb)))}")
print(f"   3/3 today: 95% interval [{0.025 ** (1 / 3):.2f} ; 1.00]")

# ----------------------------------------------------------------------------- 4. episodes clustered in campaigns
rng = np.random.default_rng(2026)


def detected_total(n_camp, m, p, rho, n_sim):
    if rho == 0:
        pc = np.full((n_sim, n_camp), p)
    else:
        s = (1 - rho) / rho
        pc = rng.beta(p * s, (1 - p) * s, size=(n_sim, n_camp))
    return rng.binomial(m, pc).sum(axis=1)


def reject(k, n, p0, deff=1.0, alpha=0.05):
    ke, ne = k / deff, n / deff        # exact test on the effective sample size (design-effect approximation)
    return ke > 0 and stats.beta.ppf(alpha, ke, ne - ke + 1) > p0


print("\n4. Episodes clustered in campaigns of 3 (H0 p <= 0.80 vs p = 0.95; 4,000 simulations per cell)")
for rho in (0.0, 0.1, 0.3, 0.5):
    deff_c = 1 + 2 * rho
    cells = []
    for n_camp in (10, 20, 30):
        k0, k1 = detected_total(n_camp, 3, 0.80, rho, 4000), detected_total(n_camp, 3, 0.95, rho, 4000)
        n = 3 * n_camp
        cells.append(f"{n_camp} campaigns: naive type I {np.mean([reject(k, n, 0.8) for k in k0]):.2f}, "
                     f"adjusted type I {np.mean([reject(k, n, 0.8, deff_c) for k in k0]):.2f}, "
                     f"adjusted power {np.mean([reject(k, n, 0.8, deff_c) for k in k1]):.2f}")
    print(f"   ICC {rho:.1f}: " + " | ".join(cells))

# ----------------------------------------------------------------------------- 5. paired comparison of two methods
def symmetric_sample(diffs, n, scale):
    dev = np.abs(diffs - np.median(diffs)) * scale
    return rng.choice(dev, n, replace=True) * rng.choice([-1, 1], n)   # symmetric around 0: the null holds


def wilcoxon_power(diffs, delta, n, scale=1.0, n_sim=2000):
    hits = 0
    for _ in range(n_sim):
        x = symmetric_sample(diffs, n, scale) + delta
        hits += bool(np.any(x != 0) and stats.wilcoxon(x).pvalue < 0.05)
    return hits / n_sim


def tost_power(diffs, margin, n, scale=1.0, n_sim=2000):
    hits = 0
    for _ in range(n_sim):
        x = symmetric_sample(diffs, n, scale)
        half = stats.t.ppf(0.95, n - 1) * x.std(ddof=1) / np.sqrt(n)
        hits += (x.mean() - half > -margin) and (x.mean() + half < margin)
    return hits / n_sim


print("\n5. Paired comparison by campaign (resampled observed differences, symmetrised; Wilcoxon, alpha = 0.05;"
      " 2,000 simulations per cell)")
NS = (12, 30, 50, 100)
for label, diffs, delta in (("recall", d_rec, 0.0), ("recall", d_rec, 0.10), ("ROC-AUC", d_roc, 0.0), ("ROC-AUC", d_roc, 0.01)):
    for scale in (1.0, 1.5):
        res = " | ".join(f"{n}: {wilcoxon_power(diffs, delta, n, scale):.2f}" for n in NS)
        tag = "type I error" if delta == 0 else "power"
        print(f"   {label:7s} difference {delta:+.2f}, spread x{scale}: {tag} by number of campaigns  {res}")
for scale in (1.0, 1.5):
    res = " | ".join(f"{n}: {tost_power(d_roc, 0.01, n, scale):.2f}" for n in NS)
    print(f"   equivalence (TOST) of ROC-AUC within +/-0.01, spread x{scale}, true difference 0: {res}")

# ----------------------------------------------------------------------------- 6. false-alarm rate precision
print("\n6. False-alarm rate: relative half-width of the 95% interval (resampled 7-day blocks, stationarity within a season)")
for label, mask in (("June-September", winter), ("October-April", ~winter)):
    x = ev[mask].to_numpy()
    nb = len(x) // 7
    weeks = x[: nb * 7].reshape(nb, 7)
    res = []
    for w in (4, 13, 26, 52):
        r = np.array([weeks[rng.integers(0, nb, w)].mean() * 7 for _ in range(5000)])
        res.append(f"{w} weeks: +/-{(np.percentile(r, 97.5) - np.percentile(r, 2.5)) / 2 / r.mean():.0%}")
    print(f"   {label:15s} ({7 * x.mean():.2f}/week): " + " | ".join(res))
