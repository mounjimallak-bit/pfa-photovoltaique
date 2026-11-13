"""Numerical checks of the mathematical notes of the case study
(portfolio page post/pfa/index.html, boxes marked "Mathematical note").

Read-only (writes no file). Run from the repository root:

    python analysis/math_checks.py

Each section prints the value given by a closed-form derivation next to the value measured on the
project's data, with the deployed detector (model_final/) and the split of notebook section 16.1.
The test set is only read to check identities that do not depend on the labels' content
(ROC-AUC = Mann-Whitney, bootstrap bookkeeping); nothing is selected on it.
Runtime: about 2 to 4 minutes on a laptop CPU.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detecteur import DetecteurPV  # noqa: E402

FEATURES = ["GTI", "Pg", "Va", "Vg", "Ia", "TPV"]
GAP = pd.Timedelta("7min")
RNG = np.random.default_rng(0)


def load_split():
    """Same split as notebook section 16.1 and analysis/knn_vs_ensemble.py."""
    df = pd.read_csv(ROOT / "data/merged_5min_day.csv", parse_dates=["time"], index_col="time")
    df = df[~((df["Va"] == 0) & (df["Vg"] == 0) & (df["Fg"] == 0))]
    end = df.loc[df["Fault"] != 0].index.max() + pd.Timedelta(days=7)
    study, unused = df.loc[:end], df.loc[df.index > end]
    d = study.index.date
    cut_train, cut_val = pd.Timestamp("2022-03-22").date(), pd.Timestamp("2022-04-28").date()
    train, val, test = study[d <= cut_train], study[(d > cut_train) & (d <= cut_val)], study[d > cut_val]
    assert (len(train), len(val), len(test)) == (10_863, 4_219, 2_975)
    return train, val, test, unused


def deployed_scores(det, d):
    """Ranks of each family and fusion score, exactly as DetecteurPV.predire_lot."""
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
    out = pd.DataFrame({"r_ae": det._rang(s_ae.loc[ic].values, det.ref_ae),
                        "r_lstm": det._rang(s_l.loc[ic].values, det.ref_lstm)}, index=ic)
    out["S"] = (out["r_ae"] + out["r_lstm"]) / 2
    out["y"] = (d.loc[ic, "Fault"] != 0).astype(int)
    return out.sort_index()


train, val, test, unused = load_split()
det = DetecteurPV(verbeux=False)
T = det.seuil
V, TE, U = (deployed_scores(det, p) for p in (val, test, unused))
UH = U[U["y"] == 0]

print("1. Clopper-Pearson interval for k = n")
for n in (3, 17):
    closed = 0.025 ** (1 / n)
    exact = stats.beta.ppf(0.025, n, 1)
    print(f"   n = {n:2d}: lower bound (alpha/2)^(1/n) = {closed:.4f} | beta quantile = {exact:.4f}")
print(f"   smallest n with lower bound >= 0.80: ln(0.025)/ln(0.8) = {np.log(0.025) / np.log(0.8):.2f} -> n = 17")
print(f"   one-sided Bayesian bound, uniform prior, 3/3 (posterior Beta(4,1)): 0.05^(1/4) = {0.05 ** 0.25:.3f} "
      f"| beta quantile = {stats.beta.ppf(0.05, 4, 1):.3f}")

print("\n2. Average precision of a random score equals the prevalence")
y = V["y"].to_numpy()
ap_random = np.array([average_precision_score(y, RNG.permutation(V["S"].to_numpy())) for _ in range(1000)])
print(f"   validation: prevalence pi = {y.mean():.4f} | mean AP of 1,000 random orderings = {ap_random.mean():.4f} "
      f"(sd {ap_random.std():.4f})")
N, n1v = len(y), int(y.sum())
H = (1 / np.arange(1, N + 1)).sum()
pi_ = (n1v - 1) / (N - 1)
print(f"   exact expectation for N = {N}, n+ = {n1v}: pi' + (1 - pi') H_N / N with pi' = (n+ - 1)/(N - 1) "
      f"= {pi_ + (1 - pi_) * H / N:.4f}")
print(f"   deployed AP = {average_precision_score(y, V['S']):.4f}; lift = AP / pi = "
      f"{average_precision_score(y, V['S']) / y.mean():.1f}")

print("\n3. ROC-AUC equals the normalised Mann-Whitney statistic")
yt, st = TE["y"].to_numpy(), TE["S"].to_numpy()
ranks = stats.rankdata(st)                      # mid-ranks handle ties: the 1/2 term
n1, n0 = yt.sum(), (1 - yt).sum()
u = ranks[yt == 1].sum() - n1 * (n1 + 1) / 2
print(f"   test: U / (n+ n-) = {u / (n1 * n0):.10f} | roc_auc_score = {roc_auc_score(yt, st):.10f}")

print("\n4. What the fusion threshold means")
print(f"   threshold t = {T:.4f}")
print(f"   bound if the two ranks were independent: 2(1-t)^2 = {2 * (1 - T) ** 2:.4f}")
print(f"   bound if the two ranks were identical:   1-t      = {1 - T:.4f}")
for name, s in (("validation (rank reference, faults included)", V), ("unused period, healthy points", UH)):
    rho = stats.spearmanr(s["r_ae"], s["r_lstm"]).statistic
    shuffled = (s["r_ae"].to_numpy() + RNG.permutation(s["r_lstm"].to_numpy())) / 2
    print(f"   {name}: Spearman(r_AE, r_LSTM) = {rho:.2f} | alarm rate P(S >= t) = {(s['S'] >= T).mean():.4f} "
          f"| with r_LSTM shuffled (independence) = {(shuffled >= T).mean():.4f}")

print("\n5. Mahalanobis distance against the chi-square law")
healthy_tr = train.loc[train["Fault"] == 0, FEATURES].dropna()
healthy_va = val.loc[val["Fault"] == 0, FEATURES].dropna()
mu, cov = healthy_tr.mean().to_numpy(), np.cov(healthy_tr.to_numpy().T)
z = healthy_va.to_numpy() - mu
d2 = np.einsum("ij,ij->i", z, np.linalg.solve(cov, z.T).T)
q99 = stats.chi2.ppf(0.99, len(FEATURES))
print(f"   condition number of the healthy covariance: {np.linalg.cond(cov):.3g} (depends on the units) | "
      f"of the correlation matrix: {np.linalg.cond(np.corrcoef(healthy_tr.to_numpy().T)):.3g} (collinearity only)")
print(f"   chi2(6) 99th percentile = {q99:.2f} | share of healthy validation points above it = {(d2 > q99).mean():.3f} "
      f"(0.010 if the data were Gaussian)")
print(f"   median d2 = {np.median(d2):.2f} (chi2(6) median {stats.chi2.median(len(FEATURES)):.2f})")

print("\n6. Min-max scaling and sigmoid output: the reconstruction floor")
Xu = pd.DataFrame(det.sc_ae.transform(unused.loc[UH.index, FEATURES]), index=UH.index, columns=FEATURES)
out_of_range = ((Xu < 0) | (Xu > 1)).any(axis=1)
alarm = UH["S"] >= T
print(f"   unused period, healthy points: {len(UH):,}; with at least one scaled feature outside [0, 1]: "
      f"{out_of_range.mean():.3f}")
print(f"   among alarmed points: {out_of_range[alarm].mean():.3f} ({int(alarm.sum())} points) | "
      f"among the others: {out_of_range[~alarm].mean():.3f}")
day = pd.DataFrame({"alarm": alarm, "oor": out_of_range}).groupby(UH.index.normalize()).mean()
print(f"   per day ({len(day)} days): Spearman(share out of range, share alarmed) = "
      f"{stats.spearmanr(day['oor'], day['alarm']).statistic:.2f}  (descriptive; days are autocorrelated)")

print("\n7. Temporal dependence and effective sample size")
r1 = [np.corrcoef(b["S"].values[:-1], b["S"].values[1:])[0, 1]
      for _, b in UH.groupby((UH.index.to_series().diff() > GAP).cumsum()) if len(b) > 20]
phi = float(np.nanmedian(r1))
print(f"   median within-day lag-1 autocorrelation of the healthy fusion score: phi = {phi:.2f}")
print(f"   AR(1) effective sample size factor (1 - phi) / (1 + phi) = {(1 - phi) / (1 + phi):.3f} "
      f"-> about {(1 + phi) / (1 - phi):.0f} consecutive points carry one independent observation")
print(f"   design effect with ICC 0.36 and 86 points per incident (analysis/sample_size.py): "
      f"1 + 85 x 0.36 = {1 + 85 * 0.36:.1f}")

print("\n8. Day-level bootstrap bookkeeping")
days = TE.index.normalize()
n_days, n_fault_days = days.nunique(), days[TE["y"] == 1].nunique()
p_none = (1 - n_fault_days / n_days) ** n_days
print(f"   test: {n_days} days, {n_fault_days} with faults; P(a resample has no fault day) = "
      f"(1 - {n_fault_days}/{n_days})^{n_days} = {p_none:.4f}")
print(f"   expected resamples dropped out of 10,000: {10_000 * p_none:.0f} "
      f"(sd {np.sqrt(10_000 * p_none * (1 - p_none)):.0f}); notebook section 16.3 kept 9,613, i.e. dropped 387")

print("\n9. Gaussian copula with the measured Spearman correlation (simulation, not project data)")
rho_s = stats.spearmanr(V["r_ae"], V["r_lstm"]).statistic
rho = 2 * np.sin(np.pi * rho_s / 6)              # Spearman -> Pearson for a Gaussian copula
z1 = RNG.standard_normal(2_000_000)
z2 = rho * z1 + np.sqrt(1 - rho ** 2) * RNG.standard_normal(2_000_000)
sim = (stats.norm.cdf(z1) + stats.norm.cdf(z2)) / 2
print(f"   Spearman {rho_s:.2f} -> Gaussian-copula alarm rate P(S >= t) = {(sim >= T).mean():.4f} "
      f"(measured on validation: {(V['S'] >= T).mean():.4f})")

# ============================================================================ linear algebra
print("\n10. Eigen-structure of the healthy covariance (six sensors, standardised as for kNN and Mahalanobis)")
Zs = (healthy_tr - healthy_tr.mean()) / healthy_tr.std()
R = np.corrcoef(Zs.to_numpy().T)
lam, Qv = np.linalg.eigh(R)
lam, Qv = lam[::-1], Qv[:, ::-1]
print("   eigenvalues of the correlation matrix:", " ".join(f"{x:.4f}" for x in lam))
print("   share of variance:", " ".join(f"{x:.1%}" for x in lam / lam.sum()),
      f"| first two: {lam[:2].sum() / lam.sum():.1%}")
print(f"   condition number lambda_max / lambda_min = {lam[0] / lam[-1]:.3g}")
for j in (4, 5):
    print(f"   eigenvector {j + 1} (variance {lam[j]:.4f}):",
          " ".join(f"{f}={v:+.2f}" for f, v in zip(FEATURES, Qv[:, j])))
# where do faults deviate? share of the Mahalanobis distance carried by each principal direction
Zv = ((val.loc[V.index, FEATURES] - healthy_tr.mean()) / healthy_tr.std()).to_numpy()
proj2 = (Zv @ Qv) ** 2 / lam                     # contribution of each direction to d_M^2
share_minor = proj2[:, 4:].sum(axis=1) / proj2.sum(axis=1)
yv = V["y"].to_numpy()
print(f"   share of d_M^2 carried by the two smallest directions: healthy validation median "
      f"{np.median(share_minor[yv == 0]):.2f} | faulty validation median {np.median(share_minor[yv == 1]):.2f}")
print(f"   PR-AUC on validation (same {len(yv)} points): d_M^2 {average_precision_score(yv, proj2.sum(axis=1)):.3f} | "
      f"T^2 on 4 leading directions {average_precision_score(yv, proj2[:, :4].sum(axis=1)):.3f} | "
      f"minor part (2 smallest) {average_precision_score(yv, proj2[:, 4:].sum(axis=1)):.3f}")

# ============================================================================ linear autoencoder and PCA
print("\n11. Linear autoencoder against PCA (input scaled exactly as the dense autoencoder's: model_final scaler)")
import tensorflow as tf
Xh = det.sc_ae.transform(healthy_tr)                # 9,618 x 6, the deployed AE's input space
mu_h = Xh.mean(axis=0)
U_, D_, Vt_ = np.linalg.svd(Xh - mu_h, full_matrices=False)
ev = D_ ** 2 / (len(Xh) - 1)
V4 = Vt_[:4].T                                      # 6 x 4 leading principal directions
print("   PCA eigenvalues (scaled space):", " ".join(f"{x:.2e}" for x in ev),
      f"| variance kept by 4 components {ev[:4].sum() / ev.sum():.4%}")
def pca_err(X):
    C = X - mu_h
    return ((C - C @ V4 @ V4.T) ** 2).mean(axis=1)
tf.keras.utils.set_random_seed(0)
inp = tf.keras.Input(shape=(6,))
code = tf.keras.layers.Dense(4, activation=None, name="enc")(inp)
out = tf.keras.layers.Dense(6, activation=None, name="dec")(code)
lin = tf.keras.Model(inp, out)
lin.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
lin.fit(Xh, Xh, epochs=400, batch_size=64, validation_split=0.15, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)])
We, be = lin.get_layer("enc").get_weights()
Wd, bd = lin.get_layer("dec").get_weights()
Qd, _ = np.linalg.qr(Wd.T)                          # orthonormal basis of the decoder's 4-dimensional image
cosines = np.linalg.svd(V4.T @ Qd, compute_uv=False)
angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))
print("   principal angles between the linear AE's subspace and the PCA subspace (degrees):",
      " ".join(f"{a:.2f}" for a in angles))
P_ae = We @ Wd
print(f"   || W_enc W_dec - V4 V4^T ||_F = {np.linalg.norm(P_ae - V4 @ V4.T):.3f} "
      f"(the product is close to a projector; W_enc alone is not orthonormal: ||W_enc^T W_enc - I||_F = "
      f"{np.linalg.norm(We.T @ We - np.eye(4)):.2f})")
mse_lin = ((Xh - lin.predict(Xh, verbose=0)) ** 2).mean()
print(f"   training reconstruction MSE: linear AE {mse_lin:.3e} | PCA {pca_err(Xh).mean():.3e}")
Xv_s = det.sc_ae.transform(val.loc[V.index, FEATURES])
e_pca = pca_err(Xv_s)
e_lin = ((Xv_s - lin.predict(Xv_s, verbose=0)) ** 2).mean(axis=1)
e_dense = det._err_ae(Xv_s)
print(f"   validation PR-AUC on the same {len(yv)} points: PCA residual (m = 4) {average_precision_score(yv, e_pca):.3f} | "
      f"linear AE {average_precision_score(yv, e_lin):.3f} | deployed dense AE x5 {average_precision_score(yv, e_dense):.3f}")
print(f"   Spearman correlation of scores: PCA vs linear AE {stats.spearmanr(e_pca, e_lin).statistic:.3f} | "
      f"PCA vs dense AE {stats.spearmanr(e_pca, e_dense).statistic:.3f}")

# ============================================================================ generalisation and distribution shift
print("\n12. Reconstruction error of the deployed dense AE on healthy points: training, validation, seasons")
Xu_all = det.sc_ae.transform(unused.loc[UH.index, FEATURES])
e_u = pd.Series(det._err_ae(Xu_all), index=UH.index)
winter = e_u.index.month.isin([6, 7, 8, 9])
e_tr = det._err_ae(Xh)
e_vh = e_dense[yv == 0]
print(f"   median error: training (healthy) {np.median(e_tr):.2e} | validation (healthy) {np.median(e_vh):.2e} | "
      f"unused outside Jun-Sep {e_u[~winter].median():.2e} | unused Jun-Sep {e_u[winter].median():.2e}")
q99 = np.quantile(e_tr, 0.99)
print(f"   share above the training 99th percentile: validation healthy {(e_vh > q99).mean():.3f} | "
      f"outside Jun-Sep {(e_u[~winter] > q99).mean():.3f} | Jun-Sep {(e_u[winter] > q99).mean():.3f}")

# ============================================================================ probability: dependence and calibration
print("\n13. Tail dependence between the two families and calibration of the fused score (validation)")
ra, rl = V["r_ae"].to_numpy(), V["r_lstm"].to_numpy()
sim_a, sim_b = stats.norm.cdf(z1), stats.norm.cdf(z2)  # Gaussian copula draws of section 9
for q in (0.90, 0.95):
    emp = ((rl > q) & (ra > q)).mean() / (ra > q).mean()
    gau = ((sim_b > q) & (sim_a > q)).mean() / (sim_a > q).mean()
    print(f"   P(r_LSTM > {q} | r_AE > {q}): measured {emp:.2f} | Gaussian copula with the same Spearman {gau:.2f} | "
          f"independence {1 - q:.2f}")
bins = [0, 0.5, 0.8, 0.9, T, 1.0001]
cut = pd.cut(V["S"], bins, right=False)
cal = V.groupby(cut, observed=True)["y"].agg(["size", "mean"])
for iv, r in cal.iterrows():
    print(f"   S in [{iv.left:.3f}, {min(iv.right, 1):.3f}): {int(r['size']):5d} points, share of faults {r['mean']:.3f}")

# ============================================================================ statistics of a rank
print("\n14. Sampling error of a rank against the frozen reference")
F0 = 0.946
for name, n_ref in (("AE", len(det.ref_ae)), ("LSTM", len(det.ref_lstm))):
    print(f"   {name}: n = {n_ref}; iid standard error of the empirical CDF at {F0}: "
          f"sqrt(F(1-F)/n) = {np.sqrt(F0 * (1 - F0) / n_ref):.4f}; DKW 95% band {np.sqrt(np.log(40) / (2 * n_ref)):.4f}")
r1v = [np.corrcoef(b["r_ae"].values[:-1], b["r_ae"].values[1:])[0, 1]
       for _, b in V.groupby((V.index.to_series().diff() > GAP).cumsum()) if len(b) > 20]
phi_v = float(np.nanmedian(r1v))
neff = len(det.ref_ae) * (1 - phi_v) / (1 + phi_v)
print(f"   lag-1 autocorrelation of the AE rank within validation days: {phi_v:.2f} -> n_eff ~ {neff:.0f}; "
      f"standard error with n_eff: {np.sqrt(F0 * (1 - F0) / neff):.4f}")

# ============================================================================ time series
print("\n15. Autocorrelation function and stationarity of the healthy fusion score (unused period)")
segs = [b["S"].to_numpy() for _, b in UH.groupby((UH.index.to_series().diff() > GAP).cumsum()) if len(b) > 30]
def acf_pooled(segs, h):
    x = np.concatenate([s[:-h] - s.mean() for s in segs]); y = np.concatenate([s[h:] - s.mean() for s in segs])
    v = np.concatenate([s - s.mean() for s in segs]).var()
    return (x * y).mean() / v
lags = list(range(1, 13))
acf = [acf_pooled(segs, h) for h in lags]
print("   lag (5 min):      " + " ".join(f"{h:5d}" for h in lags))
print("   pooled ACF:       " + " ".join(f"{a:5.2f}" for a in acf))
print("   AR(1) phi^h:      " + " ".join(f"{acf[0] ** h:5.2f}" for h in lags))
tau = 1 + 2 * sum(a for a in acf if a > 0)
print(f"   integrated autocorrelation time over 12 lags: tau = 1 + 2 sum rho(h) = {tau:.1f} "
      f"(AR(1) with phi = {acf[0]:.2f}: (1 + phi)/(1 - phi) = {(1 + acf[0]) / (1 - acf[0]):.1f})")
hour = UH.groupby(UH.index.hour)["S"].mean()
month = UH.groupby(UH.index.month)["S"].mean()
print("   mean healthy score by UTC hour: " + " ".join(f"{h:02d}h {v:.2f}" for h, v in hour.items()))
print("   mean healthy score by month:    " + " ".join(f"{m:02d} {v:.2f}" for m, v in month.items()))
