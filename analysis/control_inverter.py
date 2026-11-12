"""Winter false alarms: does the fault-free control inverter see the same low-sun deficit?

Read-only diagnostic (writes no file). Requires the raw control-inverter file from Zenodo record 7828879:

    data/dt3_electrical_production_inverter_2.csv      (not versioned, like dt1 and dt2)

Run from the repository root:

    python analysis/control_inverter.py

Inverter 1 carries the staged faults; inverter 2 was never faulted (Graillet et al., Data in Brief 46, 2023).
dt3 is resampled to 5 minutes exactly as dt2 is in notebook section 2, then joined to data/merged_5min_day.csv.
The test period (29 April to 26 May 2022) is never used. Runtime: 2 to 5 minutes.

How to read the result (decision rule fixed before running):
  * ratio Pg1/Pg2 stable at low sun in winter        -> the deficit is common to both arrays (sun geometry, sensor):
                                                         a differential signal Pg1/Pg2 should remove most winter alarms.
  * ratio Pg1/Pg2 drops at low sun in winter only    -> the deficit is local to array 1 (nearby obstacle, unlabelled fault).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DT3 = ROOT / "data" / "dt3_electrical_production_inverter_2.csv"
FEATURES = ["GTI", "Pg", "Va", "Vg", "Ia", "TPV"]
GAP = pd.Timedelta("7min")
LAT, LON = -20.90, 55.48            # approximate (University of La Réunion campus); < 1 degree error on sun elevation
TEST = (pd.Timestamp("2022-04-29", tz="UTC"), pd.Timestamp("2022-05-27", tz="UTC"))
pd.set_option("display.width", 200)

if not DT3.exists():
    sys.exit(f"Missing {DT3.relative_to(ROOT)}: download it from https://zenodo.org/records/7828879 into data/.")


# ----------------------------------------------------------------------------- inverter 2, same processing as dt2
print("Loading dt3 ...")
dt3 = pd.read_csv(DT3)
print(f"   {len(dt3):,} rows, columns {dt3.columns.tolist()}")
needed = {"time", "Pg", "Ia", "Va"}
missing = needed - set(dt3.columns)
if missing:
    sys.exit(f"Unexpected format: columns {sorted(missing)} not found in dt3.")
dt3["time"] = pd.to_datetime(dt3["time"], utc=True)
dt3 = dt3.set_index("time").sort_index()
num = dt3.select_dtypes("number").drop(columns=[c for c in ("Fault",) if c in dt3.columns])
inv2 = num.resample("5min").mean().dropna().add_suffix("_2")
print(f"   after 5-min resampling: {len(inv2):,} rows, {inv2.index.min()} -> {inv2.index.max()}")
if "Fault" in dt3.columns:
    print(f"   note: dt3 has a Fault column; non-zero values: {int((dt3['Fault'].fillna(0) != 0).sum())}")
del dt3, num

df = pd.read_csv(ROOT / "data/merged_5min_day.csv", parse_dates=["time"], index_col="time")
df = df[~((df["Va"] == 0) & (df["Vg"] == 0) & (df["Fg"] == 0))]
d = df.join(inv2, how="inner")
d = d[~((d.index >= TEST[0]) & (d.index < TEST[1]))]          # the test period is never used
d = d[(d["Pg_2"] > 20) & (d["Pg"] >= 0)]
print(f"   joined with inverter 1 (test period removed): {len(d):,} five-minute points")


# ----------------------------------------------------------------------------- sun position (NOAA approximation)
t = d.index.tz_convert("UTC")
g = 2 * np.pi / 365 * (t.dayofyear.values - 1 + (t.hour.values + t.minute.values / 60 - 12) / 24)
eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
ha = np.radians(((t.hour.values * 60 + t.minute.values + eqt + 4 * LON) / 4) - 180)
lat = np.radians(LAT)
d["elev"] = 90 - np.degrees(np.arccos(np.clip(np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha), -1, 1)))
d["morning"] = ha < 0
d["kd"] = (d["DTI"] / d["GTI"]).clip(0, 2)
d["clear"] = d["kd"] < 0.3
month = d.index.month
d["period"] = np.select([d.index < pd.Timestamp("2022-04-29", tz="UTC"), month.isin([6, 7, 8, 9])],
                        ["training/validation (Oct-Apr)", "winter (Jun-Sep)"], "following summer (Oct-Apr)")
ref = d[(d["period"] == "training/validation (Oct-Apr)") & (d["Fault"] == 0) & d["clear"] & (d["elev"] > 45)]
k = (ref["Pg"] / ref["Pg_2"]).median()
d["ratio"] = d["Pg"] / d["Pg_2"] / k                          # 1.0 = typical healthy high-sun value
print(f"   size factor Pg1/Pg2 (healthy, clear sky, sun > 45 deg, Oct-Apr): {k:.3f}; ratios below are normalised by it")


# ----------------------------------------------------------------------------- 1. ratio by season and sun elevation
print("\n1. Normalised ratio Pg1/Pg2, healthy points, clear sky (median, n)")
bins = [(5, 15), (15, 20), (20, 30), (30, 45), (45, 91)]
rows = []
for per in ("training/validation (Oct-Apr)", "winter (Jun-Sep)", "following summer (Oct-Apr)"):
    for side in (True, False):
        r = {"period": per, "side": "morning" if side else "afternoon"}
        for lo, hi in bins:
            s = d[(d["period"] == per) & (d["Fault"] == 0) & d["clear"] & (d["morning"] == side) & d["elev"].between(lo, hi)]
            r[f"{lo}-{hi} deg"] = f"{s['ratio'].median():.3f} ({len(s)})" if len(s) >= 10 else "-"
        rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\n2. Production per unit irradiance of EACH inverter, clear sky, sun 10-30 deg (median Pg/GTI)")
for per in ("training/validation (Oct-Apr)", "winter (Jun-Sep)", "following summer (Oct-Apr)"):
    s = d[(d["period"] == per) & (d["Fault"] == 0) & d["clear"] & d["elev"].between(10, 30)]
    print(f"   {per:30s} inverter 1 {np.median(s['Pg'] / s['GTI']):.3f} | inverter 2 {np.median(s['Pg_2'] / s['GTI']):.3f}   (n={len(s)})")


# ----------------------------------------------------------------------------- 3. at the deployed detector's winter alarms
from detecteur import DetecteurPV  # noqa: E402
det = DetecteurPV(verbeux=False)
w = d[d["period"] == "winter (Jun-Sep)"]
w = w[w[FEATURES].notna().all(axis=1)]
s_ae = pd.Series(det._err_ae(det.sc_ae.transform(w[FEATURES])), index=w.index)
X, idx = [], []
for _, b in w.groupby((w.index.to_series().diff() > GAP).cumsum()):
    v = b[FEATURES].values
    for st in range(len(b) - det.seq_len + 1):
        X.append(v[st:st + det.seq_len]); idx.append(b.index[st + det.seq_len - 1])
X = np.array(X)
s_l = pd.Series(det._err_lstm(det.sc_lstm.transform(X.reshape(-1, len(FEATURES))).reshape(X.shape)), index=pd.DatetimeIndex(idx))
ic = s_ae.index.intersection(s_l.index)
w = w.loc[ic].copy()
w["alarm"] = (det._rang(s_ae[ic].values, det.ref_ae) + det._rang(s_l[ic].values, det.ref_lstm)) / 2 >= det.seuil
print(f"\n3. Winter alarms of the deployed detector on these points: {int(w['alarm'].sum())} of {len(w)}")
for lo, hi in ((10, 20), (20, 30)):
    s = w[w["elev"].between(lo, hi) & w["clear"]]
    a, n = s[s["alarm"]], s[~s["alarm"]]
    if len(a) >= 5:
        print(f"   sun {lo}-{hi} deg, clear sky: ratio at alarms {a['ratio'].median():.3f} (n={len(a)}) | "
              f"at non-alarmed points {n['ratio'].median():.3f} (n={len(n)}) | "
              f"Pg1/GTI {np.median(a['Pg'] / a['GTI']):.3f} vs {np.median(n['Pg'] / n['GTI']):.3f} | "
              f"Pg2/GTI {np.median(a['Pg_2'] / a['GTI']):.3f} vs {np.median(n['Pg_2'] / n['GTI']):.3f}")


# ----------------------------------------------------------------------------- 4. is the ratio sensitive to real faults?
print("\n4. Ratio during the labelled faults of inverter 1 (training and validation only, never the test)")
dev = d[d["period"] == "training/validation (Oct-Apr)"]
for f, s in dev[dev["Fault"] != 0].groupby("Fault"):
    print(f"   F{f:.1f}: median ratio {s['ratio'].median():.3f} (n={len(s)})")
print(f"   healthy reference: median {dev.loc[dev['Fault'] == 0, 'ratio'].median():.3f}, "
      f"1st percentile {dev.loc[dev['Fault'] == 0, 'ratio'].quantile(0.01):.3f}")
