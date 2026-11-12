# Experimental Design and Sample Size Considerations

Every number on this page is recomputed by `python analysis/sample_size.py` , from the deployed detector in `model_final/`, the 5-minute data cache and the leave-one-incident-out score cache `data/loio_scores.csv.gz`. Nothing is retrained.

Data: [Zenodo record 7828879](https://zenodo.org/records/7828879) (version 3), described in Graillet et al., *Data in Brief* 46 (2023) 108723, [doi:10.1016/j.dib.2022.108723](https://doi.org/10.1016/j.dib.2022.108723). One 4 kW rooftop installation in La Réunion, with staged shading faults.

## 1. What the current data can and cannot show

The chronological test set contains **three episodes of a single fault sub-type (F3.1)**. They are three repetitions of one staged experiment, all starting at 05:45 UTC and lasting about two hours. All three are detected (recall 0.92, 1.00 and 1.00). This is a **proof of concept, not a measured detection rate**:

- 3/3 detected episodes give an exact 95% confidence interval of **[0.29, 1.00]** for the detection rate (Clopper–Pearson).
- With n = 3, an exact one-sided test of H0: p ≤ 0.80 against p = 0.95 has a power of **0.00** at α = 0.05: even 3/3 cannot reject H0. The same holds for every scenario in section 4.
- The only fault type absent from development (F3.1) is a near-twin of a type present in validation (F3.2). The evidence for generalisation to unseen fault types therefore rests on **one type**, and the evidence for generalisation to other installations on **one site**.
- The episode of 19 May 2022 is detected in the chronological test but **missed** in the leave-one-incident-out evaluation by the same architecture, whose reference and calibration data differ. The outcome of an individual episode depends on the protocol: a single pass over three episodes cannot estimate a detection rate.
- Validation shows how much episodes differ: with the same model, recall is 0.52 and 0.69 on intermittent shading (F4.0), and 0.96 and 1.00 on F3.2.

## 2. Unit of analysis

Five-minute points are **not** independent observations:

- Among the faulty points of the 17 incidents, alarms have an intraclass correlation of **0.36**. With 86 faulty points per incident on average, the design effect is about **31**, so the 1,458 faulty points carry the information of about **46 independent observations**, at best.
- Healthy scores have a median lag-1 autocorrelation of **0.84** within a day; daily false-alarm counts have a lag-1 autocorrelation of 0.52.
- Sliding LSTM windows overlap by five of their six points.

The unit of analysis depends on the claim:

| Claim | Unit of analysis | Currently available |
|---|---|---|
| Detects faults of the types studied | fault **episode**, clustered by **test campaign** | 17 episodes in 12 campaigns |
| Generalises to unseen fault types | fault **type** | 7 sub-types (4 families); 1 held out chronologically |
| Generalises to other plants | **installation** | 1 |
| False-alarm rate | **day** (overdispersed, autocorrelated, seasonal) | 279 fault-free days, 1 site, 1 year |

Point-level tests (for example a binomial test on the 73 faulty test points), DeLong's test on point scores, McNemar's test on points and normal approximations to the binomial near p = 1 are **not** appropriate here and are not used.

## 3. Endpoints, minimal relevant differences and error rates

| Endpoint | Definition | Minimal relevant difference | Justification |
|---|---|---|---|
| **E1** — episode detection rate (primary) | share of fault episodes with at least one alarm, at a false-alarm budget fixed in advance and calibrated on healthy data | H0: p ≤ 0.80 against p = 0.95 (15 percentage points) | Missing one fault in five does not justify monitoring; the staged faults cost about 18% of output while they last |
| **E2** — false-alarm rate | alarm events per week on fault-free data, reported by season | factor 1.5, or ±25% precision on the rate | An event costs an operator check; the observed rate differs by a factor of 12 between seasons |
| **E3** — method comparison | per-campaign difference in recall at equal false-alarm budget (primary) or in ROC-AUC (secondary) | 0.10 in recall; ±0.01 in ROC-AUC as an equivalence margin | ROC-AUC sits near its ceiling (median 0.99), where a difference of 0.01 doubles the share of misranked pairs; recall at a fixed budget is what an operator experiences |
| E4 — detection delay | steps from episode onset to first alarm | 12 steps (1 hour) | Shading faults last hours; minutes do not matter operationally |

Error rates: α = 0.05 (one-sided for "at least p0" and non-inferiority claims, two-sided otherwise), target power 0.80 (0.90 for a confirmatory study). The minimal relevant differences are expert choices and must be fixed **before** any new experiment.

## 4. Sample size

### 4.1 Detection rate, independent episodes

Exact one-sided binomial test, α = 0.05. Because the power of a discrete test is saw-toothed, the table gives the smallest n from which power stays above the target.

| H0 → alternative | n for 80% power | n for 90% power | Power with n = 3 |
|---|---|---|---|
| 0.50 → 0.80 | 18 | 26 | 0.00 |
| 0.70 → 0.90 | 33 | 37 | 0.00 |
| **0.80 → 0.95** | **37** | **44** | 0.00 |
| 0.90 → 0.99 | 46 | 61 | 0.00 |

If every episode is detected (k/k), the lower bound of the 95% interval reaches 0.80 only from **17/17**, 0.90 from **36/36** and 0.95 from **72/72**.

### 4.2 Detection rate, episodes clustered in campaigns

Simulation: campaigns of three episodes (as for F3.1), each campaign with its own detection probability drawn from a beta distribution with intraclass correlation ρ; 4,000 simulations per cell. The "naive" analysis treats episodes as independent; the "adjusted" analysis applies the exact test to the effective sample size n / (1 + 2ρ).

| ρ | Naive type I error, 10 campaigns | Adjusted power, 10 / 20 / 30 campaigns |
|---|---|---|
| 0 | 0.05 | 0.82 / 0.97 / 1.00 |
| 0.1 | 0.06 | 0.57 / 0.95 / 0.99 |
| 0.3 | **0.10** | 0.31 / 0.87 / 0.96 |
| 0.5 | **0.13** | 0.38 / 0.76 / 0.90 |

Ignoring clustering roughly doubles or more than doubles the false-positive rate when ρ ≥ 0.3, while the adjusted test keeps it at or below 0.05. Under plausible clustering, about **20 to 30 independent campaigns** (60 to 90 episodes) are needed. Repeating one staged fault three times adds far less information than three different campaigns.

### 4.3 Comparison of two methods

Simulation: the 12 observed campaign-level differences between the fusion and kNN (leave-one-incident-out) are resampled with random signs, so that the null hypothesis holds exactly, then shifted by the effect of interest; Wilcoxon signed-rank test, α = 0.05; 2,000 simulations per cell. The type I error was checked at 0.04 to 0.05 in every scenario. "×1.5" inflates the observed spread to represent new sites and fault types.

| Effect | Spread | 12 campaigns | 30 | 50 | 100 |
|---|---|---|---|---|---|
| Recall +0.10 | ×1 | 0.49 | 0.77 | 0.94 | 1.00 |
| Recall +0.10 | ×1.5 | 0.39 | 0.67 | 0.86 | 0.99 |
| ROC-AUC +0.01 | ×1 | 0.34 | 0.67 | 0.87 | 0.99 |
| ROC-AUC +0.01 | ×1.5 | 0.26 | 0.56 | 0.77 | 0.97 |
| Equivalence within ±0.01 ROC-AUC (TOST, true difference 0) | ×1 | 0.09 | 0.64 | 0.90 | 1.00 |
| Equivalence within ±0.01 ROC-AUC (TOST, true difference 0) | ×1.5 | 0.01 | 0.10 | 0.43 | 0.88 |

About **30 to 50 independent campaigns** are needed to detect a 10-point recall difference, and 50 to 100 to establish equivalence within ±0.01 ROC-AUC. The current 12 campaigns cannot show that kNN and the ensemble are equivalent, only that no large difference was observed. The ROC-AUC rows assume an additive shift, which is unrealistic near the ceiling: recall at a fixed false-alarm budget is the preferred endpoint.

### 4.4 False-alarm rate

Resampling of 7-day blocks of daily event counts, assuming stationarity within a season:

| Season (observed rate) | 4 weeks | 13 weeks | 26 weeks | 52 weeks |
|---|---|---|---|---|
| June–September (6.25 / week) | ±44% | **±25%** | ±18% | ±13% |
| October–April (0.51 / week) | ±219% | ±151% | ±104% | ±81% |

Daily counts are overdispersed (variance 1.5 times the mean). A rate must be reported per season and estimated over **at least one full year**; more than one year is needed to separate the effect of the season from that of the year.

## 5. Data acquisition, in order of cost

1. **The control inverter of the same dataset.** The data descriptor reports a second inverter whose panels were never faulted. As a same-weather reference, it directly targets the low-sun and winter false alarms. Cost: none: the file is in the published record (`dt3_electrical_production_inverter_2.csv`), and `analysis/control_inverter.py` shows that it sees the same deficit as the faulted inverter at the winter alarms.
2. **Other versions of the same record.** The database was announced as still being updated: later versions may contain further campaigns and a second winter.
3. **Public multi-site data without fault labels**, such as NREL PVDAQ (over 160 systems): suitable for E2 (false-alarm rates across sites and seasons), not for detection.
4. **Public labelled fault data from other set-ups**, such as GPVS-Faults (16 laboratory experiments, 7 fault types, sampling every ~10 µs): fault-type diversity after aggregation to the same time scale, but under laboratory conditions; results must be reported separately, never pooled.
5. **Fault injection in simulation**: shading of controlled severity applied with a physical PV model to healthy data from several sites. It yields many episodes of graded severity and a sensitivity curve, but no evidence about real faults.
6. **A new staged campaign** with the original team, in a balanced and randomised schedule of fault type × severity × season × time of day: at least **37 independent episodes** for E1 (0.80 → 0.95), spread over **20 to 30 campaigns** if ρ is 0.3 or more, and at least **17 episodes per fault type** for a type-level claim with all episodes detected.

## 6. Limitations

- The intraclass correlation, the between-campaign spread and the false-alarm rates are estimated from **one installation and one year**. Their own uncertainty is large, which is why several scenarios are reported; the sample sizes are orders of magnitude, not commitments.
- The comparison simulations resample 12 observed differences: they cannot represent effects or failure modes absent from these 12 campaigns.
- The leave-one-incident-out results used for the variance estimates are optimistic by construction (evaluation within incident days; hyperparameters chosen on four of the incidents).
- All faults are **staged shading**. No sample size computed here licenses a claim about natural faults, other fault mechanisms or other plants.
- **Three episodes do not demonstrate generalisation.** The current results show feasibility on one installation and one fault family.
