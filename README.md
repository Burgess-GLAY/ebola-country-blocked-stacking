# Country-blocked validation of a leakage-free stacked ensemble for district-level Ebola risk prediction

Data and code for the paper *Country-blocked validation of an explainable stacked ensemble for district-level Ebola virus disease risk prediction in five African countries* (Glay & He, 2026).

## What this repository contains

| Path | Content |
|---|---|
| `data/evd_final_feature_matrix.csv` | Harmonised district-month matrix: 1,773 rows, 5 countries (Guinea, Liberia, Sierra Leone, Uganda, DRC), 10 covariates, binary outcome `outbreak_binary` (571 positives) |
| `code/core.py` | Pipeline: SMOTE-ENN, XGBoost / Random Forest / Gradient Boosting, leakage-free nested meta-learner (`meta_oof`) |
| `code/run_main.py` | Country-blocked validation, three meta-learner constructions (`old`, `strat`, `group`) |
| `code/run_schemes.py` | District-blocked and random stratified validation |
| `code/run_abl.py` | Spatial-lag covariate ablation |
| `code/run_roll3.py` | Rolling-window forecasts (1-3 months ahead) |
| `code/run_l2.py` | Level 2 KernelSHAP attribution of the meta-learner |
| `code/boot.py` | Paired bootstrap (stack vs Random Forest, per-country intervals) |
| `code/agg_main.py`, `code/agg2.py` | Aggregation of results into the paper's tables |
| `results/` | Aggregated results (JSON) used in the paper |

## Reproduce

```bash
pip install -r requirements.txt
mkdir -p results
for s in 42 43 44; do python code/run_main.py $s; python code/run_abl.py $s; python code/run_schemes.py $s; done
python code/run_roll3.py; python code/run_l2.py; python code/boot.py
python code/agg_main.py; python code/agg2.py
```

Runs are CPU-only; the full set takes a few hours on two cores.

## The leakage-free meta-learner

In the usual stacking recipe the meta-learner is trained on out-of-fold probabilities produced by cross-validating the base learners on the **resampled** (SMOTE-ENN) rows. Synthetic neighbours of a row then fall into other inner folds, so the out-of-fold probabilities are optimistic. `meta_oof` in `code/core.py` instead (i) generates the meta-features on the **original** rows and (ii) refits SMOTE-ENN inside every inner training split. The primary construction (`mode='group'`) uses leave-one-training-country-out inner folds, so the meta-learner is fitted to predictions for a country the base learners have not seen.

Headline results (mean of three seeds, country-blocked): stack AUC-ROC 0.784 vs Random Forest 0.787; random stratified folds 0.945 and district-blocked folds 0.938; the meta-learner puts about 81% of its attribution on Random Forest, whereas the conventional construction suggests about 37% / 37% / 26% for XGBoost / Random Forest / Gradient Boosting.

## Data sources

The matrix was harmonised from public sources: WHO and INRB/INSP situation reports and data for the 2026 DRC outbreak, published Uganda outbreak records, and the Humanitarian Data Exchange / ReliefWeb West Africa Ebola records, combined with ERA5 (climate), WorldPop (population), Hansen forest cover and geoBoundaries (administrative boundaries). Please cite and respect the original sources' terms. Note: the matrix contains 2 exact duplicate rows (the analysis uses all 1,773 rows).

## License

Code: MIT (see `LICENSE`). The data matrix is derived from the public sources above; reuse is subject to those sources' terms.

## Citation

See `CITATION.cff`.

## Results files

`results/agg2.json` (validation schemes, ablation, rolling-window), `results/boot.json` (bootstrap), `results/l2.json` (Level 2 attribution, three seeds), `results/main_summary.txt` (country-blocked metrics for every model and meta-learner construction).
