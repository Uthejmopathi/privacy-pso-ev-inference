# A2P2 code - PSO of a privacy-constrained execution-mode policy (in-vehicle EV assistant)

## Run
```
pip install -r requirements.txt     # and Graphviz (`dot`) on the PATH
python3 run_all.py                   # reads ../data/A2P2_ev_assistant_inference.csv
```
Seeds are fixed (PSO/random search seeds 0-4, twin seeds 500+seed). Runtime about 6.5 min on 2 CPU cores (LightGBM n_jobs=1).
`python3 make_figures.py` redraws Figs. 2-6 from the CSVs in ../results without re-running the experiments.

## Files
- `run_all.py`: whole pipeline - MAD outlier repair + median imputation (fitted on 2025), LightGBM surrogates
  (deadline-hit classifier, log-energy, log-temperature-rise and quality regressors; 5-fold out-of-fold residuals),
  8-parameter scoring policy with hard privacy mask, inertia-weight PSO, baselines (always-local full / int8,
  always-offload with privacy mask, grid-tuned threshold policy, random search with the same budget, logged controller),
  5-seed evaluation, ablation (3 seeds), energy-budget sweep, thermal stress test, swarm-size study, Wilcoxon test,
  Graphviz flowchart (Fig. 1).
- `twin.py`: counterfactual outcome twin. Re-implements the generator's latency/energy/temperature equations and averages
  200 Monte-Carlo noise draws per request. Used ONLY for evaluation; the optimisers see only the surrogates.
- `make_figures.py`: Figs. 2-6.

## Outputs (../results; PNG 300 dpi copies of figures in ../figures)
| file | content |
|---|---|
| table_dataset_summary.csv | Table II |
| table_surrogate_accuracy.csv, deadline_surrogate_importance.csv | surrogate ROC-AUC/Brier/R2, LightGBM gain importance |
| main_comparison_per_seed.csv, table_main_comparison.csv | Table IV (deadline hit, quality, energy, violation rates, utility) |
| table_significance.csv | per-request Wilcoxon tests, PSO vs. each baseline (seed 0) |
| per_task_deadline_hit.csv, action_mix_by_task.csv | per-task results, Fig. 6 |
| convergence_per_seed.csv, pso_best_theta.csv | Fig. 2, tuned parameters |
| ablation_per_seed.csv, table_ablation.csv | Table V |
| sensitivity_epsE_per_seed.csv, table_sensitivity_epsE.csv | Fig. 4 |
| thermal_stress_per_seed.csv, table_thermal_stress.csv | Fig. 5 |
| sensitivity_swarm_per_seed.csv, table_sensitivity_swarm.csv | swarm-size study |
| metrics.json | all headline numbers used in the paper |
| fig*.jpg, fig1_flowchart.dot | figures |
