#!/usr/bin/env python3
"""A2P2 - PSO of a privacy-constrained adaptive execution-mode policy for an in-vehicle multimodal EV assistant.

Pipeline (all seeds fixed):
 1. load ../data/A2P2_ev_assistant_inference.csv, robust outlier repair + median imputation (fitted on 2025 only)
 2. LightGBM surrogates (context + action only, no outcome leakage): deadline_met classifier, log-energy and
    log-temperature-rise regressors (+ residual sigma), quality regressor. Train 2025, test 2026 H1.
 3. Parametric scoring policy pi_theta (8 params) with hard privacy mask; fitness J(theta) on 2025 contexts using
    out-of-fold surrogate predictions; optimised by inertia-weight PSO.
 4. Baselines: always-local, always-offload (privacy masked), grid-searched threshold policy, random search over
    theta with the same evaluation budget; logged field controller as reference.
 5. Counterfactual evaluation on 2026 H1 contexts with the twin (twin.py), 5 seeds; ablation; sensitivity
    (energy budget eps_E, thermal stress Delta T, swarm size); Wilcoxon test; figures + tables.
"""
import json, os, time, subprocess, shutil
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm, wilcoxon
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, r2_score, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from twin import simulate, ACTIONS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "A2P2_ev_assistant_inference.csv")
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
T0 = time.time()
SEEDS = [0, 1, 2, 3, 4]
T_LIM = 65.0          # NPU thermal limit (deg C) after the request
EPS_E, EPS_T = 0.10, 0.03   # allowed energy-allowance and thermal-limit violation rates
KAPPA = 200.0         # J per (% SoC / h): per-request energy allowance scale
LAMBDA_Q = 0.5        # weight of quality in utility
MU = 20.0             # penalty weight
N_PART, N_ITER = 30, 80
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "figure.dpi": 100})
C_PROP, C_GREY = "#c0392b", "#7f8c8d"
METRICS = {}


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    fig.savefig(os.path.join(RES, name + ".jpg"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ 1. data
df = pd.read_csv(DATA, parse_dates=["request_ts"])
raw_missing = df[["network_rtt_ms", "infotainment_soc_temp_c", "cpu_load"]].isna().sum().to_dict()
train_mask = df.request_ts < "2026-01-01"
tr = df[train_mask]
# robust outlier repair: values beyond median + 6*MAD (log scale) are set to NaN, then imputed
n_out = {}
for c in ["network_rtt_ms", "prompt_tokens"]:
    lv = np.log(df[c].clip(lower=1))
    med = np.log(tr[c].clip(lower=1)).median(); mad = (np.log(tr[c].clip(lower=1)) - med).abs().median()
    bad = (lv - med).abs() > 4 * 1.4826 * mad
    n_out[c] = int(bad.sum())
    df.loc[bad, c] = np.nan
# impute: medians fitted on 2025 (RTT median per signal decile, others global)
sig_bins = pd.qcut(tr.cellular_signal_dbm, 10, retbins=True)[1]; sig_bins[0], sig_bins[-1] = -1e9, 1e9
df["_sb"] = pd.cut(df.cellular_signal_dbm, sig_bins, labels=False)
rtt_med = df[train_mask].groupby("_sb").network_rtt_ms.median()
df["network_rtt_ms"] = df.network_rtt_ms.fillna(df._sb.map(rtt_med))
for c in ["infotainment_soc_temp_c", "cpu_load", "prompt_tokens"]:
    df[c] = df[c].fillna(df.loc[train_mask, c].median())
df = df.drop(columns="_sb")
df["hour"] = df.request_ts.dt.hour
df["work"] = df.prompt_tokens / 300 + 2.5 * df.image_megapixels
df["e_cap"] = KAPPA * df.assistant_energy_budget_pct_per_h * np.minimum(1.0, df.battery_soc_pct / 30.0)
df["a_idx"] = df.inference_action.map({a: i for i, a in enumerate(ACTIONS)})
CAT = ["task_type", "input_modality", "driving_state"]
for c in CAT:
    df[c] = df[c].astype("category")
CTX = ["task_type", "input_modality", "privacy_level", "prompt_tokens", "image_megapixels", "driving_state",
       "ambient_temp_c", "battery_soc_pct", "assistant_energy_budget_pct_per_h", "npu_temp_c",
       "infotainment_soc_temp_c", "cellular_signal_dbm", "network_rtt_ms", "cpu_load", "latency_deadline_ms", "hour"]
FEAT = CTX + ["a_idx"]
tr, te = df[train_mask].reset_index(drop=True), df[~train_mask].reset_index(drop=True)
print(f"train {len(tr)} test {len(te)} outliers {n_out}")

# dataset summary table
summ = [["Requests (train 2025 / test 2026 H1)", f"{len(tr):,} / {len(te):,}"],
        ["Vehicles / task types / modalities", f"{df.vehicle_id.nunique()} / {df.task_type.nunique()} / {df.input_modality.nunique()}"],
        ["Privacy level 0/1/2/3 (%)", " / ".join(f"{100*(df.privacy_level==k).mean():.1f}" for k in range(4))],
        ["Logged action full/int8/offload/defer (%)", " / ".join(f"{100*(df.inference_action==a).mean():.1f}" for a in ACTIONS)],
        ["NPU temperature, mean (max) [deg C]", f"{df.npu_temp_c.mean():.1f} ({df.npu_temp_c.max():.1f})"],
        ["Network RTT, median (p95) [ms]", f"{df.network_rtt_ms.median():.0f} ({df.network_rtt_ms.quantile(.95):.0f})"],
        ["Latency, median (p95) [ms]", f"{df.latency_ms.median():.0f} ({df.latency_ms.quantile(.95):.0f})"],
        ["Energy per request, median [J]", f"{df.energy_j.median():.2f}"],
        ["Logged deadline-hit rate (%)", f"{100*df.deadline_met.mean():.1f}"],
        ["Missing RTT / SoC temp / CPU (rows)", f"{raw_missing['network_rtt_ms']} / {raw_missing['infotainment_soc_temp_c']} / {raw_missing['cpu_load']}"],
        ["Outliers repaired RTT / tokens (rows)", f"{n_out['network_rtt_ms']} / {n_out['prompt_tokens']}"]]
pd.DataFrame(summ, columns=["Item", "Value"]).to_csv(os.path.join(RES, "table_dataset_summary.csv"), index=False)

# ------------------------------------------------------------------ 2. surrogates
LGB = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=30, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.9, n_jobs=1, verbose=-1, random_state=0)
targets = {"dl": ("clf", tr.deadline_met.values),
           "le": ("reg", np.log(tr.energy_j.values)),
           "lr": ("reg", np.log(tr.npu_temp_rise_c.values)),
           "q": ("reg", tr.response_quality.values)}


def fit(kind, X, y):
    m = (lgb.LGBMClassifier if kind == "clf" else lgb.LGBMRegressor)(**LGB)
    return m.fit(X, y)


def pred(m, kind, X):
    return m.predict_proba(X)[:, 1] if kind == "clf" else m.predict(X)


def ctx_action(d, a):
    X = d[CTX].copy(); X["a_idx"] = a
    return X


models, oof, sig = {}, {}, {}
kf = KFold(5, shuffle=True, random_state=0)
for k, (kind, y) in targets.items():
    o = np.zeros(len(tr))
    for i_tr, i_va in kf.split(tr):
        m = fit(kind, tr.loc[i_tr, FEAT], y[i_tr]); o[i_va] = pred(m, kind, tr.loc[i_va, FEAT])
    oof[k] = o
    models[k] = fit(kind, tr[FEAT], y)
    if kind == "reg":
        sig[k] = float(np.std(y - o))
# surrogate accuracy on the 2026 H1 test logs (logged actions)
acc = []
p = pred(models["dl"], "clf", te[FEAT])
acc.append(["Deadline met (clf)", "ROC-AUC", roc_auc_score(te.deadline_met, p)])
acc.append(["Deadline met (clf)", "Brier", brier_score_loss(te.deadline_met, p)])
for k, col, nm in [("le", "energy_j", "log energy"), ("lr", "npu_temp_rise_c", "log temp. rise")]:
    acc.append([nm + " (reg)", "R2", r2_score(np.log(te[col]), pred(models[k], "reg", te[FEAT]))])
acc.append(["quality (reg)", "R2", r2_score(te.response_quality, pred(models["q"], "reg", te[FEAT]))])
acc = pd.DataFrame(acc, columns=["Surrogate", "Metric", "Value"]); acc.to_csv(os.path.join(RES, "table_surrogate_accuracy.csv"), index=False)
METRICS["surrogate"] = {f"{r.Surrogate}|{r.Metric}": round(r.Value, 4) for r in acc.itertuples()}
METRICS["surrogate_sigma"] = {k: round(v, 4) for k, v in sig.items()}
imp = pd.Series(models["dl"].booster_.feature_importance("gain"), index=FEAT).sort_values(ascending=False)
imp.to_csv(os.path.join(RES, "deadline_surrogate_importance.csv"), header=["gain"])
print("surrogates", acc.Value.round(3).tolist(), f"{time.time()-T0:.0f}s")


def surrogate_tensor(d, use_oof=False):
    """Per request x action: p_dl, mean energy, P(E>cap), P(T+rise>T_lim), quality, exposure."""
    n = len(d); S = {k: np.zeros((n, 4)) for k in ["pdl", "E", "pE", "pT", "q", "x"]}
    for a in range(4):
        X = ctx_action(d, a)
        if use_oof:   # out-of-fold for the logged action, full model for counterfactual actions
            pass
        pdl = pred(models["dl"], "clf", X); le = pred(models["le"], "reg", X)
        lr = pred(models["lr"], "reg", X); q = pred(models["q"], "reg", X)
        S["pdl"][:, a] = pdl
        S["E"][:, a] = np.exp(le + sig["le"] ** 2 / 2)
        S["pE"][:, a] = 1 - norm.cdf((np.log(d.e_cap.values) - le) / sig["le"])
        head = T_LIM - d.npu_temp_c.values
        S["pT"][:, a] = np.where(head <= 0, 1.0, 1 - norm.cdf((np.log(np.maximum(head, 1e-6)) - lr) / sig["lr"]))
        S["q"][:, a] = q
        S["x"][:, a] = (a == 2) * d.privacy_level.values * 0.25 * 0.75
    if use_oof:  # replace the logged-action column with out-of-fold predictions to reduce optimism
        ai = d.a_idx.values; r = np.arange(n)
        S["pdl"][r, ai] = oof["dl"]
        S["E"][r, ai] = np.exp(oof["le"] + sig["le"] ** 2 / 2)
        S["pE"][r, ai] = 1 - norm.cdf((np.log(d.e_cap.values) - oof["le"]) / sig["le"])
        S["q"][r, ai] = oof["q"]
    return S


def mask_of(d, privacy_mask=True):
    M = np.ones((len(d), 4), bool)
    if privacy_mask:
        M[:, 2] = d.privacy_level.values < 2
    return M


S_tr = surrogate_tensor(tr, use_oof=True)
S_te = surrogate_tensor(te)
M_tr, M_te = mask_of(tr), mask_of(te)


# ------------------------------------------------------------------ 3. policy + fitness
# theta = [w_dl, w_q, w_E, w_T, w_X, b_int8, b_off, b_defer]
LB = np.array([0, 0, 0, 0, 0, -2, -2, -2.0]); UB = np.array([5, 5, 5, 5, 5, 2, 2, 2.0])
PNAMES = ["w_dl", "w_q", "w_E", "w_T", "w_X", "b_int8", "b_off", "b_def"]


def policy(theta, S, M):
    w_dl, w_q, w_E, w_T, w_X, b1, b2, b3 = theta
    sc = w_dl * S["pdl"] + w_q * S["q"] - w_E * S["pE"] - w_T * S["pT"] - w_X * S["x"]
    sc = sc + np.array([0, b1, b2, b3])
    sc = np.where(M, sc, -1e9)
    return sc.argmax(1)


def agg(S, a, priv):
    r = np.arange(len(a))
    return dict(D=S["pdl"][r, a].mean(), Q=S["q"][r, a].mean(), VE=S["pE"][r, a].mean(), VT=S["pT"][r, a].mean(),
                VP=((a == 2) & (priv >= 2)).mean())


def fitness_from(m, eps_E=EPS_E, eps_T=EPS_T):
    util = m["D"] + LAMBDA_Q * m["Q"]
    pen = max(0, m["VE"] - eps_E) + max(0, m["VT"] - eps_T) + m["VP"]
    return -util + MU * pen


def make_fitness(S, M, priv, **kw):
    return lambda th: fitness_from(agg(S, policy(th, S, M), priv), **kw)


def pso(f, seed, n_part=N_PART, n_iter=N_ITER, w0=0.9, w1=0.4, c1=1.49, c2=1.49):
    rng = np.random.default_rng(seed)
    X = rng.uniform(LB, UB, (n_part, len(LB))); V = rng.uniform(-1, 1, X.shape) * (UB - LB) * 0.1
    P = X.copy(); Pf = np.array([f(x) for x in X]); g = P[Pf.argmin()].copy(); gf = Pf.min()
    hist = [gf]; vmax = 0.2 * (UB - LB)
    for t in range(n_iter):
        w = w0 - (w0 - w1) * t / max(1, n_iter - 1)
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        V = np.clip(w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X), -vmax, vmax)
        X = X + V
        out = (X < LB) | (X > UB); X = np.clip(X, LB, UB); V[out] *= -0.5   # reflective bound handling
        fx = np.array([f(x) for x in X])
        imp_ = fx < Pf; P[imp_] = X[imp_]; Pf[imp_] = fx[imp_]
        if Pf.min() < gf:
            gf = Pf.min(); g = P[Pf.argmin()].copy()
        hist.append(gf)
    return g, gf, np.array(hist)


def random_search(f, seed, n_eval=N_PART * (N_ITER + 1)):
    rng = np.random.default_rng(1000 + seed); best, bf, hist = None, np.inf, []
    for i in range(n_eval):
        x = rng.uniform(LB, UB); fx = f(x)
        if fx < bf:
            best, bf = x, fx
        if (i + 1) % N_PART == 0:
            hist.append(bf)
    return best, bf, np.array(hist)


# threshold policy (hand-designed rule, 4 tuned thresholds):
#   privacy >= 2 : local_full if (T_npu < tT and SoC > tS and deadline >= tD) else local_quantized
#   privacy <  2 : secure_offload if RTT < tR, else the same local rule
def threshold_policy(d, tT, tS, tR, tD):
    local = np.where((d.npu_temp_c.values < tT) & (d.battery_soc_pct.values > tS) &
                     (d.latency_deadline_ms.values >= tD), 0, 1)
    return np.where((d.privacy_level.values < 2) & (d.network_rtt_ms.values < tR), 2, local)


def tune_threshold(d, S, eps_E=EPS_E, eps_T=EPS_T):
    best, bf = None, np.inf
    for tT in [35, 40, 45, 50, 55, 60, 200]:
        for tS in [0, 20, 40, 60, 80, 101]:
            for tR in [0, 100, 150, 200, 300, 500, 1e9]:
                for tD in [0, 400, 1200, 2500, 1e9]:
                    f = fitness_from(agg(S, threshold_policy(d, tT, tS, tR, tD), d.privacy_level.values), eps_E, eps_T)
                    if f < bf:
                        best, bf = (tT, tS, tR, tD), f
    return best, bf


# ------------------------------------------------------------------ 4. evaluation with the twin
def twin_ctx(d, dT=0.0):
    return dict(work=d.work.values, img=d.image_megapixels.values, npu=d.npu_temp_c.values + dT,
                cpu=d.cpu_load.values, rtt=d.network_rtt_ms.values, dbm=d.cellular_signal_dbm.values,
                priv=d.privacy_level.values, deadline=d.latency_deadline_ms.values)


def evaluate(d, a, seed, dT=0.0):
    rng = np.random.default_rng(500 + seed)
    o = simulate(twin_ctx(d, dT), np.asarray(a), d.e_cap.values, T_LIM, rng)
    priv = d.privacy_level.values
    return {"deadline_hit": o["p_dl"].mean(), "quality": o["quality"].mean(), "energy_J": o["energy"].mean(),
            "energy_viol": o["p_e_viol"].mean(), "thermal_viol": o["p_t_viol"].mean(),
            "privacy_viol": ((np.asarray(a) == 2) & (priv >= 2)).mean(), "exposure": o["exposure"].mean(),
            "utility": o["p_dl"].mean() + LAMBDA_Q * o["quality"].mean(), "_pdl": o["p_dl"]}


fit_tr = make_fitness(S_tr, M_tr, tr.privacy_level.values)
thr_best, thr_f = tune_threshold(tr, S_tr)
print("threshold", thr_best, round(thr_f, 4))
METRICS["threshold_params"] = dict(zip(["T_npu", "SoC", "RTT", "deadline"], [float(v) for v in thr_best]))

rows, conv, thetas, per_req = [], {"PSO": [], "Random search": []}, [], {}
for s in SEEDS:
    th_p, f_p, h_p = pso(fit_tr, s); th_r, f_r, h_r = random_search(fit_tr, s)
    conv["PSO"].append(h_p); conv["Random search"].append(h_r); thetas.append(th_p)
    pols = {"Always-local (full)": np.zeros(len(te), int), "Always-local (int8)": np.ones(len(te), int),
            "Always-offload (masked)": np.where(te.privacy_level.values < 2, 2, 1),
            "Threshold (tuned)": threshold_policy(te, *thr_best),
            "Random search": policy(th_r, S_te, M_te),
            "PSO (proposed)": policy(th_p, S_te, M_te),
            "Logged field controller": te.a_idx.values}
    for name, a in pols.items():
        r = evaluate(te, a, s); per_req[(name, s)] = r.pop("_pdl")
        r.update(method=name, seed=s, train_fitness=(f_p if name.startswith("PSO") else f_r if name == "Random search" else np.nan))
        rows.append(r)
    if s == 0:
        a_pso0, a_log = pols["PSO (proposed)"], te.a_idx.values
    print(f"seed {s} pso {f_p:.4f} rs {f_r:.4f} {time.time()-T0:.0f}s")
main = pd.DataFrame(rows); main.to_csv(os.path.join(RES, "main_comparison_per_seed.csv"), index=False)
pd.concat([pd.DataFrame({"method": nm, "seed": s, "evals": np.arange(len(h)) * N_PART + N_PART, "best_neg_J": -h})
           for nm in conv for s, h in zip(SEEDS, conv[nm])]).to_csv(os.path.join(RES, "convergence_per_seed.csv"), index=False)
cols = ["deadline_hit", "quality", "energy_J", "energy_viol", "thermal_viol", "privacy_viol", "utility"]
order = ["Always-local (full)", "Always-local (int8)", "Always-offload (masked)", "Threshold (tuned)", "Random search", "PSO (proposed)", "Logged field controller"]
tab = main.groupby("method")[cols].agg(["mean", "std"]).loc[order]
tab.columns = [f"{a}_{b}" for a, b in tab.columns]; tab.to_csv(os.path.join(RES, "table_main_comparison.csv"))
METRICS["main"] = {m: {c: [round(tab.loc[m, c + "_mean"], 4), round(tab.loc[m, c + "_std"], 4)] for c in cols} for m in order}
pd.DataFrame(thetas, columns=PNAMES).assign(seed=SEEDS).to_csv(os.path.join(RES, "pso_best_theta.csv"), index=False)
METRICS["pso_theta_mean"] = dict(zip(PNAMES, np.round(np.mean(thetas, 0), 3).tolist()))
METRICS["pso_train_fitness"] = [round(float(main[(main.method == "PSO (proposed)") & (main.seed == s)].train_fitness.iloc[0]), 4) for s in SEEDS]
METRICS["rs_train_fitness"] = [round(float(main[(main.method == "Random search") & (main.seed == s)].train_fitness.iloc[0]), 4) for s in SEEDS]

# significance: per-request deadline-hit probability, PSO vs each baseline (seed 0), Wilcoxon signed-rank
sig_rows = []
for b in order:
    if b == "PSO (proposed)":
        continue
    dlt = per_req[("PSO (proposed)", 0)] - per_req[(b, 0)]
    nz = dlt[np.abs(dlt) > 1e-12]
    pv = wilcoxon(nz).pvalue if len(nz) > 10 else np.nan
    sig_rows.append([b, dlt.mean(), (dlt > 0).mean(), (dlt < 0).mean(), pv])
sigt = pd.DataFrame(sig_rows, columns=["baseline", "mean_delta_deadline_hit", "frac_better", "frac_worse", "wilcoxon_p"])
sigt.to_csv(os.path.join(RES, "table_significance.csv"), index=False)
METRICS["significance"] = {r.baseline: {"delta": round(r.mean_delta_deadline_hit, 4), "p": float(f"{r.wilcoxon_p:.3g}")} for r in sigt.itertuples()}

# action mix per task (seed 0)
mix = []
for nm, a in [("Logged", a_log), ("PSO", a_pso0)]:
    t = pd.crosstab(te.task_type.astype(str), pd.Series(a).map(dict(enumerate(ACTIONS))), normalize="index")
    t = t.reindex(columns=ACTIONS, fill_value=0); t["policy"] = nm; mix.append(t)
mix = pd.concat(mix); mix.to_csv(os.path.join(RES, "action_mix_by_task.csv"))
METRICS["pso_action_share"] = {ACTIONS[k]: round(float((a_pso0 == k).mean()), 4) for k in range(4)}
METRICS["logged_action_share"] = {ACTIONS[k]: round(float((a_log == k).mean()), 4) for k in range(4)}
# per-task deadline hit (seed 0)
pt = pd.DataFrame({"task": te.task_type.astype(str), "PSO": per_req[("PSO (proposed)", 0)],
                   "Threshold": per_req[("Threshold (tuned)", 0)], "Logged": per_req[("Logged field controller", 0)]})
pt.groupby("task").mean().to_csv(os.path.join(RES, "per_task_deadline_hit.csv"))

# ------------------------------------------------------------------ 5. ablation (3 seeds)
def run_variant(kind, s):
    S, M = S_tr, M_tr
    lb, ub = LB.copy(), UB.copy()
    if kind == "no privacy mask (soft exposure only)":
        M = mask_of(tr, False)
    f0 = make_fitness(S, M, tr.privacy_level.values)
    if kind == "no thermal term (w_T=0)":
        f = lambda th: f0(np.r_[th[:3], 0, th[4:]])
    elif kind == "no energy term (w_E=0)":
        f = lambda th: f0(np.r_[th[:2], 0, th[3:]])
    elif kind == "no penalties in fitness (utility only)":
        f = lambda th: -(lambda m: m["D"] + LAMBDA_Q * m["Q"])(agg(S, policy(th, S, M), tr.privacy_level.values))
    else:
        f = f0
    if kind == "untuned surrogate-greedy (theta fixed)":
        th = np.array([1, 0.5, 1, 1, 1, 0, 0, 0.0])
    else:
        th, _, _ = pso(f, s)
        if kind == "no thermal term (w_T=0)": th[3] = 0
        if kind == "no energy term (w_E=0)": th[2] = 0
    Mt = mask_of(te, kind != "no privacy mask (soft exposure only)")
    return evaluate(te, policy(th, S_te, Mt), s)


VARIANTS = ["full PSO policy", "no thermal term (w_T=0)", "no energy term (w_E=0)", "no privacy mask (soft exposure only)",
            "no penalties in fitness (utility only)", "untuned surrogate-greedy (theta fixed)"]
ab = []
for v in VARIANTS:
    for s in SEEDS[:3]:
        r = run_variant(v, s); r.pop("_pdl"); r.update(variant=v, seed=s); ab.append(r)
ab = pd.DataFrame(ab); ab.to_csv(os.path.join(RES, "ablation_per_seed.csv"), index=False)
abt = ab.groupby("variant")[cols].agg(["mean", "std"]).loc[VARIANTS]
abt.columns = [f"{a}_{b}" for a, b in abt.columns]; abt.to_csv(os.path.join(RES, "table_ablation.csv"))
METRICS["ablation"] = {v: {c: [round(abt.loc[v, c + "_mean"], 4), round(abt.loc[v, c + "_std"], 4)] for c in cols} for v in VARIANTS}
print(f"ablation done {time.time()-T0:.0f}s")

# ------------------------------------------------------------------ 6. sensitivity
# (a) energy-violation budget eps_E -> trade-off front (PSO vs threshold), seeds 0-2
sens = []
for eps in [0.04, 0.07, 0.10, 0.15, 0.20]:
    fe = make_fitness(S_tr, M_tr, tr.privacy_level.values, eps_E=eps)
    tb, _ = tune_threshold(tr, S_tr, eps_E=eps)
    for s in SEEDS[:3]:
        th, _, _ = pso(fe, s)
        for nm, a in [("PSO", policy(th, S_te, M_te)), ("Threshold", threshold_policy(te, *tb))]:
            r = evaluate(te, a, s); r.pop("_pdl"); r.update(eps_E=eps, method=nm, seed=s); sens.append(r)
sens = pd.DataFrame(sens); sens.to_csv(os.path.join(RES, "sensitivity_epsE_per_seed.csv"), index=False)
sens_t = sens.groupby(["method", "eps_E"])[["deadline_hit", "energy_viol", "energy_J", "quality"]].agg(["mean", "std"])
sens_t.columns = [f"{a}_{b}" for a, b in sens_t.columns]; sens_t.to_csv(os.path.join(RES, "table_sensitivity_epsE.csv"))
METRICS["sens_epsE"] = {f"{m}|{e}": [round(sens_t.loc[(m, e), "deadline_hit_mean"], 4), round(sens_t.loc[(m, e), "energy_viol_mean"], 4)]
                        for m, e in sens_t.index}
print(f"eps sens done {time.time()-T0:.0f}s")

# (b) thermal stress: NPU temperatures shifted by dT (heat wave / sustained highway load). PSO re-optimised on the
#     shifted 2025 contexts (adaptive), threshold re-tuned, static PSO policy from dT=0 kept as reference.
th_stat = thetas[0]; stress = []
for dT in [0, 5, 10, 15, 20]:
    trs, tes = tr.copy(), te.copy(); trs["npu_temp_c"] += dT; tes["npu_temp_c"] += dT
    Ss_tr, Ss_te = surrogate_tensor(trs), surrogate_tensor(tes)
    if dT == 0:
        Ss_tr = S_tr
    fs = make_fitness(Ss_tr, M_tr, tr.privacy_level.values)
    tb, _ = tune_threshold(trs, Ss_tr)
    for s in SEEDS[:3]:
        th, _, _ = pso(fs, s)
        for nm, a in [("PSO (re-optimised)", policy(th, Ss_te, M_te)), ("PSO (static, dT=0)", policy(th_stat, Ss_te, M_te)),
                      ("Threshold (re-tuned)", threshold_policy(tes, *tb)), ("Always-local (int8)", np.ones(len(te), int))]:
            r = evaluate(te, a, s, dT=dT); r.pop("_pdl"); r.update(dT=dT, method=nm, seed=s); stress.append(r)
stress = pd.DataFrame(stress); stress.to_csv(os.path.join(RES, "thermal_stress_per_seed.csv"), index=False)
st_t = stress.groupby(["method", "dT"])[["deadline_hit", "thermal_viol", "energy_viol", "utility"]].mean()
st_t.to_csv(os.path.join(RES, "table_thermal_stress.csv"))
METRICS["thermal_stress"] = {f"{m}|{d}": [round(st_t.loc[(m, d), "deadline_hit"], 4), round(st_t.loc[(m, d), "thermal_viol"], 4)]
                             for m, d in st_t.index}
print(f"stress done {time.time()-T0:.0f}s")

# (c) swarm size / iterations at a fixed budget of ~2430 evaluations
sw = []
for npart, nit in [(10, 242), (20, 120), (30, 80), (60, 39)]:
    for s in SEEDS[:3]:
        th, f, _ = pso(fit_tr, s, npart, nit); sw.append([npart, nit, s, f])
sw = pd.DataFrame(sw, columns=["particles", "iterations", "seed", "train_fitness"])
sw.to_csv(os.path.join(RES, "sensitivity_swarm_per_seed.csv"), index=False)
swt = sw.groupby(["particles", "iterations"]).train_fitness.agg(["mean", "std"]).reset_index()
swt.to_csv(os.path.join(RES, "table_sensitivity_swarm.csv"), index=False)
METRICS["swarm"] = [[int(r.particles), int(r.iterations), round(r.mean, 4), round(r.std, 4)] for r in swt.itertuples()]

# ------------------------------------------------------------------ 7. figures (drawn from the result CSVs)
import make_figures
make_figures.main(RES, FIG, EPS_E, EPS_T, N_PART)

# ------------------------------------------------------------------ 8. flowchart (Graphviz)
dot = r"""digraph G {
graph [rankdir=TB, fontname="Helvetica", fontsize=14, nodesep=0.15, ranksep=0.22, margin=0.02, newrank=true, labeljust=l];
node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13, fillcolor="#ececec", color="#7f8c8d", margin="0.06,0.03"];
edge [color="#555555", arrowsize=0.6, fontname="Helvetica", fontsize=12];
subgraph cluster_data { label="1. Data layer"; style="rounded,filled"; fillcolor="#f5f6f7"; color="#bdc3c7";
  logs [label="EV logs: 15,000\nrequests (x, a)"];
  prep [label="MAD outliers,\nimputation"];
}
subgraph cluster_sur { label="2. Surrogates (LightGBM)"; labeljust=r; style="rounded,filled"; fillcolor="#f5f6f7"; color="#bdc3c7";
  sdl [label="p_dl\nEq. (1)"]; sen [label="P_E\nEq. (2)"]; sth [label="P_T\nEq. (3)"];
}
subgraph cluster_pso { label="3. Proposed PSO policy search"; style="rounded,filled"; fillcolor="#fdf1ef"; color="#e6b0aa";
  node [fillcolor="#f8d3cc", color="#c0392b"];
  init [label="Init swarm\nθ_i ∈ [lb, ub]^8"];
  pol [label="Score s_a, Eq. (4)\nmask: ρ≥2 → no offload"];
  fit [label="Fitness J(θ)\nEq. (6)"];
  upd [label="Update v_i, θ_i\nEqs. (7)-(8)"];
  conv [shape=diamond, style="filled", label="k = K?", margin="0.02,0.01"];
}
subgraph cluster_eval { label="4. Deployment and evaluation"; style="rounded,filled"; fillcolor="#f5f6f7"; color="#bdc3c7";
  dep [label="a* = argmax s_a\n(per request)", fillcolor="#f8d3cc", color="#c0392b"];
  base [label="Baselines:\nlocal, offload,\nthreshold, RS"];
  twin [label="Outcome twin\n200 MC draws"];
  met [label="DH, E, V_E,\nV_T, V_P (9)"];
}
{rank=same; logs; prep}
{rank=same; sdl; sen; sth}
{rank=same; init; pol}
{rank=same; conv; upd; fit}
{rank=same; dep; base}
{rank=same; twin; met}
logs -> prep;
prep -> sdl; prep -> sen; prep -> sth;
sen -> pol [label=" S(x,a)"];
init -> pol; pol -> fit; fit -> upd; upd -> conv;
conv -> pol [label="no", color="#c0392b", fontcolor="#c0392b"];
conv -> dep [label=" yes: θ*=g"];
dep -> twin; base -> twin; twin -> met;
}"""
with open(os.path.join(RES, "fig1_flowchart.dot"), "w") as fh:
    fh.write(dot)
subprocess.run(["dot", "-Tpng", "-Gdpi=300", os.path.join(RES, "fig1_flowchart.dot"), "-o", os.path.join(FIG, "fig1_flowchart.png")], check=True)
from PIL import Image
Image.open(os.path.join(FIG, "fig1_flowchart.png")).convert("RGB").save(os.path.join(RES, "fig1_flowchart.jpg"), quality=95)

METRICS["config"] = dict(T_LIM=T_LIM, EPS_E=EPS_E, EPS_T=EPS_T, KAPPA=KAPPA, LAMBDA_Q=LAMBDA_Q, MU=MU, N_PART=N_PART,
                         N_ITER=N_ITER, seeds=SEEDS, n_train=len(tr), n_test=len(te))
METRICS["runtime_s"] = round(time.time() - T0, 1)
with open(os.path.join(RES, "metrics.json"), "w") as fh:
    json.dump(METRICS, fh, indent=1, default=float)
print(f"done in {time.time()-T0:.0f}s")
