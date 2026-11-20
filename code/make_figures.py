"""Draws Figs. 2-6 of A2P2 from the CSV files in ../results (called by run_all.py; can also be run on its own)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_PROP, C_GREY = "#c0392b", "#7f8c8d"
ACTIONS = ["local_full", "local_quantized", "secure_offload", "deferred"]
ORDER = ["Always-local (full)", "Always-local (int8)", "Always-offload (masked)", "Threshold (tuned)", "Random search",
         "PSO (proposed)", "Logged field controller"]
SHORT = {"Always-local (full)": "Local-F", "Always-local (int8)": "Local-Q", "Always-offload (masked)": "Offload",
         "Threshold (tuned)": "Thresh.", "Random search": "RS", "PSO (proposed)": "PSO", "Logged field controller": "Logged"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})


def _save(fig, FIG, RES, name):
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    fig.savefig(os.path.join(RES, name + ".jpg"), dpi=300)
    plt.close(fig)


def main(RES, FIG, EPS_E=0.10, EPS_T=0.03, N_PART=30):
    main_ps = pd.read_csv(os.path.join(RES, "main_comparison_per_seed.csv"))
    tab = main_ps.groupby("method").agg(["mean", "std"])
    # Fig 2: convergence (stored per seed)
    conv = pd.read_csv(os.path.join(RES, "convergence_per_seed.csv"))
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    for nm, c in [("PSO", C_PROP), ("Random search", C_GREY)]:
        H = conv[conv.method == nm].pivot(index="evals", columns="seed", values="best_neg_J")
        m, sd = H.mean(1), H.std(1)
        ax.plot(H.index, m, color=c, label=nm); ax.fill_between(H.index, m - sd, m + sd, color=c, alpha=0.2)
    ax.set_xlabel("Fitness evaluations (log scale)"); ax.set_ylabel(r"Best training value $-J(\theta)$")
    ax.set_xscale("log"); ax.legend(frameon=False, loc="lower right"); ax.grid(alpha=0.3)
    _save(fig, FIG, RES, "fig2_convergence")

    # Fig 3: comparison bars
    fig, axs = plt.subplots(1, 3, figsize=(3.45, 2.35))
    for ax, c, lab in zip(axs, ["deadline_hit", "energy_viol", "thermal_viol"], ["Deadline hit", "Energy viol.", "Thermal viol."]):
        m = [tab.loc[o, (c, "mean")] for o in ORDER]; sd = [tab.loc[o, (c, "std")] for o in ORDER]
        ax.bar(range(7), m, yerr=sd, capsize=1.5, color=[C_PROP if o.startswith("PSO") else C_GREY for o in ORDER])
        ax.set_xticks(range(7)); ax.set_xticklabels([SHORT[o] for o in ORDER], rotation=90)
        ax.set_title(lab, fontsize=7.5); ax.grid(axis="y", alpha=0.3)
    axs[0].set_ylim(0.5, 0.85)
    axs[1].axhline(EPS_E, ls=":", color="k", lw=0.8); axs[2].axhline(EPS_T, ls=":", color="k", lw=0.8)
    _save(fig, FIG, RES, "fig3_comparison")

    # Fig 4: energy-budget trade-off
    sens = pd.read_csv(os.path.join(RES, "sensitivity_epsE_per_seed.csv"))
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for nm, c, mk in [("PSO", C_PROP, "o"), ("Threshold", C_GREY, "s")]:
        g = sens[sens.method == nm].groupby("eps_E")[["energy_viol", "deadline_hit"]].agg(["mean", "std"])
        ax.errorbar(g[("energy_viol", "mean")], g[("deadline_hit", "mean")], yerr=g[("deadline_hit", "std")], color=c,
                    marker=mk, ms=3.5, lw=1, capsize=2, label=nm + (r" (all $\varepsilon_E$)" if nm == "Threshold" else r" ($\varepsilon_E$ sweep)"))
        if nm == "PSO":
            for e, row in g.iterrows():
                if e in (0.04, 0.07, 0.10, 0.20):
                    off = {0.04: (4, -2), 0.07: (5, -4), 0.10: (-10, 6), 0.20: (2, -11)}[e]
                    ax.annotate(f"{e:.2f}", (row[("energy_viol", "mean")], row[("deadline_hit", "mean")]), fontsize=6.5,
                                xytext=off, textcoords="offset points", color=C_PROP)
    for nm, off in [("Always-local (full)", (-30, 4)), ("Always-local (int8)", (4, -2)), ("Logged field controller", (4, 2))]:
        x, y = tab.loc[nm, ("energy_viol", "mean")], tab.loc[nm, ("deadline_hit", "mean")]
        ax.scatter(x, y, marker="x", color="k", s=14); ax.annotate(SHORT[nm], (x, y), fontsize=6.5, xytext=off, textcoords="offset points")
    ax.set_xlabel("Energy-allowance violation rate"); ax.set_ylabel("Deadline-hit rate")
    ax.legend(frameon=False, loc="center right", fontsize=6.5); ax.grid(alpha=0.3)
    _save(fig, FIG, RES, "fig4_tradeoff_epsE")

    # Fig 5: thermal stress
    stress = pd.read_csv(os.path.join(RES, "thermal_stress_per_seed.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.45))
    sty = {"PSO (re-optimised)": (C_PROP, "-", "o"), "PSO (static, dT=0)": (C_PROP, "--", "^"),
           "Threshold (re-tuned)": (C_GREY, "-", "s"), "Always-local (int8)": ("k", ":", "x")}
    for nm, (c, ls, mk) in sty.items():
        g = stress[stress.method == nm].groupby("dT")[["deadline_hit", "thermal_viol"]].mean()
        axs[0].plot(g.index, g.deadline_hit, color=c, ls=ls, marker=mk, ms=3, label=nm.replace("dT", "ΔT"))
        axs[1].plot(g.index, g.thermal_viol, color=c, ls=ls, marker=mk, ms=3)
    axs[0].set_ylabel("Deadline-hit rate"); axs[1].set_ylabel("Thermal-limit violation rate")
    for ax in axs:
        ax.set_xlabel("NPU shift ΔT (°C)"); ax.grid(alpha=0.3); ax.set_xticks([0, 5, 10, 15, 20])
    axs[1].axhline(EPS_T, ls=":", color="k", lw=0.8)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.83))
    fig.savefig(os.path.join(FIG, "fig5_thermal_stress.png"), dpi=300); fig.savefig(os.path.join(RES, "fig5_thermal_stress.jpg"), dpi=300)
    plt.close(fig)

    # Fig 6: action mix per task (horizontal stacked bars)
    mix = pd.read_csv(os.path.join(RES, "action_mix_by_task.csv"))
    lab_t = {"cabin_control": "Cabin control", "charging_planner": "Charging plan.", "driver_monitoring": "Driver monitor.",
             "media": "Media", "messaging": "Messaging", "navigation_query": "Navigation", "vehicle_diagnostics": "Diagnostics"}
    colors = ["#34495e", "#a6acaf", "#c0392b", "#f5b7b1"]
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ylab, y = [], 0
    tasks = sorted(mix.task_type.unique())
    for i, t in enumerate(tasks):
        for nm in ["Logged", "PSO"]:
            row = mix[(mix.policy == nm) & (mix.task_type == t)].iloc[0]; left = 0
            for k, a in enumerate(ACTIONS):
                ax.barh(y, row[a], left=left, color=colors[k], height=0.8, label=a.replace("_", " ") if y == 0 else None); left += row[a]
            ylab.append(f"{lab_t[t]} ({'L' if nm == 'Logged' else 'P'})"); y += 1
        y += 0.5
    pos = [j + 0.5 * (j // 2) for j in range(len(ylab))]
    ax.set_yticks(pos); ax.set_yticklabels(ylab, fontsize=6.5); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("Share of test requests")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.4, 1.0), fontsize=6.5)
    _save(fig, FIG, RES, "fig6_action_mix")


if __name__ == "__main__":
    H = os.path.dirname(os.path.abspath(__file__))
    main(os.path.join(H, "..", "results"), os.path.join(H, "..", "figures"))
