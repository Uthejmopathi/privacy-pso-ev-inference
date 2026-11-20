"""Counterfactual evaluation twin for A2P2 (in-vehicle multimodal assistant, EV).

The logged CSV only records the outcome of the action the field controller actually took. To score any NEW
execution-mode policy we re-implement the outcome equations of the synthetic generator
(datasets/A2_Uthej/gen_author2.py, function paper2) and integrate their hidden noise by Monte Carlo:
  latency  : local = work*(420|170)*(1+1.8*sigmoid((T-85)/4))*(1+0.6*cpu); offload = 2*RTT + 60*work + 80 + 30*[priv>=2];
             deferred ~ U(3,30) s; times lognormal(0,0.6) and 10 % stalls x U(2,6)
  energy   : full 9*work, int8 3.5*work, offload 1.5+0.06*(-dBm-60)+0.2*work, deferred 2*work; x(1+0.012*max(T-70,0)) x LN(0,0.35)
  temp rise: full 1.6*work, int8 0.6*work, else 0.1; x LN(0,0.2)
  quality  : 0.88 / 0.80 (-0.03 if image > 1 MP) / 0.93 / 0.90 ; exposure: offload priv*0.25*U(0.5,1)
The optimisers NEVER call this module; they only see LightGBM surrogates trained on the 2025 logs.
"""
import numpy as np

ACTIONS = ["local_full", "local_quantized", "secure_offload", "deferred"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(ctx, action_idx, e_cap, t_max, rng, K=200):
    """ctx: dict of arrays (work, img, npu, cpu, rtt, dbm, priv, deadline); action_idx: int array (n,).
    Returns dict of per-request expectations: p_dl, energy, p_e_viol, p_t_viol, quality, exposure, latency_med."""
    n = len(action_idx)
    a = action_idx
    work, T, cpu, rtt, dbm, priv, dl = (ctx[k] for k in ("work", "npu", "cpu", "rtt", "dbm", "priv", "deadline"))
    thr = 1 + 1.8 * sigmoid((T - 85) / 4)
    loc = work * np.where(a == 0, 420.0, 170.0) * thr * (1 + 0.6 * cpu)
    off = 2 * rtt + 60 * work + 80 + 30 * (priv >= 2)
    base = np.where(a <= 1, loc, off)[:, None]
    noise = rng.lognormal(0, 0.6, (n, K)) * np.where(rng.random((n, K)) < 0.10, rng.uniform(2, 6, (n, K)), 1.0)
    lat = np.where((a == 3)[:, None], rng.uniform(3000, 30000, (n, K)), base) * noise
    p_dl = (lat <= dl[:, None]).mean(1)
    e_base = np.select([a == 0, a == 1, a == 2], [work * 9.0, work * 3.5, 1.5 + (-dbm - 60) * 0.06 + 0.2 * work],
                       default=work * 2.0)
    e = (e_base * (1 + 0.012 * np.maximum(T - 70, 0)))[:, None] * rng.lognormal(0, 0.35, (n, K))
    rise = np.select([a == 0, a == 1], [work * 1.6, work * 0.6], default=0.1)[:, None] * rng.lognormal(0, 0.2, (n, K))
    q = np.select([a == 0, a == 2, a == 1], [0.88, 0.93, 0.80], default=0.9) - 0.03 * (ctx["img"] > 1) * (a == 1)
    expo = np.where(a == 2, priv * 0.25 * 0.75, 0.0)
    return {"p_dl": p_dl, "energy": e.mean(1), "p_e_viol": (e > e_cap[:, None]).mean(1),
            "p_t_viol": ((T[:, None] + rise) > t_max).mean(1), "quality": q, "exposure": expo,
            "latency_med": np.median(lat, 1)}
