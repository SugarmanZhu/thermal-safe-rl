"""Evaluation.

    python evaluate.py estimation     # RQ1: KF vs AKF in every scenario (no RL needed)
    python evaluate.py tune-pid       # grid search for the PID gains on tuning seeds
    python evaluate.py controllers    # controllers x scenarios x seeds (RQ2-RQ4), incl. trained PPO in runs/
    python evaluate.py shield-sweep   # shield energy/safety vs assumed worst-case burst length
    python evaluate.py training       # training-time safety from runs/*/seed*/train_log.csv
    python evaluate.py sensor-failure # permanent surface-sensor failure 10 min into the episode

Evaluation seeds are 0-19; PID tuning uses seeds 1000-1019.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from estimators import KalmanFilter  # noqa: E402
from thermal_env import DT, EPISODE_STEPS, SCENARIOS, T_LIMIT, ThermalSim  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
WARMUP = 150  # steps (5 min) excluded from estimation metrics

# Chart styling: reference palette, light surface. Colour follows the entity.
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLOR = {"AKF": "#2a78d6", "KF": "#eb6834", "true": INK}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 1.5,
})


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def estimation_metrics(h: dict, warmup: int = WARMUP) -> dict:
    s = slice(warmup, None)
    e = h["Tc_hat"][s] - h["Tc"][s]
    sig = h["sigma"][s]
    err2 = np.stack([h["Tc"][s] - h["Tc_hat"][s], h["Ts"][s] - h["Ts_hat"][s]], axis=-1)
    nees = np.einsum("ti,tij,tj->t", err2, np.linalg.inv(h["P2"][s]), err2)
    nu, S = h["nu"][s], h["S"][s]
    ok = ~np.isnan(nu)
    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "bias": float(np.mean(e)),
        "coverage_2sigma": float(np.mean(np.abs(e) <= 2 * sig) * 100),
        "nees": float(np.mean(nees)),  # consistent filter: about 2
        "nis": float(np.mean(nu[ok] ** 2 / S[ok])),  # consistent filter: about 1
        "mean_norm_innov": float(np.mean(nu[ok] / np.sqrt(S[ok]))),  # about 0
        "theta_end": float(h["theta"][-1]),
    }


def run_episode(scenario_name: str, augment: bool, seed: int, policy=lambda sim: 1.0) -> dict:
    sim = ThermalSim(SCENARIOS[scenario_name], KalmanFilter(augment=augment))
    sim.reset(seed)
    while not sim.done:
        sim.step(policy(sim))
    return sim.hist


# ---------------------------------------------------------------------------
# Estimation study (RQ1)
# ---------------------------------------------------------------------------

def estimation_study(n_seeds: int = 20, plot_seed: int = 0) -> pd.DataFrame:
    """Both filters, every scenario, full cooling (u = 1) so only the estimator differs."""
    rows, traces = [], {}
    for name in SCENARIOS:
        for augment in (False, True):
            fname = "AKF" if augment else "KF"
            for seed in range(n_seeds):
                h = run_episode(name, augment, seed)
                rows.append({"scenario": name, "filter": fname, "seed": seed, **estimation_metrics(h)})
                if seed == plot_seed:
                    traces[(name, fname)] = h
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "estimation.csv", index=False)
    write_estimation_table(df)
    plot_estimation(traces)
    plot_calibration(df)
    return df


def write_estimation_table(df: pd.DataFrame) -> None:
    cols = {"rmse": "RMSE (°C)", "bias": "bias (°C)", "coverage_2sigma": "2σ coverage (%)",
            "nees": "NEES (≈2)", "nis": "NIS (≈1)", "mean_norm_innov": "mean innov. (≈0)",
            "theta_end": "θ̂ at end"}
    g = df.groupby(["scenario", "filter"], sort=False)
    mean, std = g[list(cols)].mean(), g[list(cols)].std()
    out = pd.DataFrame(index=mean.index)
    for c, label in cols.items():
        out[label] = [f"{m:.2f} ± {s:.2f}" if c != "coverage_2sigma" else f"{m:.0f} ± {s:.0f}"
                      for m, s in zip(mean[c], std[c])]
    out.loc[out.index.get_level_values("filter") == "KF", "θ̂ at end"] = "–"
    out = out.reset_index()
    header = ("Estimation study: full cooling (u = 1), 20 seeds per row, mean ± std over seeds, "
              f"metrics after a {WARMUP * DT / 60:.0f}-min warm-up. "
              "Generated by `python evaluate.py estimation`.\n\n")
    (RESULTS / "estimation.md").write_text(header + out.to_markdown(index=False) + "\n")


def plot_estimation(traces: dict) -> None:
    cases = [("heat_mismatch", "Heat mismatch (R_int ×1.3): visible, AKF corrects it"),
             ("resistance_mismatch", "Resistance mismatch (Rc ×1.2): invisible from the surface")]
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.6, 1]})
    for col, (name, title) in enumerate(cases):
        akf, kf = traces[(name, "AKF")], traces[(name, "KF")]
        t = np.arange(len(akf["Tc"])) * DT / 60
        ax = axes[0, col]
        ax.plot(t, akf["Tc"], color=COLOR["true"], lw=1.5, label="true core (simulation only)")
        for f, h in (("KF", kf), ("AKF", akf)):
            ax.plot(t, h["Tc_hat"], color=COLOR[f], lw=1.5, label=f"{f} estimate")
        ax.axhline(T_LIMIT, color=MUTED, lw=1, ls="--")
        ax.text(t[-1], T_LIMIT, "45 °C limit", color=INK2, ha="right", va="bottom", fontsize=8)
        ax.set_title(title, loc="left")
        ax.set_ylim(24, 47)
        ax.set_ylabel("core temperature (°C)")

        ax = axes[1, col]
        ax.fill_between(t, -2 * akf["sigma"], 2 * akf["sigma"], color=COLOR["AKF"], alpha=0.18,
                        lw=0, label="AKF ±2σ band (KF's is similar)")
        for f, h in (("KF", kf), ("AKF", akf)):
            ax.plot(t, h["Tc_hat"] - h["Tc"], color=COLOR[f], lw=1.5, label=f"{f} error")
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_ylim(-3, 1.5)
        ax.set_ylabel("estimate − true (°C)")

        ax = axes[2, col]
        true_scale = SCENARIOS[name].R_int_scale * np.ones_like(t)
        ax.plot(t, true_scale, color=COLOR["true"], lw=1.5, ls="--", label="true heat scale")
        ax.plot(t, akf["theta"], color=COLOR["AKF"], lw=1.5, label="AKF θ̂")
        ax.set_ylabel("heat scale θ")
        ax.set_xlabel("time (min)")
        ax.set_ylim(0.8, 1.5)

    for row, loc in enumerate(("lower right", "lower left", "lower right")):
        axes[row, 0].legend(loc=loc, fontsize=8, ncol=1 if row == 0 else 3)
    for ax in axes.flat:
        ax.axvspan(0, WARMUP * DT / 60, color=GRID, alpha=0.5, lw=0, zorder=0)
    axes[2, 1].text(WARMUP * DT / 120, 1.45, "warm-up\n(not scored)", color=INK2, fontsize=8,
                    ha="center", va="top")
    fig.suptitle("What a surface sensor can and cannot tell you (one episode, seed 0, full cooling)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS / "estimation.png", dpi=150)
    plt.close(fig)


def plot_calibration(df: pd.DataFrame) -> None:
    names = list(SCENARIOS)
    g = df.groupby(["scenario", "filter"])
    panels = [("rmse", "RMSE (°C)", None, False),
              ("coverage_2sigma", "2σ coverage (%), ideal ≈95", 95.0, False),
              ("nis", "NIS, log scale, ideal ≈1", 1.0, True)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    y = np.arange(len(names))
    bar_h = 0.36
    for ax, (metric, label, ref, log) in zip(axes, panels):
        for i, f in enumerate(("AKF", "KF")):
            vals = [g[metric].mean()[(n, f)] for n in names]
            ax.barh(y + (i - 0.5) * (bar_h + 0.04), vals, height=bar_h, color=COLOR[f],
                    edgecolor=SURFACE, linewidth=1, label=f, zorder=2)
        if ref is not None:
            ax.axvline(ref, color=INK2, lw=1, ls="--", zorder=3)
            ax.text(ref, -0.8, f" target {ref:g}", color=INK2, fontsize=8, va="center")
        if log:
            ax.set_xscale("log")
            ax.set_xticks([0.5, 1, 2, 5, 10])
            ax.set_xticklabels(["0.5", "1", "2", "5", "10"])
        ax.set_title(label, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([n.replace("_", " ") for n in names])
    axes[0].set_ylim(len(names) - 0.5, -1.1)  # top-to-bottom, room for target labels
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Estimator accuracy and calibration, mean of 20 seeds (full table: results/estimation.md)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS / "calibration.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Controller comparison (RQ2-RQ4)
# ---------------------------------------------------------------------------

# Scatter plots compare every pair of colours, and only three palette hues stay
# distinguishable that way. So colour marks the family (orange = PID, aqua = model-based
# safety, blue = PPO, grey = references) and marker shape/fill marks the member.
ORANGE, AQUA, BLUE = "#eb6834", "#1baf7a", "#2a78d6"
CTRL_STYLE = {
    "always_on": dict(color=INK2, marker="s", mfc="none", label="always on (floor)"),
    "oracle_pid": dict(color=INK2, marker="D", mfc="none", label="oracle PID (sees true core)"),
    "pid": dict(color=ORANGE, marker="o", label="PID"),
    "pid40_shield": dict(color=ORANGE, marker="p", label="PID aiming at 40 °C + shield"),
    "shield_only_nobackup": dict(color=AQUA, marker="^", mfc="none", label="shield only, v1 (no backup check)"),
    "shield_only": dict(color=AQUA, marker="^", label="shield only (backup check)"),
    "mpc": dict(color=AQUA, marker="v", label="MPC (backup check)"),
    "mpc_preview": dict(color=AQUA, marker=">", label="MPC + perfect 60 s load preview"),
    "ppo_dr": dict(color=BLUE, marker="o", mfc="none", label="PPO (DR), no shield"),
    "ppo_dr+shield": dict(color=BLUE, marker="*", label="PPO (DR) + shield"),
    "ppo_dr_shieldtrain+shield": dict(color=BLUE, marker="P", label="PPO trained with shield (penalty 0.1) + shield"),
    "ppo_dr_shieldtrain_p1": dict(color=BLUE, marker="d", mfc="none", label="PPO trained with shield (penalty 1.0), no shield"),
    "ppo_dr_shieldtrain_p1+shield": dict(color=BLUE, marker="d", label="PPO trained with shield (penalty 1.0) + shield"),
    "ppo_dr_raw": dict(color=BLUE, marker="s", mfc="none", label="PPO, raw surface readings only, no shield"),
    "ppo_dr_raw+shield": dict(color=BLUE, marker="s", label="PPO, raw surface readings only + shield"),
    "ppo": dict(color=BLUE, marker="X", mfc="none", label="PPO (no DR), no shield"),
    "ppo+shield": dict(color=BLUE, marker="X", label="PPO (no DR) + shield"),
}
OVERVIEW_SET = ["always_on", "pid", "oracle_pid", "pid40_shield", "shield_only", "mpc", "mpc_preview",
                "ppo_dr", "ppo_dr+shield", "ppo_dr_raw+shield", "ppo_dr_shieldtrain_p1",
                "ppo_dr_shieldtrain_p1+shield"]
TRACE_SET = ["always_on", "pid", "pid40_shield", "shield_only", "mpc_preview", "ppo_dr", "ppo_dr+shield"]
TRACE_SCENARIOS = ("nominal", "resistance_mismatch")
TRACE_SEED = 0
RUNS = Path(__file__).resolve().parent / "runs"


def style(name: str) -> dict:
    return CTRL_STYLE.get(name, dict(color=BLUE, marker="h", label=name))


def discover_runs(runs_dir: Path) -> dict[str, dict[int, Path]]:
    """{variant: {train_seed: final_dir}} for every finished training run."""
    found: dict[str, dict[int, Path]] = {}
    for cfg in sorted(Path(runs_dir).glob("*/seed*/final/config.json")):
        final = cfg.parent
        if (final / "model.zip").exists():
            variant, seed = final.parent.parent.name, int(final.parent.name.removeprefix("seed"))
            found.setdefault(variant, {})[seed] = final
    return found


_PPO_CACHE: dict = {}


def _make(ctrl_name: str, final_dir):
    from controllers import PPOController, make_controller

    if final_dir is None:
        return make_controller(ctrl_name)
    key = (str(final_dir), ctrl_name.endswith("+shield"))
    if key not in _PPO_CACHE:  # loading a policy is slow; reuse it within a worker
        _PPO_CACHE[key] = PPOController(final_dir, shield=key[1])
    return _PPO_CACHE[key]


def _log_energy_axis(ax) -> None:
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([10, 30, 100, 300, 1000, 3000]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())


def control_metrics(h: dict) -> dict:
    Tc, u = h["Tc"][1:], h["u"][:-1]
    viol = np.maximum(0.0, Tc - T_LIMIT)
    return {
        "energy": float(np.sum(u**3) * DT),  # pump-seconds at full power
        "viol_steps_pct": float(np.mean(viol > 0) * 100),
        "any_violation": float(np.any(viol > 0)),
        "deg_s": float(np.sum(viol) * DT),  # degree-seconds above 45 degC
        "max_Tc": float(np.nanmax(h["Tc"])),
        "mean_u": float(np.mean(u)),
    }


def run_controller_episode(job: tuple) -> tuple[dict, dict | None]:
    """job = (controller, scenario, seed, estimator, pid_gains or None[, final_dir or None]).
    Top-level so it can run in a process pool."""
    from controllers import PID

    ctrl_name, scen, seed, estimator, gains = job[:5]
    final_dir = job[5] if len(job) > 5 else None
    ctrl = PID(**gains) if gains is not None else _make(ctrl_name, final_dir)
    sim = ThermalSim(SCENARIOS[scen], KalmanFilter(augment=(estimator == "akf")))
    sim.reset(seed)
    ctrl.reset()
    while not sim.done:
        sim.step(ctrl.act(sim))
    h = sim.hist
    est = estimation_metrics(h)
    train_seed = getattr(ctrl, "train_seed", -1)
    row = {"controller": ctrl_name, "train_seed": train_seed, "scenario": scen, "seed": seed,
           "estimator": estimator, **control_metrics(h), **ctrl.stats(),
           "est_rmse": est["rmse"], "est_coverage": est["coverage_2sigma"], "est_nis": est["nis"]}
    keep = seed == TRACE_SEED and scen in TRACE_SCENARIOS and gains is None and train_seed <= 0
    return row, (h if keep else None)


def _pool_map(fn, jobs, n_jobs):
    if n_jobs <= 1:
        return [fn(j) for j in jobs]
    import multiprocessing as mp

    with mp.get_context("spawn" if sys.platform == "darwin" else "fork").Pool(n_jobs) as pool:
        return pool.map(fn, jobs, chunksize=max(1, len(jobs) // (8 * n_jobs)))


def tune_pid(n_jobs: int, seeds=range(1000, 1020)) -> None:
    """Grid search on separate tuning seeds, nominal scenario, AKF estimate.
    Objective = energy + 10 * degree-seconds above 45 degC (the RL reward's weights)."""
    grid = [dict(kp=kp, ki=ki, kd=kd) for kp in (0.1, 0.2, 0.3, 0.5, 1.0)
            for ki in (0.0, 0.001, 0.003, 0.01) for kd in (0.0, 10.0, 30.0)]
    jobs = [("pid", "nominal", s, "akf", g) for g in grid for s in seeds]
    rows = [r for r, _ in _pool_map(run_controller_episode, jobs, n_jobs)]
    df = pd.DataFrame(rows)
    df[["kp", "ki", "kd"]] = [[g["kp"], g["ki"], g["kd"]] for g in grid for _ in seeds]
    res = df.groupby(["kp", "ki", "kd"])[["energy", "deg_s", "max_Tc"]].mean()
    res["objective"] = res["energy"] + 10 * res["deg_s"]
    print(res.sort_values("objective").head(10).to_string())


def default_controllers(runs: dict) -> list[str]:
    from controllers import BASELINES

    rl = [n for v in runs for n in (v, f"{v}+shield")]
    return BASELINES + rl


def controller_study(names: list[str], n_seeds: int, estimator: str, n_jobs: int,
                     runs: dict | None = None) -> pd.DataFrame:
    runs = runs or {}
    jobs = []
    for c in names:
        variant = c.removesuffix("+shield")
        if variant in runs:  # one controller per trained seed
            jobs += [(c, s, seed, estimator, None, final) for final in runs[variant].values()
                     for s in SCENARIOS for seed in range(n_seeds)]
        elif variant.startswith("ppo"):
            print(f"skipping {c}: no finished runs found")
        else:
            jobs += [(c, s, seed, estimator, None, None) for s in SCENARIOS for seed in range(n_seeds)]
    out = _pool_map(run_controller_episode, jobs, n_jobs)
    df = pd.DataFrame([r for r, _ in out])
    names = [n for n in names if n in set(df.controller)]
    traces = {(r["controller"], r["scenario"]): h for r, h in out if h is not None}
    suffix = "" if estimator == "akf" else f"_{estimator}"
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f"controllers{suffix}.csv", index=False)
    write_controller_tables(df, names, suffix, estimator, n_seeds)
    if estimator == "akf":
        for scen in TRACE_SCENARIOS:
            plot_trajectories(traces, [n for n in TRACE_SET if n in names], scen)
        plot_overview(df, [n for n in OVERVIEW_SET if n in names])
        if {"ppo_dr", "ppo_dr_nosigma", "ppo_dr_raw"} <= set(names):
            plot_rq3(df)
    return df


def _fmt(m: float, s: float, digits: int = 1) -> str:
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean and std per (scenario, controller). For trained policies the std is across
    training seeds (each first averaged over its evaluation episodes); for the rest it is
    across episodes."""
    df = df.copy()
    if "train_seed" not in df:
        df["train_seed"] = -1
    df["train_seed"] = df["train_seed"].fillna(-1)
    rl = df.train_seed >= 0
    per_seed = df[rl].groupby(["scenario", "controller", "train_seed"]).mean(numeric_only=True).reset_index()
    keys = ["scenario", "controller"]
    parts = [d.groupby(keys) for d in (df[~rl], per_seed) if len(d)]
    mean = pd.concat([g.mean(numeric_only=True) for g in parts])
    std = pd.concat([g.std(numeric_only=True) for g in parts])
    return mean, std


def write_controller_tables(df: pd.DataFrame, names, suffix: str, estimator: str, n_seeds: int) -> None:
    order = {n: i for i, n in enumerate(names)}
    mean, std = aggregate(df)
    n_train = df.loc[df.get("train_seed", pd.Series(-1, index=df.index)) >= 0, "train_seed"].nunique()
    rl_note = (f" For PPO rows, ± is the std across {n_train} training seeds, each averaged over its "
               "evaluation episodes." if n_train else "")
    lines = [f"Controller comparison: {n_seeds} evaluation seeds per cell, estimator {estimator.upper()}, "
             "mean ± std over episodes." + rl_note + " Generated by `python evaluate.py controllers`.\n",
             "Energy is pump-seconds at full power (always on = 1800). Degree-seconds = "
             "∫ max(0, Tc − 45 °C) dt.\n"]
    for metric, title, digits in [("energy", "Cooling energy (pump-seconds)", 0),
                                  ("deg_s", "Degree-seconds above 45 °C", 1),
                                  ("any_violation", "Episodes with any violation (%)", 0)]:
        piv = mean[metric].unstack("scenario")[list(SCENARIOS)]
        if metric == "any_violation":
            piv = piv * 100
        piv = piv.loc[sorted(piv.index, key=lambda n: order.get(n, 999))]
        lines += [f"\n### {title}\n", piv.round(digits).to_markdown() + "\n"]
    full = []
    for (scen, ctrl), m in mean.iterrows():
        s = std.loc[(scen, ctrl)]
        full.append({"scenario": scen, "controller": ctrl,
                     "energy": _fmt(m.energy, s.energy, 0),
                     "% steps > 45 °C": _fmt(m.viol_steps_pct, s.viol_steps_pct, 2),
                     "P(any violation) %": f"{m.any_violation * 100:.0f}",
                     "degree-s > 45 °C": _fmt(m.deg_s, s.deg_s, 1),
                     "max Tc (°C)": _fmt(m.max_Tc, s.max_Tc, 1),
                     "shield active (% steps)": f"{m.shield_rate:.1f}" if "shield_rate" in m and
                     not np.isnan(m.shield_rate) else "–"})
    full = pd.DataFrame(full)
    full["_o"] = full["controller"].map(lambda n: order.get(n, 999))
    full = full.sort_values(["scenario", "_o"], key=lambda c: c.map(
        {n: i for i, n in enumerate(SCENARIOS)}) if c.name == "scenario" else c).drop(columns="_o")
    lines += ["\n### Full table\n", full.to_markdown(index=False) + "\n"]
    (RESULTS / f"controllers{suffix}.md").write_text("\n".join(lines))


def plot_trajectories(traces: dict, names, scen: str) -> None:
    names = [n for n in names if (n, scen) in traces]
    any_h = traces[(names[0], scen)]
    t = np.arange(len(any_h["Tc"])) * DT / 60
    fig = plt.figure(figsize=(11, 1.25 + 1.55 * len(names)))
    gs = fig.add_gridspec(len(names) + 1, 2, height_ratios=[0.8] + [1] * len(names), hspace=0.55)
    ax = fig.add_subplot(gs[0, :])
    ax.step(t[:-1], any_h["I"][:-1], where="post", color=INK2, lw=1.2)
    ax.set_ylabel("current (A)")
    ax.set_title("Drive cycle (same for every controller)", loc="left")
    for row, n in enumerate(names, start=1):
        h = traces[(n, scen)]
        m = control_metrics(h)
        a1 = fig.add_subplot(gs[row, 0], sharex=ax)
        a1.plot(t, h["Tc"], color=INK, lw=1.3)
        a1.axhline(T_LIMIT, color=MUTED, lw=1, ls="--")
        a1.set_ylim(24, 48)
        a1.set_ylabel("core (°C)")
        a1.set_title(f"{style(n)['label']}: max {m['max_Tc']:.1f} °C, "
                     f"{m['deg_s']:.1f} degree-s above 45 °C", loc="left", fontsize=9)
        a2 = fig.add_subplot(gs[row, 1], sharex=ax)
        a2.fill_between(t[:-1], 0, h["u"][:-1], step="post", color="#2a78d6", alpha=0.25, lw=0)
        a2.step(t[:-1], h["u"][:-1], where="post", color="#2a78d6", lw=1.2)
        a2.set_ylim(0, 1.05)
        a2.set_ylabel("cooling u")
        a2.set_title(f"energy {m['energy']:.0f} pump-s", loc="left", fontsize=9)
        if row == len(names):
            a1.set_xlabel("time (min)")
            a2.set_xlabel("time (min)")
    first = fig.axes[1]
    first.text(t[-1], T_LIMIT, "45 °C", color=INK2, ha="right", va="bottom", fontsize=8)
    pretty = scen.replace("_", " ")
    fig.suptitle(f"One episode per controller, {pretty} scenario (seed {TRACE_SEED}): "
                 "true core temperature (left) and cooling command (right)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.05)
    fig.savefig(RESULTS / f"trajectory_{scen}.png", dpi=150)
    plt.close(fig)


# Sequential ramps (one hue, light to dark). Blue is the reference palette's sequential ramp;
# the second sequential context takes the next categorical hue (orange).
BLUE_RAMP = ["#e6f0fc", "#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
ORANGE_RAMP = ["#fbf1ec", "#f9d2c0", "#f4a888", "#eb6834", "#c9501f", "#9a3b16", "#6b280d"]


def _ink_on(rgb) -> str:
    """Dark or light text, whichever reads better on a cell of this colour."""
    r, g, b = rgb[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if lum > 0.5 else "#ffffff"


def plot_overview(df: pd.DataFrame, names) -> None:
    """Two heatmaps with one row per controller and one column per scenario: cooling energy
    and how often the core exceeded 45 degC. Every value is printed in its own cell, so
    nothing overlaps."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

    mean = aggregate(df)[0]
    scens = list(SCENARIOS)
    E = np.array([[mean.loc[(s, n), "energy"] for s in scens] for n in names])
    V = np.array([[100 * mean.loc[(s, n), "any_violation"] for s in scens] for n in names])
    panels = [
        (E, LinearSegmentedColormap.from_list("energy", BLUE_RAMP), LogNorm(10, 1800), "{:.0f}",
         "Cooling energy, pump-s (lighter = cheaper)"),
        (V, LinearSegmentedColormap.from_list("viol", ORANGE_RAMP), Normalize(0, 100), "{:.0f}%",
         "Episodes where the core exceeded 45 °C (lighter = safer)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.43 * len(names) + 2.2), sharey=True)
    col_labels = [s.replace("_", "\n") for s in scens]
    for ax, (M, cmap, norm, fmt, title) in zip(axes, panels):
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8.5,
                        color=_ink_on(cmap(norm(M[i, j]))))
        ax.set_xticks(range(len(scens)))
        ax.set_xticklabels(col_labels, fontsize=8.5, color=INK2)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
        ax.set_xticks(np.arange(-0.5, len(scens), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)  # 2 px gap between cells
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(title, loc="left", pad=34)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels([style(n)["label"] for n in names], fontsize=9, color=INK)
    axes[0].tick_params(axis="y", length=0)
    for lab, n in zip(axes[0].get_yticklabels(), names):
        if n == "ppo_dr+shield":
            lab.set_fontweight("bold")
    n_train = df.loc[df.get("train_seed", pd.Series(-1, index=df.index)) >= 0, "train_seed"].nunique()
    fig.suptitle(f"All controllers at a glance: mean of {df.seed.nunique()} evaluation episodes per cell"
                 + (f" (PPO: also over {n_train} training seeds)" if n_train else ""),
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.01, "Full tables with spread: results/controllers.md", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(RESULTS / "overview.png", dpi=150)
    plt.close(fig)


def plot_rq3(df: pd.DataFrame) -> None:
    """RQ3: does giving the policy the estimator's output (and its uncertainty) help?
    Policies without the shield, so the comparison is about the policy alone. Bars are means
    over training seeds; whiskers span the lowest to highest training seed (3 seeds is too
    few for a meaningful standard deviation)."""
    modes = [("ppo_dr_raw", "raw surface readings only"), ("ppo_dr_nosigma", "estimate, no σ"),
             ("ppo_dr", "estimate + σ (full)")]
    ramp = ["#86b6ef", "#2a78d6", "#104281"]  # ordinal blue ramp: more estimator information
    per_seed = (df[df.controller.isin([m for m, _ in modes])]
                .groupby(["scenario", "controller", "train_seed"]).mean(numeric_only=True))
    g = per_seed.groupby(["scenario", "controller"])
    mean, lo, hi = g.mean(), g.min(), g.max()
    scens = list(SCENARIOS)
    y = np.arange(len(scens))
    h = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
    for ax, (metric, title) in zip(axes, [("energy", "Cooling energy (pump-s)"),
                                          ("deg_s", "Degree-s above 45 °C (lower is safer)")]):
        for i, ((name, label), col) in enumerate(zip(modes, ramp)):
            vals = np.array([mean.loc[(s, name), metric] for s in scens])
            lows = np.array([lo.loc[(s, name), metric] for s in scens])
            highs = np.array([hi.loc[(s, name), metric] for s in scens])
            yy = y + (i - 1) * (h + 0.03)
            ax.barh(yy, vals, height=h, color=col, edgecolor=SURFACE, lw=1, label=label, zorder=2)
            ax.hlines(yy, lows, highs, color=INK2, lw=0.9, zorder=3)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(left=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([s.replace("_", " ") for s in scens])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, title="policy observes",
               title_fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    n_train = df.loc[df.controller == "ppo_dr", "train_seed"].nunique()
    fig.suptitle(f"RQ3: what the policy observes (PPO with DR, no shield). Bars: mean of {n_train} "
                 "training seeds; whiskers: lowest to highest seed", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(RESULTS / "rq3_ablation.png", dpi=150)
    plt.close(fig)


def training_study(runs_dir: Path) -> pd.DataFrame | None:
    """Training-time safety: how much do policies violate the limit while they learn?"""
    rows = []
    for log in sorted(Path(runs_dir).glob("*/seed*/train_log.csv")):
        d = pd.read_csv(log)
        d["variant"], d["train_seed"] = log.parent.parent.name, int(log.parent.name.removeprefix("seed"))
        rows.append(d)
    if not rows:
        print(f"no training logs under {runs_dir}")
        return None
    df = pd.concat(rows, ignore_index=True)
    per_run = df.groupby(["variant", "train_seed"]).agg(
        episodes=("deg_s", "size"), total_deg_s=("deg_s", "sum"),
        viol_episodes_pct=("deg_s", lambda x: 100 * (x > 0).mean()),
        last_energy=("energy", lambda x: x.tail(max(1, len(x) // 10)).mean()))
    g = per_run.groupby("variant")
    tab = pd.DataFrame({
        "training seeds": g.size(),
        "episodes per run": g.episodes.mean().round(0),
        "degree-s above 45 °C, whole training": [_fmt(m, s, 0) for m, s in zip(g.total_deg_s.mean(), g.total_deg_s.std().fillna(0))],
        "episodes with a violation (%)": [_fmt(m, s, 1) for m, s in zip(g.viol_episodes_pct.mean(), g.viol_episodes_pct.std().fillna(0))],
        "energy per episode, last 10 % of training": g.last_energy.mean().round(0),
    })
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "training_safety.md").write_text(
        "Training-time safety (unsafe exploration). Mean ± std over training seeds. Training episodes use "
        "randomised conditions (extra sensor noise, sensor outages and, with DR, randomised cell parameters). Generated by `python evaluate.py training`.\n\n"
        + tab.to_markdown() + "\n")
    plot_training(df)
    return tab


def plot_training(df: pd.DataFrame) -> None:
    known = ("ppo_dr", "ppo", "ppo_dr_shieldtrain", "ppo_dr_shieldtrain_p1", "ppo_dr_nosigma", "ppo_dr_raw")
    order = [v for v in known if v in set(df.variant)] + sorted(set(df.variant) - set(known))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for v, col in zip(order, colors):
        d = df[df.variant == v]
        grid = np.linspace(0, d.timesteps.max(), 60)
        cum = [np.interp(grid, r.timesteps, r.deg_s.cumsum()) for _, r in d.groupby("train_seed")]
        en = [np.interp(grid, r.timesteps, r.energy.rolling(20, min_periods=1).mean()) for _, r in d.groupby("train_seed")]
        for ax, curves in zip(axes, (cum, en)):
            c = np.array(curves)
            ax.plot(grid / 1e3, c.mean(0), color=col, lw=1.8, label=v)
            if len(c) > 1:
                ax.fill_between(grid / 1e3, c.min(0), c.max(0), color=col, alpha=0.15, lw=0)
    axes[0].set_title("Cumulative degree-s above 45 °C during training", loc="left")
    axes[0].set_ylabel("degree-s (lower is safer)")
    axes[1].set_title("Energy per training episode (rolling mean of 20)", loc="left")
    axes[1].set_ylabel("pump-s")
    for ax in axes:
        ax.set_xlabel("training steps (thousands)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Learning safely? Mean over training seeds, band = min to max",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS / "training_safety.png", dpi=150)
    plt.close(fig)


FAIL_S = 600.0  # the surface sensor dies 10 min into the episode and never comes back
FAIL_SCENARIOS = ("nominal", "heat_mismatch")
FAIL_CONTROLLERS = ["always_on", "pid", "pid40_shield", "shield_only", "ppo_dr", "ppo_dr+shield"]
FAIL_PLOT = ["pid40_shield", "shield_only", "ppo_dr+shield"]


def run_failure_episode(job: tuple):
    """job = (controller, scenario, seed, final_dir or None, failed). Returns (row, time series)."""
    from dataclasses import replace

    ctrl_name, scen, seed, final_dir, failed = job
    sc = replace(SCENARIOS[scen], sensor_fail_s=FAIL_S if failed else None)
    ctrl = _make(ctrl_name, final_dir)
    sim = ThermalSim(sc, KalmanFilter(augment=True))
    sim.reset(seed)
    ctrl.reset()
    while not sim.done:
        sim.step(ctrl.act(sim))
    h = sim.hist
    k0 = int(round(FAIL_S / DT))
    after_T = np.maximum(0.0, h["Tc"][k0 + 1:] - T_LIMIT)
    row = {"controller": ctrl_name, "scenario": scen, "seed": seed, "failed": failed,
           "train_seed": getattr(ctrl, "train_seed", -1), **control_metrics(h),
           "energy_after": float(np.sum(h["u"][k0:-1] ** 3) * DT),
           "deg_s_after": float(np.sum(after_T) * DT),
           "viol_after": float(np.any(after_T > 0))}
    series = {"u": h["u"][:-1].copy(), "sigma": h["sigma"].copy(),
              "abs_err": np.abs(h["Tc"] - h["Tc_hat"])}
    return row, series


def sensor_failure_study(n_seeds: int, n_jobs: int, runs: dict) -> pd.DataFrame:
    jobs = []
    for c in FAIL_CONTROLLERS:
        variant = c.removesuffix("+shield")
        finals = list(runs[variant].values()) if variant in runs else ([None] if not c.startswith("ppo") else [])
        jobs += [(c, s, seed, f, failed) for f in finals for s in FAIL_SCENARIOS
                 for seed in range(n_seeds) for failed in (False, True)]
    out = _pool_map(run_failure_episode, jobs, n_jobs)
    df = pd.DataFrame([r for r, _ in out])
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "sensor_failure.csv", index=False)
    names = [c for c in FAIL_CONTROLLERS if c in set(df.controller)]
    g = df.groupby(["scenario", "controller", "failed"])
    E, Ea = g.energy_after.mean(), g.energy_after.mean()
    V, D = g.viol_after.mean() * 100, g.deg_s_after.mean()
    rows = []
    for scen in FAIL_SCENARIOS:
        for c in names:
            e0, e1 = Ea.loc[(scen, c, False)], Ea.loc[(scen, c, True)]
            rows.append({"scenario": scen.replace("_", " "), "controller": style(c)["label"],
                         "energy after 10 min, sensor OK": f"{e0:.0f}",
                         "energy after 10 min, sensor dead": f"{e1:.0f}",
                         "change": f"{(e1 - e0) / max(e0, 1e-9) * 100:+.0f} %" if e0 > 0 else "–",
                         "episodes > 45 °C after 10 min, sensor OK": f"{V.loc[(scen, c, False)]:.0f} %",
                         "episodes > 45 °C after 10 min, sensor dead": f"{V.loc[(scen, c, True)]:.0f} %"})
    text = (f"Permanent sensor failure: the surface sensor dies {FAIL_S / 60:.0f} min into the 30-min "
            f"episode and never comes back. {n_seeds} evaluation seeds per cell (PPO: also over its "
            "training seeds), AKF. Energy and violations are counted after the failure time, and "
            "compared with the same episodes with a working sensor. Generated by "
            "`python evaluate.py sensor-failure`.\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n")
    (RESULTS / "sensor_failure.md").write_text(text)
    plot_sensor_failure(out)
    return df


def plot_sensor_failure(out) -> None:
    t = np.arange(EPISODE_STEPS) * DT / 60
    fig, axes = plt.subplots(2, len(FAIL_SCENARIOS), figsize=(12, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1]})
    color = {"pid40_shield": ORANGE, "shield_only": AQUA, "ppo_dr+shield": BLUE}
    k0 = int(round(FAIL_S / DT))
    for col, scen in enumerate(FAIL_SCENARIOS):
        ax = axes[0, col]
        for c in FAIL_PLOT:
            for failed, ls in ((False, ":"), (True, "-")):
                U = [s["u"] for r, s in out if r["controller"] == c and r["scenario"] == scen and r["failed"] == failed]
                if U:
                    mu = pd.Series(np.mean(U, axis=0)).rolling(15, min_periods=1, center=True).mean()
                    ax.plot(t, mu, color=color[c], ls=ls, lw=1.8 if failed else 1.3)
        ax.axhline(1.0, color=INK2, lw=1, ls="--")
        ax.text(t[-1], 1.0, "always on ", color=INK2, fontsize=8, va="bottom", ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("mean cooling u")
        title = scen.replace("_", " ") + ": cooling command"
        if scen == "heat_mismatch":
            title += " (all three overlap)"
        ax.set_title(title, loc="left", fontsize=9.5)
        ax2 = axes[1, col]
        S = [s for r, s in out if r["controller"] == "shield_only" and r["scenario"] == scen and r["failed"]]
        ax2.plot(t, np.mean([s["sigma"][:-1] for s in S], axis=0), color=INK, lw=1.8)
        ax2.plot(t, np.mean([s["abs_err"][:-1] for s in S], axis=0), color=INK2, lw=1.4, ls="--")
        ax2.set_ylim(0, 1.3)
        ax2.set_ylabel("°C")
        ax2.set_xlabel("time (min)")
        ax2.set_title("core estimate: reported σ vs actual error", loc="left", fontsize=9.5)
        for a in (ax, ax2):
            a.axvspan(t[k0], t[-1], color=GRID, alpha=0.45, lw=0, zorder=0)
        ax.text(t[k0] + 0.3, 1.04, "sensor dead", color=INK2, fontsize=8, va="bottom")
    ctrl_handles = [plt.Line2D([], [], color=color[c], lw=2, label=style(c)["label"]) for c in FAIL_PLOT]
    ls_handles = [plt.Line2D([], [], color=INK2, lw=1.8, ls="-", label="sensor dead after 10 min"),
                  plt.Line2D([], [], color=INK2, lw=1.3, ls=":", label="sensor working (same episodes)")]
    err_handles = [plt.Line2D([], [], color=INK, lw=1.8, label="σ the filter reports (shield only)"),
                   plt.Line2D([], [], color=INK2, lw=1.4, ls="--", label="actual error |true − estimate|")]
    fig.legend(handles=ctrl_handles + ls_handles + err_handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Surface sensor dead from 10 min on (mean of all episodes, 30 s smoothing)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(RESULTS / "sensor_failure.png", dpi=150)
    plt.close(fig)


SWEEP_SCENARIOS = ("nominal", "heat_mismatch", "ageing", "resistance_mismatch")
SWEEP_BURSTS_S = (0, 30, 60, 90, 120, 180)


def run_sweep_episode(job: tuple) -> dict:
    """job = (burst_s, scenario, seed); burst_s = 0 means no backup check."""
    from controllers import Shield, ShieldOnly

    burst_s, scen, seed = job
    ctrl = ShieldOnly(Shield(backup=burst_s > 0, burst_steps=max(1, int(round(burst_s / DT)))))
    sim = ThermalSim(SCENARIOS[scen], KalmanFilter(augment=True))
    sim.reset(seed)
    ctrl.reset()
    while not sim.done:
        sim.step(ctrl.act(sim))
    return {"burst_s": burst_s, "scenario": scen, "seed": seed, **control_metrics(sim.hist)}


def shield_sweep(n_seeds: int, n_jobs: int) -> pd.DataFrame:
    """Shield-only with different worst-case burst assumptions: the price of robustness."""
    jobs = [(b, s, seed) for b in SWEEP_BURSTS_S for s in SWEEP_SCENARIOS for seed in range(n_seeds)]
    df = pd.DataFrame(_pool_map(run_sweep_episode, jobs, n_jobs))
    ref_jobs = [(c, s, seed, "akf", None, None) for c in ("always_on", "pid")
                for s in SWEEP_SCENARIOS for seed in range(n_seeds)]
    ref = pd.DataFrame([r for r, _ in _pool_map(run_controller_episode, ref_jobs, n_jobs)])
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "shield_tradeoff.csv", index=False)
    cols = list(SWEEP_SCENARIOS)
    parts = [f"Shield-only with a worst-case burst of the given length (25 A, full cooling) in the "
             f"backup check; 0 s = no backup check (the v1 shield). {n_seeds} evaluation seeds per "
             "cell, AKF, means. Rows: assumed burst length (s). Generated by "
             "`python evaluate.py shield-sweep`.\n"]
    for metric, title, scale, digits in [("energy", "Cooling energy (pump-seconds)", 1, 0),
                                         ("deg_s", "Degree-seconds above 45 °C", 1, 1),
                                         ("any_violation", "Episodes with any violation (%)", 100, 0)]:
        t1 = (df.groupby(["burst_s", "scenario"])[metric].mean().unstack()[cols] * scale).round(digits)
        t2 = (ref.groupby(["controller", "scenario"])[metric].mean().unstack()[cols] * scale).round(digits)
        t1.index = [f"shield, {b:.0f} s" for b in t1.index]
        t2.index = [CTRL_STYLE[c]["label"] for c in t2.index]
        parts += [f"\n### {title}\n", pd.concat([t1, t2]).to_markdown() + "\n"]
    text = "\n".join(parts)
    (RESULTS / "shield_tradeoff.md").write_text(text)
    plot_shield_tradeoff(df, ref)
    return df


def plot_shield_tradeoff(df: pd.DataFrame, ref: pd.DataFrame) -> None:
    m = df.groupby(["scenario", "burst_s"])[["energy", "deg_s"]].mean()
    r = ref.groupby(["scenario", "controller"])[["energy", "deg_s"]].mean()
    fig, axes = plt.subplots(1, len(SWEEP_SCENARIOS), figsize=(12, 3.9), sharey=True)
    for ax, scen in zip(axes, SWEEP_SCENARIOS):
        d = m.loc[scen]
        ax.plot(d.energy, d.deg_s, color=CTRL_STYLE["shield_only"]["color"], lw=1.5, marker="o",
                ms=6, mec=SURFACE, zorder=3)
        for b, row in d.iterrows():
            if b in (0, 90, 180):  # label a few points; the legend explains the rest
                ax.annotate(f"{b:.0f} s", (row.energy, row.deg_s), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=8, color=INK2)
        for c in ("always_on", "pid"):
            st = CTRL_STYLE[c]
            ax.plot(r.loc[(scen, c)].energy, r.loc[(scen, c)].deg_s, ls="none", marker=st["marker"],
                    ms=8, color=st["color"], mfc=st.get("mfc", st["color"]), mec=st["color"]
                    if st.get("mfc") else SURFACE, mew=1.5 if st.get("mfc") else 1.0, zorder=3)
        _log_energy_axis(ax)
        ax.set_title(scen.replace("_", " "), loc="left")
        ax.set_xlabel("cooling energy, pump-s (log)")
    axes[0].set_ylabel("degree-s above 45 °C (lower is safer)")
    handles = [plt.Line2D([], [], color=CTRL_STYLE["shield_only"]["color"], marker="o", mec=SURFACE,
                          label="shield only; points = assumed burst 0, 30, 60, 90, 120, 180 s")] + [
        plt.Line2D([], [], ls="none", marker=CTRL_STYLE[c]["marker"], ms=8, color=CTRL_STYLE[c]["color"],
                   mfc=CTRL_STYLE[c].get("mfc", CTRL_STYLE[c]["color"]), label=CTRL_STYLE[c]["label"])
        for c in ("always_on", "pid")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("The price of robustness: assuming longer worst-case bursts buys safety with energy "
                 f"(mean of {df.seed.nunique()} seeds)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(RESULTS / "shield_tradeoff.png", dpi=150)
    plt.close(fig)


def main() -> None:
    global RESULTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS, help="output folder (default: results/)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    est = sub.add_parser("estimation", help="KF vs AKF in every scenario")
    est.add_argument("--seeds", type=int, default=20)
    ctl = sub.add_parser("controllers", help="controllers x scenarios x seeds")
    ctl.add_argument("--controllers", nargs="+", default=None,
                     help="default: all baselines plus every trained variant found in --runs, with and without shield")
    ctl.add_argument("--runs", type=Path, default=RUNS, help="folder with trained policies (train.py output)")
    ctl.add_argument("--seeds", type=int, default=20)
    ctl.add_argument("--estimator", choices=["akf", "kf"], default="akf")
    ctl.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    tun = sub.add_parser("tune-pid", help="grid search for the PID gains (tuning seeds only)")
    tun.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    trn = sub.add_parser("training", help="training-time safety from train_log.csv files")
    trn.add_argument("--runs", type=Path, default=RUNS)
    sf = sub.add_parser("sensor-failure", help="permanent surface-sensor failure 10 min into the episode")
    sf.add_argument("--seeds", type=int, default=20)
    sf.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    sf.add_argument("--runs", type=Path, default=RUNS)
    swp = sub.add_parser("shield-sweep", help="shield energy/safety vs assumed worst-case burst length")
    swp.add_argument("--seeds", type=int, default=20)
    swp.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    RESULTS = args.results
    if args.cmd == "estimation":
        df = estimation_study(args.seeds)
        print((RESULTS / "estimation.md").read_text())
        print(f"wrote {RESULTS}/estimation.csv, estimation.md, estimation.png, calibration.png ({len(df)} episodes)")
    elif args.cmd == "controllers":
        runs = discover_runs(args.runs)
        names = args.controllers or default_controllers(runs)
        df = controller_study(names, args.seeds, args.estimator, args.jobs, runs)
        suffix = "" if args.estimator == "akf" else f"_{args.estimator}"
        print((RESULTS / f"controllers{suffix}.md").read_text())
        print(f"{len(df)} episodes")
    elif args.cmd == "tune-pid":
        tune_pid(args.jobs)
    elif args.cmd == "training":
        if training_study(args.runs) is not None:
            print((RESULTS / "training_safety.md").read_text())
    elif args.cmd == "sensor-failure":
        sensor_failure_study(args.seeds, args.jobs, discover_runs(args.runs))
        print((RESULTS / "sensor_failure.md").read_text())
    elif args.cmd == "shield-sweep":
        shield_sweep(args.seeds, args.jobs)
        print((RESULTS / "shield_tradeoff.md").read_text())


if __name__ == "__main__":
    main()
