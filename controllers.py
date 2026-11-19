"""Controllers. Each has reset() and act(sim) -> u in [0, 1].

All of them read the estimator inside the ThermalSim (sim.est), except the oracle PID,
which reads the true core temperature (a reference that is impossible in practice).
"""

from __future__ import annotations

import itertools

import numpy as np

from thermal_env import DT, T_LIMIT

SHIELD_LIMIT = T_LIMIT - 1.0  # 44 degC: one degree of margin below the hard limit
K_SIGMA = 2.0  # require mean + 2 sigma below the shield limit

# PID gains from `python evaluate.py tune-pid` (grid search on tuning seeds 1000-1019,
# nominal scenario, objective = energy + 10 * degree-seconds above 45 degC, matching the
# RL reward). Seeds 0-19 are kept for evaluation. The search chose kp only (ki = kd = 0):
# on the nominal tuning seeds, a proportional controller is cheapest while staying < 45 degC.
PID_GAINS = {"kp": 0.3, "ki": 0.0, "kd": 0.0}


class Controller:
    name = "controller"

    def reset(self) -> None:
        pass

    def act(self, sim) -> float:
        raise NotImplementedError

    def stats(self) -> dict:
        return {}


class AlwaysOn(Controller):
    """Full cooling all the time: the physical floor for violations."""
    name = "always_on"

    def act(self, sim) -> float:
        return 1.0


class PID(Controller):
    """PID on the core temperature estimate (or the true value for the oracle).

    Error is T - setpoint, so a hot core gives more cooling. Anti-windup by conditional
    integration: the integral only moves when the output is not saturated, or when the
    error pushes the output back into range.
    """

    def __init__(self, kp=PID_GAINS["kp"], ki=PID_GAINS["ki"], kd=PID_GAINS["kd"],
                 setpoint: float = 42.0, oracle: bool = False):
        self.kp, self.ki, self.kd, self.setpoint, self.oracle = kp, ki, kd, setpoint, oracle
        self.name = "oracle_pid" if oracle else "pid"
        self.reset()

    def reset(self) -> None:
        self.integ = 0.0
        self.prev_e = None

    def act(self, sim) -> float:
        T = sim.Tc_true if self.oracle else sim.est.Tc_hat
        e = T - self.setpoint
        d = 0.0 if self.prev_e is None else (e - self.prev_e) / DT
        self.prev_e = e
        raw = self.kp * e + self.ki * self.integ + self.kd * d
        u = min(1.0, max(0.0, raw))
        if raw == u or (raw > 1.0 and e < 0.0) or (raw < 0.0 and e > 0.0):
            self.integ += e * DT
        return u


class BackupCheck:
    """Can we hold u for a while and still survive a worst-case burst afterwards?

    Hold u for `hold` steps at the current I, then apply full cooling (the backup) while
    the current is at I_MAX for `burst_steps`. Require Tc_mean + k_sigma * Tc_sigma <= limit
    over the whole window. The estimator's own state (including theta, the heat scale) and
    covariance are propagated, so a cell that makes more heat gets a bigger reserve.

    Holding u (rather than applying it for one step) matters: with a one-step hold the
    check barely depends on u, so at the boundary only u = 1 passes and the command
    chatters between low and full cooling. A 30 s hold gave the same safety with 25 % less
    energy and far fewer on/off switches (tuning seeds).

    The backup dynamics are fixed, so the burst window uses precomputed powers of the backup
    transition matrix. Assumption: bursts are at most 25 A for 180 s (the drive-cycle spec).
    The long_bursts scenario breaks it on purpose.
    """

    def __init__(self, I_max: float = 25.0, burst_steps: int = 90, hold: int = 15,
                 limit: float = SHIELD_LIMIT, k_sigma: float = K_SIGMA):
        self.I_max, self.burst_steps, self.hold = I_max, burst_steps, hold
        self.limit, self.k_sigma = limit, k_sigma
        self._cache_key = None

    def _prepare(self, est) -> None:
        key = (id(est), est.n)
        if key == self._cache_key:
            return
        Fb, cb = est.model(self.I_max, 1.0)
        n = est.n
        phi = np.zeros((self.burst_steps, n))  # first row of Fb^j, j = 1..B
        gamma = np.zeros(self.burst_steps)  # Tc offset from the inputs
        sig00 = np.zeros(self.burst_steps)  # Tc variance added by process noise
        Fj, off, Sig = np.eye(n), np.zeros(n), np.zeros((n, n))
        for j in range(self.burst_steps):
            off = Fb @ off + cb
            Sig = Fb @ Sig @ Fb.T + est.Qk
            Fj = Fb @ Fj
            phi[j], gamma[j], sig00[j] = Fj[0], off[0], Sig[0, 0]
        self.phi, self.gamma, self.sig00 = phi, gamma, sig00
        self._cache_key = key

    def ok(self, sim, u_cands: np.ndarray) -> np.ndarray:
        est = sim.est
        self._prepare(est)
        u_cands = np.asarray(u_cands, dtype=float)
        n_c = len(u_cands)
        F, c = est.model(sim.I_now, u_cands)  # (n_c, n, n), (n_c, n)
        m = np.broadcast_to(est.x, (n_c, est.n)).copy()
        P = np.broadcast_to(est.P, (n_c, est.n, est.n)).copy()
        worst = np.full(n_c, -np.inf)
        for _ in range(self.hold):  # hold the candidate at the current I
            m = np.einsum("cij,cj->ci", F, m) + c
            P = F @ P @ np.swapaxes(F, -1, -2) + est.Qk
            worst = np.maximum(worst, m[:, 0] + self.k_sigma * np.sqrt(P[:, 0, 0]))
        worst = np.maximum(worst, self.burst_worst(est, m, P))  # then the worst-case burst
        return worst <= self.limit

    def burst_worst(self, est, m: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Worst Tc_mean + k_sigma * Tc_sigma during a worst-case burst with full cooling,
        starting from predicted states m (n_c, n) with covariances P (n_c, n, n)."""
        self._prepare(est)
        mean = m @ self.phi.T + self.gamma  # (n_c, B)
        var = np.einsum("jk,ckl,jl->cj", self.phi, P, self.phi) + self.sig00
        return (mean + self.k_sigma * np.sqrt(var)).max(axis=1)


class Shield:
    """Uncertainty-aware safety filter.

    A candidate u passes if, holding u for `horizon` steps (30 s) at the current I with no
    measurements, Tc_mean + k_sigma * Tc_sigma <= limit at every step, and (with
    backup=True) a worst-case burst can still be survived with full cooling afterwards
    (see BackupCheck). backup=False is the v1 design: the 30 s check only.
    Return u_rl if it passes, otherwise the smallest grid value above u_rl that passes,
    otherwise u = 1.
    """

    def __init__(self, horizon: int = 15, limit: float = SHIELD_LIMIT, k_sigma: float = K_SIGMA,
                 n_grid: int = 21, backup: bool = True, burst_steps: int = 90):
        self.horizon, self.limit, self.k_sigma = horizon, limit, k_sigma
        self.grid = np.linspace(0.0, 1.0, n_grid)
        self.backup = (BackupCheck(burst_steps=burst_steps, hold=horizon, limit=limit, k_sigma=k_sigma)
                       if backup else None)
        self.reset()

    def reset(self) -> None:
        self.n_calls = 0
        self.n_interventions = 0

    def passes(self, sim, cands: np.ndarray) -> np.ndarray:
        if self.backup is not None:  # its hold phase already covers the 30 s check
            return self.backup.ok(sim, cands)
        mean, sig = sim.est.forecast(sim.I_now, np.repeat(cands[:, None], self.horizon, axis=1))
        return np.all(mean + self.k_sigma * sig <= self.limit, axis=1)

    def safe_action(self, sim, u_rl: float) -> float:
        u_rl = float(np.clip(u_rl, 0.0, 1.0))
        cands = np.concatenate([[u_rl], self.grid[self.grid > u_rl]])
        ok = self.passes(sim, cands)
        u = float(cands[np.argmax(ok)]) if ok.any() else 1.0
        self.n_calls += 1
        self.n_interventions += u > u_rl + 1e-9
        return u

    def stats(self) -> dict:
        return {"shield_rate": 100.0 * self.n_interventions / max(1, self.n_calls)}


class ShieldOnly(Controller):
    """The shield applied to u_rl = 0: the cheapest action the shield accepts.

    If an RL policy plus the shield cannot beat it, the RL part adds nothing.
    `shield_only_nobackup` is the v1 shield (horizon check only), kept as an ablation.
    """

    def __init__(self, shield: Shield | None = None):
        self.shield = shield or Shield()
        self.name = "shield_only" if self.shield.backup is not None else "shield_only_nobackup"

    def reset(self) -> None:
        self.shield.reset()

    def act(self, sim) -> float:
        return self.shield.safe_action(sim, 0.0)

    def stats(self) -> dict:
        return self.shield.stats()


class PIDShield(Controller):
    """A PID that aims below the shield's limit, with the shield as backstop.

    The PID tries to keep the estimated core near `setpoint` (default 40 degC), cooling
    gently and early; the shield (limit 44 degC, backup check) only steps in when that is not
    enough. Aiming lower than the brake line leaves a margin that also covers estimation
    errors the filter cannot see (the resistance-mismatch case).
    """

    def __init__(self, setpoint: float = 40.0):
        self.pid = PID(setpoint=setpoint)
        self.shield = Shield()
        self.name = f"pid{setpoint:.0f}_shield"

    def reset(self) -> None:
        self.pid.reset()
        self.shield.reset()

    def act(self, sim) -> float:
        return self.shield.safe_action(sim, self.pid.act(sim))

    def stats(self) -> dict:
        return self.shield.stats()


class MPC(Controller):
    """Economic MPC by enumeration.

    Horizon 30 steps (60 s) in 3 blocks of 10 steps; each block's u is one of
    {0, 0.2, ..., 1}, so 216 sequences are evaluated in one batch. Cost:
    sum(u^3 dt) + 0.01 * sum|du| (including the change from the previous u). Constraint:
    the same Tc_mean + 2 sigma <= 44 degC as the shield, at every step, and the first
    block's action must pass the backup check (held for the block, then a worst-case burst),
    so the two differ mainly in how they plan for energy. The current is held constant (no
    drive-cycle preview). If nothing is feasible, take the plan with the smallest
    worst-case excess, which is full cooling in practice.

    preview=True gives the MPC the true current for the next 60 s (a perfect load forecast:
    an upper bound on what forecasting can buy). The plans are predicted with that current,
    and the worst-case burst is only assumed after the known window: each plan must leave
    a state from which a 25 A / 180 s burst can still be survived with full cooling.
    """

    def __init__(self, n_blocks: int = 3, block_len: int = 10, levels=np.linspace(0, 1, 6),
                 limit: float = SHIELD_LIMIT, k_sigma: float = K_SIGMA, w_du: float = 0.01,
                 backup: bool = True, preview: bool = False):
        self.name = "mpc_preview" if preview else "mpc"
        self.preview = preview
        plans = np.array(list(itertools.product(levels, repeat=n_blocks)))  # (n_plans, n_blocks)
        self.u_seq = np.repeat(plans, block_len, axis=1)  # (n_plans, horizon)
        du = np.abs(np.diff(plans, axis=1)).sum(axis=1)
        self.base_cost = (self.u_seq**3).sum(axis=1) * DT + w_du * du
        self.levels = np.asarray(levels, dtype=float)
        self.first = plans[:, 0]
        self.first_idx = np.searchsorted(self.levels, self.first)
        self.limit, self.k_sigma, self.w_du = limit, k_sigma, w_du
        self.backup = (BackupCheck(hold=block_len, limit=limit, k_sigma=k_sigma)
                       if backup else None)

    def reset(self) -> None:
        self.u_prev = 0.0

    def act(self, sim) -> float:
        cost = self.base_cost + self.w_du * np.abs(self.first - self.u_prev)
        if self.preview:
            H = self.u_seq.shape[1]
            I_known = sim.ep.I[sim.k : sim.k + H]
            I_seq = np.concatenate([I_known, np.full(H - len(I_known), I_known[-1])])
            mean, sig, m_end, P_end = sim.est.forecast(I_seq, self.u_seq, return_final=True)
            excess = (mean + self.k_sigma * sig - self.limit).max(axis=1)
            if self.backup is not None:  # a burst may start right after the known window
                excess = np.maximum(excess, self.backup.burst_worst(sim.est, m_end, P_end) - self.limit)
            feasible = excess <= 0.0
        else:
            mean, sig = sim.est.forecast(sim.I_now, self.u_seq)
            excess = (mean + self.k_sigma * sig - self.limit).max(axis=1)
            feasible = excess <= 0.0
            if self.backup is not None:
                feasible &= self.backup.ok(sim, self.levels)[self.first_idx]
        i = np.argmin(np.where(feasible, cost, np.inf)) if feasible.any() else np.argmin(excess)
        self.u_prev = float(self.first[i])
        return self.u_prev


class PPOController(Controller):
    """A trained PPO policy (from train.py), optionally behind the shield.

    Builds the observation with the same ObsBuilder as the training env, normalises it with
    the saved VecNormalize statistics, and acts deterministically.
    """

    def __init__(self, final_dir, shield: bool = False):
        import json
        from pathlib import Path

        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from thermal_env import ObsBuilder, ThermalEnv

        torch.set_num_threads(1)  # evaluation runs one process per core
        final_dir = Path(final_dir)
        cfg = json.loads((final_dir / "config.json").read_text())
        self.variant, self.train_seed = cfg["variant"], cfg["seed"]
        self.name = self.variant + ("+shield" if shield else "")
        self.obs_builder = ObsBuilder(cfg["env"]["obs_mode"])
        self.model = PPO.load(str(final_dir / "model.zip"), device="cpu")
        env_opts = {k: v for k, v in cfg["env"].items() if k != "shield"}
        self.vecnorm = VecNormalize.load(str(final_dir / "vecnormalize.pkl"),
                                         DummyVecEnv([lambda: ThermalEnv(**env_opts)]))
        self.vecnorm.training = False
        self.shield = Shield() if shield else None

    def reset(self) -> None:
        self.obs_builder.reset()
        if self.shield is not None:
            self.shield.reset()

    def act(self, sim) -> float:
        obs = self.vecnorm.normalize_obs(self.obs_builder(sim)[None, :])
        a, _ = self.model.predict(obs, deterministic=True)
        u_rl = float(np.clip((float(a.ravel()[0]) + 1.0) / 2.0, 0.0, 1.0))
        return self.shield.safe_action(sim, u_rl) if self.shield is not None else u_rl

    def stats(self) -> dict:
        return self.shield.stats() if self.shield is not None else {}


def make_controller(name: str) -> Controller:
    factories = {
        "always_on": AlwaysOn,
        "pid": PID,
        "oracle_pid": lambda: PID(oracle=True),
        "shield_only": ShieldOnly,
        "shield_only_nobackup": lambda: ShieldOnly(Shield(backup=False)),
        "mpc": MPC,
        "mpc_preview": lambda: MPC(preview=True),
        "pid40_shield": lambda: PIDShield(40.0),
    }
    return factories[name]()


BASELINES = ["always_on", "pid", "oracle_pid", "pid40_shield", "shield_only_nobackup", "shield_only", "mpc",
             "mpc_preview"]
