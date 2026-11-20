"""Two-state lumped thermal model of a cylindrical cell, drive cycles, scenarios and the simulator.

States: Tc (core, not measured) and Ts (surface, measured with noise).

    Cc * dTc/dt = Q - (Tc - Ts) / Rc
    Cs * dTs/dt = (Tc - Ts) / Rc - (Ts - Tf) * h(u)
    Q = I^2 * R_int,   h(u) = h0 + h1 * u,   u in [0, 1]

For a fixed u and Q the model is linear, so each 2 s step is discretised exactly
(zero-order hold) with the closed-form 2x2 matrix exponential.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

DT = 2.0  # s
T_LIMIT = 45.0  # degC, core constraint
EPISODE_STEPS = 900  # 30 min
SENSOR_SIGMA = 0.3  # degC, nominal surface-sensor noise
I_MAX = 25.0  # A


@dataclass(frozen=True)
class CellParams:
    # Cc, Cs, Rc: Table 1 of Lin et al., J. Power Sources 257 (2014) 1-11 (A123 26650 LiFePO4 cell).
    Cc: float = 62.7  # J/K, core heat capacity
    Cs: float = 4.5  # J/K, surface heat capacity
    Rc: float = 1.94  # K/W, core-to-surface resistance
    h0: float = 0.15  # W/K, natural convection (u = 0)
    h1: float = 0.85  # W/K, extra cooling at u = 1
    R_int: float = 0.01  # ohm, internal resistance
    Tf: float = 25.0  # degC, coolant / ambient temperature

    def scaled(self, R_int_scale: float = 1.0, Rc_scale: float = 1.0) -> "CellParams":
        return replace(self, R_int=self.R_int * R_int_scale, Rc=self.Rc * Rc_scale)


NOMINAL = CellParams()


def discretise(p: CellParams, u, dt: float = DT):
    """Exact ZOH discretisation for input w = [Q, Tf].

    Returns (Ad, Bd) with x[k+1] = Ad @ x[k] + Bd @ w. Works for scalar u (shapes (2,2))
    or an array of u (shapes (..., 2, 2)).
    """
    u = np.asarray(u, dtype=float)
    h = p.h0 + p.h1 * u
    a = np.full_like(h, -1.0 / (p.Rc * p.Cc))
    b = np.full_like(h, 1.0 / (p.Rc * p.Cc))
    c = np.full_like(h, 1.0 / (p.Rc * p.Cs))
    d = -(1.0 / p.Rc + h) / p.Cs
    half_tr = 0.5 * (a + d)
    det = a * d - b * c  # = h / (Rc Cc Cs) > 0, so A is invertible
    # Discriminant (a-d)^2/4 + b*c > 0: real, distinct eigenvalues.
    disc = np.sqrt(0.25 * (a - d) ** 2 + b * c)
    l1, l2 = half_tr + disc, half_tr - disc
    e1, e2 = np.exp(l1 * dt), np.exp(l2 * dt)
    # Sylvester formula: expm(A dt) = (e1 (A - l2 I) - e2 (A - l1 I)) / (l1 - l2)
    k1 = (e1 - e2) / (l1 - l2)
    k0 = (l1 * e2 - l2 * e1) / (l1 - l2)
    Ad = np.empty(h.shape + (2, 2))
    Ad[..., 0, 0] = k1 * a + k0
    Ad[..., 0, 1] = k1 * b
    Ad[..., 1, 0] = k1 * c
    Ad[..., 1, 1] = k1 * d + k0
    # Bd = A^-1 (Ad - I) B, with B = diag(1/Cc, h/Cs)
    M00, M01 = Ad[..., 0, 0] - 1.0, Ad[..., 0, 1]
    M10, M11 = Ad[..., 1, 0], Ad[..., 1, 1] - 1.0
    i00, i01, i10, i11 = d / det, -b / det, -c / det, a / det
    Bd = np.empty(h.shape + (2, 2))
    Bd[..., 0, 0] = (i00 * M00 + i01 * M10) / p.Cc
    Bd[..., 1, 0] = (i10 * M00 + i11 * M10) / p.Cc
    Bd[..., 0, 1] = (i00 * M01 + i01 * M11) * h / p.Cs
    Bd[..., 1, 1] = (i10 * M01 + i11 * M11) * h / p.Cs
    return Ad, Bd


def steady_state(p: CellParams, I: float, u: float) -> tuple[float, float]:
    """Steady-state (Tc, Ts) for constant current and cooling."""
    Q = I * I * p.R_int
    h = p.h0 + p.h1 * u
    Ts = p.Tf + Q / h
    return Ts + Q * p.Rc, Ts


# ---------------------------------------------------------------------------
# Scenarios and episode data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str = "nominal"
    R_int_scale: float = 1.0  # true R_int / nominal
    Rc_scale: float = 1.0  # true Rc / nominal
    noise_scale: float = 1.0  # sensor noise / nominal
    outage_frac: float = 0.0  # expected fraction of time the sensor is offline
    outage_len_s: tuple[float, float] = (30.0, 120.0)
    ageing_end: float = 1.0  # R_int scale at the end of the episode (ramps from 1.0)
    burst_len_s: tuple[float, float] = (60.0, 180.0)
    sensor_fail_s: float | None = None  # surface sensor dead from this time on (permanent failure)


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in [
        Scenario("nominal"),
        Scenario("heat_mismatch", R_int_scale=1.3),
        Scenario("resistance_mismatch", Rc_scale=1.2),
        Scenario("noisy", noise_scale=3.0),
        Scenario("outage", outage_frac=0.2),
        Scenario("ageing", ageing_end=1.4),
        Scenario("long_bursts", burst_len_s=(180.0, 360.0)),
    ]
}


def drive_cycle(
    rng: np.random.Generator,
    n_steps: int = EPISODE_STEPS,
    seg_s: tuple[float, float] = (30.0, 120.0),
    burst_s: tuple[float, float] = (60.0, 180.0),
    burst_I: tuple[float, float] = (20.0, I_MAX),
    burst_prob: float = 0.15,
    dt: float = DT,
) -> np.ndarray:
    """Random piecewise-constant current profile (A), one value per step."""
    I = np.empty(n_steps)
    k = 0
    while k < n_steps:
        if rng.random() < burst_prob:
            level, dur = rng.uniform(*burst_I), rng.uniform(*burst_s)
        else:
            level, dur = rng.uniform(0.0, I_MAX), rng.uniform(*seg_s)
        n = max(1, int(round(dur / dt)))
        I[k : k + n] = level
        k += n
    return I


def outage_mask(
    rng: np.random.Generator, n: int, frac: float, len_s: tuple[float, float], dt: float = DT
) -> np.ndarray:
    """Boolean array, True where the sensor is online. Outages last len_s seconds."""
    online = np.ones(n, dtype=bool)
    if frac <= 0.0:
        return online
    mean_len = 0.5 * (len_s[0] + len_s[1]) / dt
    p_start = frac / (mean_len * (1.0 - frac))  # stationary offline fraction = frac
    k = 0
    while k < n:
        if rng.random() < p_start:
            m = max(1, int(round(rng.uniform(*len_s) / dt)))
            online[k : k + m] = False
            k += m
        else:
            k += 1
    return online


@dataclass
class Episode:
    I: np.ndarray  # (n,) current applied during step k
    R_int_scale: np.ndarray  # (n,) true R_int scale during step k (ageing ramp)
    noise: np.ndarray  # (n+1,) measurement noise at sample k (degC)
    online: np.ndarray  # (n+1,) sensor online at sample k
    T0: float  # initial core and surface temperature


def make_episode(scenario: Scenario, seed: int, n_steps: int = EPISODE_STEPS) -> Episode:
    """Deterministic episode data. Each random stream has its own child seed, so for a
    given seed the drive cycle (except burst length), noise and outages do not depend on
    which controller runs, and the drive cycle matches across most scenarios."""
    ss = np.random.SeedSequence(seed)
    r_cycle, r_noise, r_out, r_init = (np.random.default_rng(s) for s in ss.spawn(4))
    I = drive_cycle(r_cycle, n_steps, burst_s=scenario.burst_len_s)
    ramp = 1.0 + (scenario.ageing_end - 1.0) * np.linspace(0.0, 1.0, n_steps)
    noise = r_noise.standard_normal(n_steps + 1) * SENSOR_SIGMA * scenario.noise_scale
    online = outage_mask(r_out, n_steps + 1, scenario.outage_frac, scenario.outage_len_s)
    if scenario.sensor_fail_s is not None:
        online[int(round(scenario.sensor_fail_s / DT)):] = False
    T0 = NOMINAL.Tf + r_init.uniform(0.0, 3.0)
    return Episode(I, scenario.R_int_scale * ramp, noise, online, T0)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class ThermalSim:
    """True plant + estimator for one episode. Controllers and the Gym env both use this.

    Timing: at sample k the controller sees the estimate after measurement k and the
    current I[k], chooses u[k]; the plant then moves to k+1 with (I[k], u[k]), the
    estimator predicts with the same inputs, and measurement k+1 (if online) updates it.
    """

    def __init__(self, scenario: Scenario = SCENARIOS["nominal"], estimator=None,
                 params: CellParams = NOMINAL, n_steps: int = EPISODE_STEPS):
        from estimators import KalmanFilter  # local import avoids a cycle

        self.scenario = scenario
        self.params = params
        self.true_params = params.scaled(Rc_scale=scenario.Rc_scale)  # R_int set per step
        self.n_steps = n_steps
        self.est = estimator if estimator is not None else KalmanFilter(augment=True)

    def reset(self, seed: int, scenario: Scenario | None = None) -> None:
        if scenario is not None:  # per-episode scenario (domain randomisation in training)
            self.scenario = scenario
            self.true_params = self.params.scaled(Rc_scale=scenario.Rc_scale)
        n = self.n_steps
        self.ep = make_episode(self.scenario, seed, n)
        self.k = 0
        self.x = np.array([self.ep.T0, self.ep.T0])
        self.u_prev = 0.0
        self.est.reset()
        h = self.hist = {
            key: np.full(n + 1, np.nan)
            for key in ("Tc", "Ts", "Tc_hat", "Ts_hat", "sigma", "theta", "z", "nu", "S", "u", "I")
        }
        h["online"] = self.ep.online.copy()
        h["P2"] = np.full((n + 1, 2, 2), np.nan)  # [Tc, Ts] block of the covariance
        self.Ts_last = np.nan
        self._measure_and_record()

    @property
    def done(self) -> bool:
        return self.k >= self.n_steps

    @property
    def I_now(self) -> float:
        return float(self.ep.I[min(self.k, self.n_steps - 1)])

    @property
    def sensor_ok(self) -> bool:
        return bool(self.ep.online[self.k])

    @property
    def Tc_true(self) -> float:
        return float(self.x[0])

    def step(self, u: float) -> dict:
        assert not self.done, "episode finished; call reset()"
        u = float(np.clip(u, 0.0, 1.0))
        k = self.k
        I = float(self.ep.I[k])
        p = replace(self.true_params, R_int=self.params.R_int * float(self.ep.R_int_scale[k]))
        Ad, Bd = discretise(p, u)
        self.x = Ad @ self.x + Bd @ np.array([I * I * p.R_int, p.Tf])
        self.est.predict(I, u)
        self.hist["u"][k], self.hist["I"][k] = u, I
        self.k += 1
        self._measure_and_record()
        du = abs(u - self.u_prev)
        self.u_prev = u
        Tc = self.Tc_true
        return {
            "Tc_true": Tc,
            "violation": max(0.0, Tc - T_LIMIT),
            "energy": u**3 * DT,
            "du": du,
            "Tc_hat": self.est.Tc_hat,
            "sigma_Tc": self.est.sigma_Tc,
            "theta_hat": self.est.theta_hat,
            "sensor_ok": self.sensor_ok,
        }

    def _measure_and_record(self) -> None:
        k, h = self.k, self.hist
        z = None
        if self.ep.online[k]:
            z = float(self.x[1] + self.ep.noise[k])
            self.Ts_last = z
            nu, S = self.est.update(z)
            h["z"][k], h["nu"][k], h["S"][k] = z, nu, S
        h["Tc"][k], h["Ts"][k] = self.x
        h["Tc_hat"][k], h["Ts_hat"][k] = self.est.x[0], self.est.x[1]
        h["sigma"][k], h["theta"][k] = self.est.sigma_Tc, self.est.theta_hat
        h["P2"][k] = self.est.P[:2, :2]


# ---------------------------------------------------------------------------
# Observations and the Gymnasium environment (RL)
# ---------------------------------------------------------------------------

OBS_MODES = ("full", "no_sigma", "raw")
RAW_HISTORY = 10  # surface readings in the `raw` observation (20 s)


class ObsBuilder:
    """Builds the policy observation from a ThermalSim. Shared by the env and the
    evaluation controller, so training and testing see exactly the same thing.

    full:     [Tc_hat, sigma_Tc, theta_hat, Ts_last, sensor_ok, I, u_prev]
    no_sigma: [Tc_hat, theta_hat, Ts_last, sensor_ok, I, u_prev]
    raw:      last 10 surface readings, their sensor_ok flags, I, u_prev (no estimator)

    Ts_last is the last valid reading (held during outages). The true core temperature is
    never part of the observation. Call once per sample, in order: the raw mode keeps a
    history.
    """

    def __init__(self, mode: str = "full"):
        assert mode in OBS_MODES, mode
        self.mode = mode
        self.dim = {"full": 7, "no_sigma": 6, "raw": 2 * RAW_HISTORY + 2}[mode]
        self._hist_T: list[float] = []
        self._hist_ok: list[float] = []

    def reset(self) -> None:
        self._hist_T, self._hist_ok = [], []

    def __call__(self, sim: ThermalSim) -> np.ndarray:
        Ts = sim.Ts_last if np.isfinite(sim.Ts_last) else float(sim.est.x[1])
        ok = float(sim.sensor_ok)
        # Rough centring/scaling; VecNormalize does the rest.
        I, u = sim.I_now / I_MAX, sim.u_prev
        if self.mode == "raw":
            if not self._hist_T:
                self._hist_T = [Ts] * RAW_HISTORY
                self._hist_ok = [ok] * RAW_HISTORY
            self._hist_T = self._hist_T[1:] + [Ts]
            self._hist_ok = self._hist_ok[1:] + [ok]
            temps = (np.array(self._hist_T) - NOMINAL.Tf) / 10.0
            return np.concatenate([temps, self._hist_ok, [I, u]]).astype(np.float32)
        est = sim.est
        theta = est.theta_hat if est.augment else 1.0
        core = [(est.Tc_hat - NOMINAL.Tf) / 10.0]
        sigma = [est.sigma_Tc] if self.mode == "full" else []
        rest = [theta, (Ts - NOMINAL.Tf) / 10.0, ok, I, u]
        return np.array(core + sigma + rest, dtype=np.float32)


def sample_training_scenario(rng: np.random.Generator, dr: bool) -> Scenario:
    """Training conditions. Every variant sees some extra sensor noise
    and outages in half of the episodes; `dr` also randomises the cell's heat generation
    and core-to-surface resistance."""
    return Scenario(
        name="train",
        R_int_scale=rng.uniform(0.8, 1.3) if dr else 1.0,
        Rc_scale=rng.uniform(0.8, 1.2) if dr else 1.0,
        noise_scale=rng.uniform(1.0, 1.5),
        outage_frac=0.2 if rng.random() < 0.5 else 0.0,
    )


try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # the RL parts need gymnasium; the rest of the module does not
    gym = None

if gym is not None:

    class ThermalEnv(gym.Env):
        """Gymnasium wrapper around ThermalSim.

        Action a in [-1, 1] maps to cooling u = (a + 1) / 2 (SB3 prefers symmetric actions).
        Reward per step: -u^3 - 10 * max(0, Tc_true - 45) - 0.01 * |u - u_prev|, using the true
        core temperature (privileged training: only the simulator knows it).

        With shield=True the uncertainty-aware shield is part of the environment: the
        executed action is the shielded one, and the policy pays an extra
        intervention_penalty * (u_safe - u_rl) so it does not simply lean on the shield.

        scenario=None samples a training scenario per episode; otherwise it is fixed.
        """

        metadata = {"render_modes": []}

        def __init__(self, obs_mode: str = "full", dr: bool = False, shield: bool = False,
                     intervention_penalty: float = 0.1, scenario: Scenario | None = None,
                     augment: bool = True):
            from controllers import Shield
            from estimators import KalmanFilter

            super().__init__()
            self.obs_builder = ObsBuilder(obs_mode)
            self.dr, self.fixed_scenario = dr, scenario
            self.shield = Shield() if shield else None
            self.intervention_penalty = intervention_penalty
            self.sim = ThermalSim(scenario or SCENARIOS["nominal"], KalmanFilter(augment=augment))
            self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_builder.dim,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            rng = self.np_random
            scen = self.fixed_scenario or sample_training_scenario(rng, self.dr)
            # Episode seeds stay clear of evaluation (0-19) and tuning (1000-1019) seeds.
            ep_seed = int(rng.integers(1_000_000, 2**31 - 1))
            if options and "episode_seed" in options:
                ep_seed = int(options["episode_seed"])
            self.sim.reset(ep_seed, scen)
            self.obs_builder.reset()
            if self.shield is not None:
                self.shield.reset()
            self._ep = {"energy": 0.0, "deg_s": 0.0, "interventions": 0, "max_Tc": self.sim.Tc_true}
            return self.obs_builder(self.sim), {"R_int_scale": scen.R_int_scale, "Rc_scale": scen.Rc_scale}

        def step(self, action):
            u_rl = float(np.clip((float(np.asarray(action).ravel()[0]) + 1.0) / 2.0, 0.0, 1.0))
            u = u_rl
            if self.shield is not None:
                u = self.shield.safe_action(self.sim, u_rl)
            u_prev = self.sim.u_prev
            info = self.sim.step(u)
            reward = -(u**3) - 10.0 * info["violation"] - 0.01 * abs(u - u_prev)
            if self.shield is not None:
                reward -= self.intervention_penalty * (u - u_rl)
            ep = self._ep
            ep["energy"] += info["energy"]
            ep["deg_s"] += info["violation"] * DT
            ep["interventions"] += int(u > u_rl + 1e-9)
            ep["max_Tc"] = max(ep["max_Tc"], info["Tc_true"])
            info.update(u=u, u_rl=u_rl)
            truncated = self.sim.done
            if truncated:
                info["episode_stats"] = {
                    "energy": ep["energy"], "deg_s": ep["deg_s"], "max_Tc": ep["max_Tc"],
                    "shield_rate": 100.0 * ep["interventions"] / self.sim.n_steps,
                    "R_int_scale": self.sim.scenario.R_int_scale, "Rc_scale": self.sim.scenario.Rc_scale,
                }
            return self.obs_builder(self.sim), float(reward), False, truncated, info
