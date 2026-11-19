"""Unit tests: physics, discretisation, scenarios and filters."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimators import KalmanFilter  # noqa: E402
from thermal_env import (  # noqa: E402
    DT, NOMINAL, SCENARIOS, ThermalSim, discretise, drive_cycle, make_episode,
    outage_mask, steady_state,
)


def simulate(p, I_seq, u, x0):
    x = np.array(x0, dtype=float)
    peak = x[0]
    Ad, Bd = discretise(p, u)
    for I in I_seq:
        x = Ad @ x + Bd @ np.array([I * I * p.R_int, p.Tf])
        peak = max(peak, x[0])
    return x, peak


def euler_step(p, x, I, u, n_sub=20000):
    Tc, Ts = x
    d = DT / n_sub
    h = p.h0 + p.h1 * u
    Q = I * I * p.R_int
    for _ in range(n_sub):
        Tc, Ts = (Tc + d * (Q - (Tc - Ts) / p.Rc) / p.Cc,
                  Ts + d * ((Tc - Ts) / p.Rc - (Ts - p.Tf) * h) / p.Cs)
    return np.array([Tc, Ts])


# -- physics ------------------------------------------------------------------

@pytest.mark.parametrize("u, expected", [(1.0, 43.4), (0.0, 78.8)])
def test_steady_state_25A(u, expected):
    assert steady_state(NOMINAL, 25.0, u)[0] == pytest.approx(expected, abs=0.2)
    x, _ = simulate(NOMINAL, [25.0] * 20000, u, [25.0, 25.0])
    assert x[0] == pytest.approx(expected, abs=0.2)


@pytest.mark.parametrize("u", [0.0, 0.37, 1.0])
def test_exact_discretisation_matches_fine_euler(u):
    x0 = np.array([40.0, 33.0])
    Ad, Bd = discretise(NOMINAL, u)
    exact = Ad @ x0 + Bd @ np.array([20.0**2 * NOMINAL.R_int, NOMINAL.Tf])
    np.testing.assert_allclose(exact, euler_step(NOMINAL, x0, 20.0, u), atol=1e-3)


def test_discretise_vectorised_matches_scalar():
    us = np.linspace(0, 1, 7)
    Ad, Bd = discretise(NOMINAL, us)
    for i, u in enumerate(us):
        a, b = discretise(NOMINAL, u)
        np.testing.assert_allclose(Ad[i], a)
        np.testing.assert_allclose(Bd[i], b)


@pytest.mark.parametrize("R_int_scale, expected", [(1.0, 40.9), (1.3, 45.6)])
def test_burst_peak_from_20A(R_int_scale, expected):
    """180 s at 25 A, starting from steady state at 20 A, full cooling."""
    p = NOMINAL.scaled(R_int_scale=R_int_scale)
    x0 = steady_state(p, 20.0, 1.0)
    _, peak = simulate(p, [25.0] * 90, 1.0, x0)
    assert peak == pytest.approx(expected, abs=0.2)


# -- scenarios ------------------------------------------------------------------

def test_drive_cycle_bounds_and_determinism():
    a = drive_cycle(np.random.default_rng(1))
    b = drive_cycle(np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (900,) and a.min() >= 0.0 and a.max() <= 25.0


def test_outage_fraction_and_lengths():
    rng = np.random.default_rng(0)
    m = outage_mask(rng, 200_000, 0.2, (30.0, 120.0))
    assert (~m).mean() == pytest.approx(0.2, abs=0.03)
    # lengths of offline runs, in steps
    edges = np.flatnonzero(np.diff(np.r_[1, m.astype(int), 1]))
    lengths = edges[1::2] - edges[::2]
    assert lengths.min() >= 15 and lengths.max() <= 120  # 30-120 s, may merge


def test_episode_shared_across_controllers_and_scenarios():
    a = make_episode(SCENARIOS["nominal"], 3)
    b = make_episode(SCENARIOS["heat_mismatch"], 3)
    np.testing.assert_array_equal(a.I, b.I)
    np.testing.assert_array_equal(a.noise, b.noise)


# -- filters ---------------------------------------------------------------------

def test_kf_steady_sigma():
    kf = KalmanFilter(augment=False)
    for _ in range(3000):
        kf.predict(20.0, 1.0)
        kf.update(kf.x[1])
    assert kf.sigma_Tc == pytest.approx(0.20, abs=0.02)


def run_constant(scenario, augment, u=1.0, seed=0):
    sim = ThermalSim(scenario, KalmanFilter(augment=augment))
    sim.reset(seed)
    sim.ep.I[:] = np.where(np.arange(sim.n_steps) < 300, 15.0, 25.0)
    while not sim.done:
        sim.step(u)
    return sim.hist


def test_akf_recovers_heat_mismatch():
    h = run_constant(SCENARIOS["heat_mismatch"], augment=True)
    assert h["theta"][-1] == pytest.approx(1.3, abs=0.05)
    assert np.abs(h["Tc"][450:] - h["Tc_hat"][450:]).mean() < 0.3


def test_plain_kf_flags_heat_mismatch_in_innovations():
    h = run_constant(SCENARIOS["heat_mismatch"], augment=False)
    nis = np.nanmean(h["nu"][100:] ** 2 / h["S"][100:])
    assert nis > 1.5  # should be about 1 when the model is right


def test_resistance_mismatch_is_invisible_from_surface():
    """Rc error: innovations look normal, yet the core estimate is biased."""
    h = run_constant(SCENARIOS["resistance_mismatch"], augment=True)
    nis = np.nanmean(h["nu"][100:] ** 2 / h["S"][100:])
    assert nis == pytest.approx(1.0, abs=0.2)
    assert (h["Tc"][450:] - h["Tc_hat"][450:]).mean() > 1.5


def test_forecast_matches_repeated_predict():
    kf = KalmanFilter(augment=True)
    kf.predict(18.0, 0.4)
    kf.update(27.0)
    us = np.array([[0.2] * 5, [1.0] * 5])
    mean, sig = kf.forecast(18.0, us)
    for row, u in enumerate([0.2, 1.0]):
        ref = KalmanFilter(augment=True)
        ref.x, ref.P = kf.x.copy(), kf.P.copy()
        for t in range(5):
            ref.predict(18.0, u)
            assert mean[row, t] == pytest.approx(ref.Tc_hat)
            assert sig[row, t] == pytest.approx(ref.sigma_Tc)


def test_outage_skips_updates_and_sigma_grows():
    sim = ThermalSim(replace(SCENARIOS["nominal"], outage_frac=0.0), KalmanFilter(augment=False))
    sim.reset(0)
    for _ in range(300):
        sim.step(1.0)
    s0 = sim.est.sigma_Tc
    sim.ep.online[sim.k + 1 : sim.k + 61] = False  # 2-minute outage
    for _ in range(60):
        sim.step(1.0)
    assert sim.est.sigma_Tc > s0 * 1.3


# -- controllers -------------------------------------------------------------------

def test_backup_check_matches_brute_force():
    from controllers import BackupCheck

    sim = ThermalSim(SCENARIOS["heat_mismatch"], KalmanFilter(augment=True))
    sim.reset(0)
    for _ in range(400):
        sim.step(0.3)
    cands = np.array([0.0, 0.5, 1.0])
    for hold in (1, 5):
        chk = BackupCheck(burst_steps=30, hold=hold)
        for limit in (40.0, 44.0, 50.0):
            chk.limit = limit
            got = chk.ok(sim, cands)
            for i, u in enumerate(cands):
                ref = KalmanFilter(augment=True)
                ref.x, ref.P = sim.est.x.copy(), sim.est.P.copy()
                worst = -np.inf
                for _ in range(hold):
                    ref.predict(sim.I_now, u)
                    worst = max(worst, ref.Tc_hat + 2 * ref.sigma_Tc)
                for _ in range(30):
                    ref.predict(25.0, 1.0)
                    worst = max(worst, ref.Tc_hat + 2 * ref.sigma_Tc)
                assert got[i] == (worst <= limit)


def test_shield_returns_rl_action_when_safe_and_full_cooling_when_not():
    from controllers import Shield

    sim = ThermalSim(SCENARIOS["nominal"], KalmanFilter(augment=True))
    sim.reset(0)
    sim.ep.I[:] = 5.0
    for _ in range(100):
        sim.step(0.0)
    sh = Shield()
    assert sh.safe_action(sim, 0.0) == 0.0  # cool cell, low current: no intervention
    sim.est.x[0] = 46.0  # pretend the core is already too hot
    assert sh.safe_action(sim, 0.0) == 1.0


# -- RL environment ------------------------------------------------------------------

gym = pytest.importorskip("gymnasium")


@pytest.mark.parametrize("mode, dim", [("full", 7), ("no_sigma", 6), ("raw", 22)])
def test_env_episode_shape_and_truncation(mode, dim):
    from thermal_env import ThermalEnv

    env = ThermalEnv(obs_mode=mode, dr=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (dim,) and obs.dtype == np.float32
    steps, done = 0, False
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        steps += 1
        done = term or trunc
        assert not term and np.isfinite(r) and r <= 0.0
    assert steps == 900 and "episode_stats" in info


def test_env_reward_matches_definition():
    from thermal_env import ThermalEnv

    env = ThermalEnv(scenario=SCENARIOS["heat_mismatch"])
    env.reset(seed=3)
    u_prev = 0.0
    for a in np.linspace(-1, 1, 50):
        _, r, _, _, info = env.step(np.array([a], dtype=np.float32))
        u = (a + 1) / 2
        expected = -u**3 - 10 * info["violation"] - 0.01 * abs(u - u_prev)
        assert r == pytest.approx(expected, abs=1e-6)
        u_prev = u


@pytest.mark.parametrize("mode", ["full", "no_sigma", "raw"])
def test_training_and_evaluation_see_the_same_observation(mode):
    """The env (training) and ObsBuilder on a plain ThermalSim (evaluation) must agree."""
    from thermal_env import ObsBuilder, ThermalEnv

    env = ThermalEnv(obs_mode=mode, scenario=SCENARIOS["outage"])
    obs_env, _ = env.reset(seed=0, options={"episode_seed": 7})
    sim = ThermalSim(SCENARIOS["outage"], KalmanFilter(augment=True))
    sim.reset(7)
    ob = ObsBuilder(mode)
    rng = np.random.default_rng(0)
    for _ in range(300):
        np.testing.assert_allclose(obs_env, ob(sim), rtol=0, atol=1e-6)
        u = rng.random()
        obs_env, *_ = env.step(np.array([2 * u - 1], dtype=np.float32))
        sim.step(u)


def test_true_core_temperature_is_not_observed():
    """Changing only the true core temperature must not change the observation."""
    from thermal_env import ObsBuilder

    sim = ThermalSim(SCENARIOS["nominal"], KalmanFilter(augment=True))
    sim.reset(0)
    for _ in range(50):
        sim.step(0.5)
    for mode in ("full", "no_sigma"):
        a = ObsBuilder(mode)(sim)
        sim.x[0] += 5.0
        b = ObsBuilder(mode)(sim)
        sim.x[0] -= 5.0
        np.testing.assert_array_equal(a, b)
