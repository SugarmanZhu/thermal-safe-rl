# Safe RL for Battery Thermal Management under Partial Observability

A small demo of physics-informed state estimation and safe adaptive control for thermal management, using a battery cell as the example.

**Problem.** Keep a battery cell's **core temperature ≤ 45 °C** with as little cooling energy as possible. Only the **surface** temperature is measured, with noise and occasional sensor outages, and the model may be wrong. The demo touches four research questions:
- **RQ1:** estimating the unmeasured core temperature with calibrated uncertainty
- **RQ2:** safe RL with an uncertainty-aware safety shield
- **RQ3:** whether combining estimation and control helps
- **RQ4:** robustness to model error, sensor loss and conditions beyond training

> **Status:** the physics model, both Kalman filters, the estimation study (RQ1), and the non-RL controllers (PID, safety shield, MPC) with their full evaluation are done. RL training comes next.

## Model

Two-state lumped thermal model of a cylindrical cell:

```
Cc · dTc/dt = I²·R_int − (Tc − Ts)/Rc
Cs · dTs/dt = (Tc − Ts)/Rc − (Ts − Tf)·(h0 + h1·u)
```

- Tc is the core temperature (not measured), Ts the surface temperature (measured, σ = 0.3 °C), and u ∈ [0, 1] the coolant pump command.
- Steady state at 25 A is 43.4 °C with full cooling and 78.8 °C with none, so cooling is necessary, but full cooling all the time wastes energy.
- Each 2 s step is discretised exactly with the closed-form 2×2 matrix exponential.

## RQ1: what a surface sensor can and cannot tell you

There are two filters, and both use the **nominal** model:
- **KF:** state [Tc, Ts].
- **AKF:** state [Tc, Ts, θ], where θ scales the heat generation (Q = θ·I²·R_int) and drifts slowly. This lets the filter learn "the cell makes more heat than the model thinks".

![Estimation](results/estimation.png)

![Calibration](results/calibration.png)

Each scenario was run for 20 seeds with full cooling, so that only the estimator differs. Full numbers are in [results/estimation.md](results/estimation.md). In the table, bias = estimate − true.

| Scenario | Finding |
|---|---|
| Heat mismatch (R_int ×1.3) | **Visible and fixable.** The plain KF reads 0.48 °C low and its 2σ band covers the truth only 43 % of the time. Its innovations reveal the problem without a core sensor: NIS 1.29 instead of ≈1, mean innovation 0.52 instead of ≈0. The AKF estimates θ = 1.31 and is back to 0.14 °C RMSE with full coverage. |
| Resistance mismatch (Rc ×1.2) | **Invisible.** Both filters read about 1.0 °C low and cover the truth only 4 % of the time, yet the innovations look perfectly normal (NIS 0.94). At steady state Ts − Tf = Q/h whatever Rc is, so the surface sensor carries no information about Rc. A safety layer that trusts σ will be overconfident here. This motivates active probing (dual control) as future work. |
| Noisy sensor (×3) | The filters assume σ = 0.3 °C, and NIS jumps to 8.4, so the wrong noise assumption is detectable. Coverage drops to 71–75 %. |
| Ageing (R_int ×1.0 → ×1.4) | The KF drifts (64 % coverage). The AKF tracks it (θ̂ = 1.32 at the end, lagging the true 1.4; 99 % coverage). |
| Outage (sensor offline 30–120 s, ~20 % of the time) | Both filters stay accurate with full cooling. Outages matter more once the controller relies on the estimate. |
| Nominal | Accurate (0.12–0.14 °C RMSE), but **conservative**: NEES is 0.4 where a calibrated filter gives 2, so σ is wider than the real error. The process noise (0.05 °C per step) is a tuning choice; a smaller value would tighten σ but make the mismatch cases even more overconfident. |

## Controllers without RL (RQ2, RQ4 baselines)

All controllers act on the AKF estimate, except the oracle PID, which sees the true core temperature (impossible in practice; a reference only).

| Controller | Rule |
|---|---|
| Always on | u = 1 all the time. The physical floor for violations. |
| PID | On the core estimate, setpoint 42 °C. The gains come from a grid search on separate tuning seeds (1000–1019) using the RL reward's weights; the search chose a proportional controller (kp = 0.3, ki = kd = 0). |
| Oracle PID | The same PID, but fed the true core temperature. |
| Shield only, v1 | The cheapest u that keeps the predicted core mean + 2σ ≤ 44 °C over the next 30 s, assuming the current stays constant. This is the original plan's shield. |
| **Shield only** | v1 plus a **backup check**: hold u for 30 s, then survive a worst-case burst (25 A for 180 s, the drive-cycle maximum) with full cooling, still with mean + 2σ ≤ 44 °C. It uses the AKF's heat-scale estimate θ, so a cell that runs hot automatically gets a larger reserve. |
| MPC | Plans 60 s ahead (216 candidate plans, 3 blocks of 20 s), minimising Σu³·dt subject to the same safety rule as the shield. |

![Energy vs safety](results/pareto.png)

20 evaluation seeds per cell (full tables: [results/controllers.md](results/controllers.md)):

| Scenario | Finding |
|---|---|
| Nominal, noisy, outage | **Everyone is safe.** PID is cheapest (31 pump-s); the backup shield and MPC use about 8× more (237–245) as insurance against bursts that never cause trouble here. |
| Heat mismatch (R_int ×1.3) | PID **and the oracle PID** exceed 45 °C in 85 % of episodes: knowing the true core temperature doesn't help a controller that only reacts. The v1 shield fails in 40 %. **The backup shield: 0 %**, using 18 % less energy than always-on (1471 vs 1800). |
| Ageing (R_int → ×1.4) | The backup shield's only violating episode (seed 17) is the one where even always-on exceeds 45 °C, though the shield exceeds by more (6.8 vs 0.03 degree-seconds). PID: 75 % of episodes. |
| Resistance mismatch (Rc ×1.2) | **Every controller that uses the estimate fails** (60–80 % of episodes), but the oracle PID does not (5 %). The estimate itself is wrong (RQ1), and no downstream safety logic can fix that. |
| Long bursts (360 s, beyond the shield's 180 s assumption) | No violations by any controller on the nominal plant. |

**How the shield design changed during evaluation.** These changes were made on the tuning seeds and then confirmed on the evaluation seeds:
1. The v1 shield failed under heat mismatch even with an accurate estimate (error about 0.1 °C, θ̂ = 1.30). A burst arrived while the core was already warm, and from there even full cooling could not keep it below 45 °C. A 30 s look-ahead with constant current cannot see this coming. Hence the backup check.
2. With a one-step hold, the backup check made the command chatter between low and full cooling. Holding the candidate for 30 s gave the same safety with 25 % less energy and far fewer on/off switches.

**The price of robustness.** The assumed worst-case burst length is a dial between energy and safety. There is no free lunch:

![Shield trade-off](results/shield_tradeoff.png)

This is where learning could help: a policy that anticipates load, or a load forecast, could keep the same safety at lower cost than "assume the worst at every moment". It also explains why PID beats the shield on energy in the nominal case. Under a u³ cost, gentle early cooling is cheaper than late hard cooling, and the shield and MPC cool late and hard. One episode per controller: [nominal](results/trajectory_nominal.png), [resistance mismatch](results/trajectory_resistance_mismatch.png).

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q                 # 19 checks: physics, discretisation, scenarios, filters, shield
python evaluate.py estimation       # RQ1: results/estimation.*, results/calibration.png (~30 s)
python evaluate.py tune-pid         # PID gain search on tuning seeds (~30 s on 4 cores)
python evaluate.py shield-sweep     # results/shield_tradeoff.* (~1 min on 4 cores)
python run_all.py                   # train 5 PPO variants x 3 seeds (~30-40 min on 4 CPU cores)
python evaluate.py training         # training-time safety: results/training_safety.*
python evaluate.py controllers      # all controllers incl. PPO: results/controllers.*, pareto.png, rq3_ablation.png
```

No GPU is needed. On free cloud machines, use `notebooks/colab_debug.ipynb` for a quick test and `notebooks/kaggle_train.ipynb` for the full run (Accelerator *None*, then *Save & Run All*, which keeps running after you close the browser).

## Code

| File | Contents |
|---|---|
| `thermal_env.py` | Cell parameters, exact discretisation, drive cycles, scenarios, `ThermalSim` (plant + estimator for one episode), `ObsBuilder` and the Gymnasium `ThermalEnv` |
| `estimators.py` | `KalmanFilter(augment=False/True)`: KF and AKF, plus batch open-loop forecasts for the shield and MPC |
| `controllers.py` | Always-on, PID / oracle PID, `Shield` with `BackupCheck`, shield-only, MPC, `PPOController` |
| `train.py` | One PPO run: variants, checkpoints every 50k steps, `--resume`, training-time safety log |
| `run_all.py` | All variants × seeds in parallel; skips finished runs, resumes unfinished ones, `--max-hours` |
| `evaluate.py` | Estimation study, PID tuning, controller comparison (incl. PPO), shield sweep, training safety |
| `notebooks/` | Colab (debug) and Kaggle (full training) notebooks |
| `tests/test_physics.py` | Unit tests: physics, filters, shield, and the RL environment (incl. train/eval observation match) |

Seeds: evaluation uses seeds 0–19; tuning (PID gains, shield design choices) used seeds 1000–1019.

Parameter values for Cc, Cs and Rc follow a 26650 cylindrical cell from the lumped thermal-model literature (exact reference to be added). The cooling map h(u) is an assumption of this demo.

**Known limitations so far:** a lumped two-state model; an assumed cooling map; the shield and MPC assume the current stays constant (apart from the worst-case burst in the backup check); the shield still switches more often than a real pump would like; and the filter's σ is conservative in the nominal case (NEES 0.4).
