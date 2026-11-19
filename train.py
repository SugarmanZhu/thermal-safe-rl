"""Train one PPO policy.

    python train.py --variant ppo_dr --seed 0                      # full run (1M steps)
    python train.py --variant ppo_dr --seed 0 --timesteps 20000    # smoke test
    python train.py --variant ppo_dr --seed 0 --resume             # continue from the latest checkpoint

Output in <out>/<variant>/seed<k>/:
    checkpoints/      model + VecNormalize every 50k steps (not committed to git)
    final/            model.zip, vecnormalize.pkl, config.json (committed)
    train_log.csv     per-episode energy, degree-seconds above 45 degC, shield rate:
                      the training-time safety record ("unsafe exploration")
CPU only: the policy is a small MLP and the environment is numpy, so a GPU does not help.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

VARIANTS = {
    # name: environment options
    "ppo_dr": dict(obs_mode="full", dr=True, shield=False),
    "ppo": dict(obs_mode="full", dr=False, shield=False),
    "ppo_dr_shieldtrain": dict(obs_mode="full", dr=True, shield=True),  # intervention penalty 0.1
    "ppo_dr_shieldtrain_p1": dict(obs_mode="full", dr=True, shield=True, intervention_penalty=1.0),
    "ppo_dr_nosigma": dict(obs_mode="no_sigma", dr=True, shield=False),
    "ppo_dr_raw": dict(obs_mode="raw", dr=True, shield=False),
}
PPO_KWARGS = dict(gamma=0.995, n_steps=1024, batch_size=256, n_epochs=10, learning_rate=3e-4,
                  ent_coef=0.0, policy_kwargs=dict(net_arch=[64, 64]))
DEFAULT_TIMESTEPS = 1_000_000
CHECKPOINT_EVERY = 50_000


def run_dir(out: Path, variant: str, seed: int) -> Path:
    return Path(out) / variant / f"seed{seed}"


def latest_checkpoint(rd: Path):
    """Return (model_path, vecnormalize_path, steps) of the newest complete save: a
    periodic checkpoint or the final/ folder, whichever has more steps."""
    best = None
    for m in (rd / "checkpoints").glob("ckpt_*_steps.zip"):
        steps = int(re.search(r"ckpt_(\d+)_steps", m.name).group(1))
        vn = m.parent / f"ckpt_vecnormalize_{steps}_steps.pkl"
        if vn.exists() and (best is None or steps > best[2]):
            best = (m, vn, steps)
    final = rd / "final"
    if (final / "model.zip").exists() and (final / "vecnormalize.pkl").exists() and (final / "config.json").exists():
        steps = json.loads((final / "config.json").read_text())["timesteps"]
        if best is None or steps > best[2]:
            best = (final / "model.zip", final / "vecnormalize.pkl", steps)
    return best


def is_finished(rd: Path, timesteps: int) -> bool:
    cfg = rd / "final" / "config.json"
    return cfg.exists() and json.loads(cfg.read_text())["timesteps"] >= timesteps


def _truncate_log(log_path: Path, max_steps: int) -> None:
    """After resuming from a checkpoint, drop log rows written after it (they will be redone)."""
    if not log_path.exists():
        return
    rows = list(csv.reader(open(log_path, newline="")))
    keep = rows[:1] + [r for r in rows[1:] if int(r[0]) <= max_steps]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerows(keep)


def make_venv(variant: str, seed: int, n_envs: int):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv

    from thermal_env import ThermalEnv

    opts = VARIANTS[variant]
    return make_vec_env(ThermalEnv, n_envs=n_envs, seed=seed, env_kwargs=opts, vec_env_cls=DummyVecEnv)


def train(variant: str, seed: int, timesteps: int, out: Path, n_envs: int = 8, resume: bool = False) -> Path:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import VecNormalize

    torch.set_num_threads(1)  # one core per run; run_all.py runs several in parallel
    rd = run_dir(out, variant, seed)
    ckpt_dir, final_dir = rd / "checkpoints", rd / "final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(seed)
    venv = make_venv(variant, seed, n_envs)

    log_path = rd / "train_log.csv"
    ckpt = latest_checkpoint(rd) if resume else None
    if ckpt is not None:
        venv = VecNormalize.load(str(ckpt[1]), venv)
        venv.training = True
        model = PPO.load(str(ckpt[0]), env=venv, device="cpu")
        _truncate_log(log_path, ckpt[2])
        print(f"resuming {variant} seed {seed} from {ckpt[2]} steps")
    else:
        if log_path.exists():
            log_path.unlink()  # fresh start: old log belongs to a discarded run
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=PPO_KWARGS["gamma"])
        tb = rd / "tb" if _has_tensorboard() else None
        model = PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0,
                    tensorboard_log=str(tb) if tb else None, **PPO_KWARGS)

    class SafetyLog(BaseCallback):
        """Write one row per finished training episode."""

        def _on_training_start(self) -> None:
            new = not log_path.exists()
            self._f = open(log_path, "a", newline="", buffering=1)  # line-buffered: survives a kill
            self._w = csv.writer(self._f)
            if new:
                self._w.writerow(["timesteps", "energy", "deg_s", "max_Tc", "shield_rate",
                                  "R_int_scale", "Rc_scale"])

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                st = info.get("episode_stats")
                if st:
                    self._w.writerow([self.num_timesteps, round(st["energy"], 3), round(st["deg_s"], 4),
                                      round(st["max_Tc"], 3), round(st["shield_rate"], 2),
                                      round(st["R_int_scale"], 4), round(st["Rc_scale"], 4)])
            return True

        def _on_training_end(self) -> None:
            self._f.close()

    remaining = timesteps - model.num_timesteps
    if remaining > 0:
        callbacks = CallbackList([
            CheckpointCallback(save_freq=max(1, CHECKPOINT_EVERY // n_envs), save_path=str(ckpt_dir),
                               name_prefix="ckpt", save_vecnormalize=True),
            SafetyLog(),
        ])
        t0 = time.time()
        model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=ckpt is None,
                    tb_log_name="ppo")  # with reset_num_timesteps=False, SB3 counts on from the checkpoint
        print(f"{variant} seed {seed}: {remaining} steps in {time.time() - t0:.0f} s")

    final_dir.mkdir(exist_ok=True)
    model.save(final_dir / "model.zip")
    venv.save(str(final_dir / "vecnormalize.pkl"))
    (final_dir / "config.json").write_text(json.dumps(
        {"variant": variant, "seed": seed, "timesteps": int(model.num_timesteps), "n_envs": n_envs,
         "env": VARIANTS[variant], "ppo": {k: v for k, v in PPO_KWARGS.items()}}, indent=2, default=str))
    return final_dir


def _has_tensorboard() -> bool:
    import importlib.util

    return importlib.util.find_spec("tensorboard") is not None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=list(VARIANTS), default="ppo_dr")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint if any")
    args = ap.parse_args()
    final = train(args.variant, args.seed, args.timesteps, args.out, args.n_envs, args.resume)
    print(f"saved {final}")


if __name__ == "__main__":
    main()
