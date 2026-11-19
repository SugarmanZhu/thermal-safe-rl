"""Train every PPO variant x seed, several at once, and survive interruptions.

    python run_all.py                                  # everything, default 1M steps, 3 seeds
    python run_all.py --variants ppo_dr --seeds 0 1 2  # only the must-have runs
    python run_all.py --out /kaggle/working/thermal-safe-rl/runs --resume-from /kaggle/input/<prev>/thermal-safe-rl/runs

- Finished runs (final/ with enough steps) are skipped.
- Unfinished runs continue from their latest checkpoint.
- --resume-from copies runs from a previous session (e.g. an earlier Kaggle version's
  output) before starting, so a second session picks up where the first stopped.
- --max-hours stops launching new runs after that time (running ones finish), so a 12 h
  Kaggle session ends cleanly.
Each run's console output goes to <out>/<variant>/seed<k>/train.out.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from train import DEFAULT_TIMESTEPS, VARIANTS, is_finished, run_dir

ORDER = ["ppo_dr", "ppo", "ppo_dr_shieldtrain", "ppo_dr_shieldtrain_p1", "ppo_dr_nosigma",
         "ppo_dr_raw"]  # must-have first


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs")
    ap.add_argument("--variants", nargs="+", default=ORDER, choices=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--resume-from", type=Path, default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    args = ap.parse_args()

    if args.resume_from and args.resume_from.exists():
        shutil.copytree(args.resume_from, args.out, dirs_exist_ok=True)
        print(f"copied previous runs from {args.resume_from}")

    todo = [(v, s) for v in sorted(args.variants, key=ORDER.index) for s in args.seeds
            if not is_finished(run_dir(args.out, v, s), args.timesteps)]
    done = len(args.variants) * len(args.seeds) - len(todo)
    print(f"{done} runs already finished, {len(todo)} to go, {args.jobs} at a time")
    t0 = time.time()

    def launch(job):
        v, s = job
        if args.max_hours is not None and (time.time() - t0) / 3600 > args.max_hours:
            return v, s, "skipped (time limit)"
        rd = run_dir(args.out, v, s)
        rd.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).with_name("train.py")), "--variant", v, "--seed", str(s),
               "--timesteps", str(args.timesteps), "--out", str(args.out), "--resume"]
        start = time.time()
        with open(rd / "train.out", "a") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        status = "ok" if code == 0 else f"FAILED (exit {code}, see {rd / 'train.out'})"
        print(f"  {v} seed {s}: {status} in {(time.time() - start) / 60:.1f} min", flush=True)
        return v, s, status

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(launch, todo))
    failed = [r for r in results if r[2] != "ok"]
    print(f"done in {(time.time() - t0) / 60:.1f} min; {len(results) - len(failed)} ok, {len(failed)} not finished")
    for v, s, st in failed:
        print(f"  {v} seed {s}: {st}")
    sys.exit(1 if any("FAILED" in r[2] for r in failed) else 0)


if __name__ == "__main__":
    main()
