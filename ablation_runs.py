#!/usr/bin/env python3
"""Generate the adversarial images of the ablation study: every variant x seed, 100 pairs.

    python ablation_runs.py                 # all 14 variants x seeds 0,1,2 on GPU 0
    python ablation_runs.py --gpus 0,1,2,3  # four GPUs: each takes the next run when it is free
    python ablation_runs.py --only goc,foa_T300 --seeds 0
    python ablation_runs.py --only foa_T1050:2,fix05_M3:0+1   # a variant with its own seeds
    python ablation_runs.py --list          # show the queue and exit

The screen shows progress bars only: runs done, and for each GPU its run, the images done and the step of
the current image. Everything the attack scripts print goes to <out>/work/<run>/run.log.

Output, one flat folder per finished run:
    <out>/images/<variant>_s<seed>/<index>.png

Re-running the same command skips finished runs and resumes the interrupted one.
Next step: python evaluate.py
"""

import argparse
import glob
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))

SOT = "SOTAttack.py"
FOA = "generate_adversarial_samples_foa_attack.py"

# name -> (script, extra Hydra overrides, description). Order = priority.
VARIANTS = {
    # Group 1: adaptive lambda (b) and encoder re-weighting (c), M = 5
    "goc":           (SOT, [], "SOTAttack: adaptive lambda, adaptive w"),
    "fixlam05":      (SOT, ["optim.mix=fixed", "optim.fixed_lambda=0.5"], "lambda = 0.5"),
    "fixlam03":      (SOT, ["optim.mix=fixed", "optim.fixed_lambda=0.3"], "lambda = 0.3"),
    "multionly":     (SOT, ["optim.mix=multi_only"], "lambda = 0 (extra views only)"),
    "uniw":          (SOT, ["optim.enc_weight=uniform"], "adaptive lambda, uniform w"),
    "fixlam05_uniw": (SOT, ["optim.mix=fixed", "optim.fixed_lambda=0.5", "optim.enc_weight=uniform"],
                      "lambda = 0.5, uniform w"),
    # Group 2: FOA-Attack at equal steps (T = 300) and equal compute (T = 1050)
    "foa_T300":      (FOA, [], "FOA-Attack, T = 300"),
    "foa_T1050":     (FOA, ["optim.steps=1050"], "FOA-Attack, T = 1050"),
    # Group 3: number of extra views (M = 5 is 'goc')
    "M1":            (SOT, ["optim.num_crops=1"], "SOTAttack, M = 1"),
    "M3":            (SOT, ["optim.num_crops=3"], "SOTAttack, M = 3"),
    "M7":            (SOT, ["optim.num_crops=7"], "SOTAttack, M = 7"),
    # Group 4: number of extra views with lambda fixed at 0.5 (M = 5 is 'fixlam05'); here the extra views
    # carry real weight, unlike under adaptive lambda where 1 - lambda is ~0.015.
    "fix05_M1":      (SOT, ["optim.mix=fixed", "optim.fixed_lambda=0.5", "optim.num_crops=1"], "lambda = 0.5, M = 1"),
    "fix05_M3":      (SOT, ["optim.mix=fixed", "optim.fixed_lambda=0.5", "optim.num_crops=3"], "lambda = 0.5, M = 3"),
    "fix05_M7":      (SOT, ["optim.mix=fixed", "optim.fixed_lambda=0.5", "optim.num_crops=7"], "lambda = 0.5, M = 7"),
}
# Not in the default queue; run with --only.
EXTRA_VARIANTS = {
    "baseonly":      (SOT, ["optim.mix=base_only"], "lambda = 1 inside SOTAttack"),
    # Same total compute as FOA-Attack T = 1050 and as M = 5, T = 300 (6300 encoder-view passes per image),
    # spent on more steps instead of more views: 3 * (M + 2) * T = 6300.
    "fix05_M1_T700": (SOT, ["optim.mix=fixed", "optim.fixed_lambda=0.5", "optim.num_crops=1", "optim.steps=700"],
                      "lambda = 0.5, M = 1, T = 700"),
    "fix05_M3_T420": (SOT, ["optim.mix=fixed", "optim.fixed_lambda=0.5", "optim.num_crops=3", "optim.steps=420"],
                      "lambda = 0.5, M = 3, T = 420"),
    # 'fixlam05' with the views of a step batched through the encoders: same method, measures wall-clock time.
    "fix05_bv":      (SOT, ["optim.mix=fixed", "optim.fixed_lambda=0.5", "optim.batch_views=true"],
                      "lambda = 0.5, views batched"),
}
ALL_VARIANTS = {**VARIANTS, **EXTRA_VARIANTS}

COMMON = [
    "--config-name=ensemble_3models",
    "attack=fgsm",
    "data.cle_data_path=resources/images/bigscale_100",
    "data.tgt_data_path=resources/images/target_images_100",
    "data.batch_size=1",
    "optim.alpha=1.0",
    "optim.epsilon=16",
    "optim.steps=300",
    "optim.num_crops=5",
]


def run_name(variant, seed):
    return f"{variant}_s{seed}"


def count_pngs(folder):
    return len(glob.glob(os.path.join(folder, "**", "*.png"), recursive=True))


def build_cmd(variant, seed, work_dir, device, num, extra=()):
    script, overrides, _ = ALL_VARIANTS[variant]
    return [
        sys.executable, script, *COMMON, *overrides,
        f"seed={seed}", f"data.num_samples={num}", f"model.device={device}",
        f"data.output={work_dir}", f"hydra.run.dir={os.path.join(work_dir, 'hydra')}",
        *extra,
    ]


def child_env(gpu):
    env = dict(os.environ)
    env["TQDM_DISABLE"] = "1"       # per-step and k-means bars would flood the log
    env["WANDB_MODE"] = "disabled"
    env["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


STOP = threading.Event()


def run_attack(cmd, work_dir, env, bar=None):
    """Run one attack subprocess, logging to work_dir/run.log. Returns the exit code."""
    os.makedirs(work_dir, exist_ok=True)
    img_dir = os.path.join(work_dir, "img")
    progress = os.path.join(work_dir, "progress")
    env = dict(env, ABL_PROGRESS=progress)
    if os.path.exists(progress):
        os.remove(progress)  # left over from an interrupted run

    def show():
        if bar is None:
            return
        bar.n = min(count_pngs(img_dir), bar.total)
        try:
            with open(progress) as f:
                step = f.read().strip()
        except OSError:
            step = ""
        if step:
            bar.set_postfix_str(f"step {step}", refresh=False)
        bar.refresh()

    with open(os.path.join(work_dir, "run.log"), "ab") as log:
        log.write(f"\n===== {time.strftime('%F %T')} | {' '.join(cmd)}\n".encode())
        log.flush()
        proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            if STOP.wait(2):
                proc.terminate()
                proc.wait()
                break
            show()
    show()
    return proc.returncode


def run_pool(jobs, slots, device, main_bar, image_bars):
    """Each slot (one GPU) takes the next job from a shared queue until it is empty.

    jobs: dicts with name, variant, seed, work, num, extra, final. Returns the failed jobs
    as (name, exit code, images produced, log path).
    """
    todo = queue.Queue()
    for job in jobs:
        todo.put(job)
    failures, lock = [], threading.Lock()

    def worker(slot, gpu):
        env = child_env(gpu)
        tag = "cpu" if gpu is None else f"gpu{gpu}"
        bar = tqdm(total=1, unit="img", position=slot + 1, leave=False) if image_bars else None
        while not STOP.is_set():
            try:
                job = todo.get_nowait()
            except queue.Empty:
                break
            if bar is not None:
                bar.reset(total=job["num"])
                bar.set_postfix_str("loading models", refresh=False)
                bar.set_description_str(f"{tag} {job['name']:<18}")
            cmd = build_cmd(job["variant"], job["seed"], job["work"], device, job["num"], job["extra"])
            code = run_attack(cmd, job["work"], env, bar)
            if STOP.is_set():
                break
            produced = count_pngs(os.path.join(job["work"], "img"))
            if code == 0 and produced == job["num"]:
                if job["final"]:
                    publish(job["work"], job["final"])
            else:
                with lock:
                    failures.append((job["name"], code, produced, os.path.join(job["work"], "run.log")))
                tqdm.write(f"FAILED {job['name']}: exit {code}, {produced}/{job['num']} images, "
                           f"log {os.path.join(job['work'], 'run.log')}")
            with lock:
                main_bar.update(1)
        if bar is not None:
            bar.close()

    threads = [threading.Thread(target=worker, args=(i, gpu), daemon=True) for i, gpu in enumerate(slots)]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)  # keeps the main thread interruptible
    except KeyboardInterrupt:
        STOP.set()
        for t in threads:
            t.join()
        raise
    return failures


def publish(work_dir, final_dir):
    """Copy the run's PNGs to a flat folder; appear atomically so a partial copy never counts."""
    tmp = final_dir + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for path in glob.glob(os.path.join(work_dir, "img", "**", "*.png"), recursive=True):
        shutil.copy2(path, os.path.join(tmp, os.path.basename(path)))
    shutil.rmtree(final_dir, ignore_errors=True)
    os.rename(tmp, final_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="ablation_out", help="output root")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--only", default=None,
                    help="comma-separated variants, each optionally with its own seeds: name or name:0+1 "
                         "(default: all variants x --seeds)")
    ap.add_argument("--num", type=int, default=100, help="image pairs per run")
    ap.add_argument("--gpus", default="0", help="comma-separated GPU indices; runs are shared out dynamically")
    ap.add_argument("--device", default=None, help="torch device override, e.g. cpu (default: the --gpus cards)")
    ap.add_argument("--shard", default="0/1", help="i/n: take every n-th run (only for several machines)")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the queue and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    ap.add_argument("overrides", nargs="*", help="extra Hydra overrides for every run (testing), e.g. optim.steps=2")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    wanted = {}   # variant -> its seeds
    for item in ([x.strip() for x in args.only.split(",")] if args.only else list(VARIANTS)):
        name, _, own = item.partition(":")
        wanted[name] = [int(s) for s in own.split("+")] if own else seeds
    names = list(wanted)
    unknown = [v for v in names if v not in ALL_VARIANTS]
    if unknown:
        ap.error(f"unknown variant(s) {unknown}; choose from {list(ALL_VARIANTS)}")
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    if not 0 <= shard_i < shard_n:
        ap.error("--shard i/n needs 0 <= i < n")

    # Default: seed outermost, so every variant gets its first seed before any gets its second.
    # With --only the order is the one given (list the longest run first so it starts at once).
    if args.only:
        queue = [(v, s) for v in names for s in wanted[v]]
    else:
        queue = [(v, s) for s in seeds for v in names]
    queue = queue[shard_i::shard_n]
    out = os.path.abspath(os.path.join(HERE, args.out)) if not os.path.isabs(args.out) else args.out
    images_root, work_root = os.path.join(out, "images"), os.path.join(out, "work")
    device = args.device or "cuda:0"   # each child sees only its own card, as cuda:0
    slots = [None] if args.device else [int(g) for g in args.gpus.split(",")]

    def is_done(v, s):
        return count_pngs(os.path.join(images_root, run_name(v, s))) >= args.num

    if args.list or args.dry_run:
        for v, s in queue:
            state = "done" if is_done(v, s) else "todo"
            print(f"[{state}] {run_name(v, s):<22} {ALL_VARIANTS[v][2]}")
            if args.dry_run:
                print("   " + " ".join(build_cmd(v, s, os.path.join(work_root, run_name(v, s)), device, args.num, args.overrides)))
        print(f"{len(queue)} runs, {sum(is_done(v, s) for v, s in queue)} done -> {images_root}")
        return 0

    if not args.device:
        probe = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
                               capture_output=True, text=True)
        count = int(probe.stdout.strip() or 0) if probe.returncode == 0 else 0
        if count == 0 or max(slots) >= count:
            print(f"--gpus {args.gpus} asked for, but torch sees {count} CUDA device(s). "
                  f"Fix the driver/torch build, adjust --gpus, or pass --device cpu.")
            return 1

    todo = [(v, s) for v, s in queue if not is_done(v, s)]

    # Smoke test: one image, two steps, once per variant still to run.
    if not args.skip_smoke and todo:
        smoke_root = os.path.join(work_root, "_smoke")
        shutil.rmtree(smoke_root, ignore_errors=True)
        jobs = [dict(name=v, variant=v, seed=0, work=os.path.join(smoke_root, v), num=1, final=None,
                     extra=[*args.overrides, "data.num_samples=1", "optim.steps=2"])
                for v in dict.fromkeys(v for v, _ in todo)]
        with tqdm(total=len(jobs), desc="smoke", unit="variant", position=0) as bar:
            failed = run_pool(jobs, slots, device, bar, image_bars=True)
        if failed:
            print("Smoke test failed, nothing was started: " + ", ".join(f[0] for f in failed))
            return 1
        shutil.rmtree(smoke_root, ignore_errors=True)

    jobs = [dict(name=run_name(v, s), variant=v, seed=s, work=os.path.join(work_root, run_name(v, s)),
                 num=args.num, final=os.path.join(images_root, run_name(v, s)), extra=args.overrides)
            for v, s in todo]
    with tqdm(total=len(queue), initial=len(queue) - len(todo), desc="runs", unit="run", position=0) as runs_bar:
        failures = [f[0] for f in run_pool(jobs, slots, device, runs_bar, image_bars=True)]

    done = sum(is_done(v, s) for v, s in queue)
    print(f"{done}/{len(queue)} runs complete in {images_root}"
          + (f"; failed: {', '.join(failures)} (re-run the same command to retry)" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.")
        sys.exit(130)
