#!/usr/bin/env python3
"""Run SOTAttack for several (attack, seed) variants back to back, unattended.

Every variant is smoke-tested on a single batch first, so a broken configuration
fails within minutes instead of hours into the real run. Only if all smoke tests
pass does the full generation start.

Each variant runs as its own subprocess: when it exits the GPU memory is returned
to the driver, so one variant can never leak memory into the next one.

Results are laid out one folder per variant:

    <root>/<attack>/seed<label>/img/<run_name>/...
    <root>/<attack>/seed<label>/run.log

e.g. remain/mifgsm/seed2026/ and remain/mifgsm/seeddefault/.

Typical use:

    python run_variants.py                      # mifgsm, seeds: default + 2026
    python run_variants.py --smoke-only         # only the pre-flight check
    python run_variants.py --dry-run            # print the commands, run nothing

A run that dies halfway can simply be restarted with the same command: SOTAttack
resumes from the images already present in that variant's own folder.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SOT_SCRIPT = os.path.join(HERE, "SOTAttack.py")


def seed_label(seed):
    """Folder-safe label for a seed; must match get_seed_label() in utils.py."""
    return "default" if seed is None else str(seed)


def parse_seed(text):
    """Parse one --seeds entry. 'default'/'none'/'null' means: leave PyTorch unseeded."""
    token = text.strip().lower()
    if token in ("default", "none", "null"):
        return None
    try:
        return int(token)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid seed '{text}'. Use an integer, or 'default' for an unseeded run."
        )


def read_config_value(config_name, *keys, default=None):
    """Read a nested value out of config/<config_name>.yaml."""
    path = os.path.join(HERE, "config", f"{config_name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    node = yaml.safe_load(open(path, encoding="utf-8"))
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def count_pngs(root):
    """Count .png files anywhere under root."""
    total = 0
    for _, _, files in os.walk(root):
        total += sum(1 for f in files if f.lower().endswith(".png"))
    return total


def fmt_duration(seconds):
    return str(timedelta(seconds=int(seconds)))


def print_gpu_memory(prefix="  "):
    """Show GPU memory after a run so a leak between variants is visible."""
    if shutil.which("nvidia-smi") is None:
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return
    if out.returncode != 0:
        return
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            print(f"{prefix}GPU {parts[0]}: {parts[1]} / {parts[2]} MiB in use")


def resolve_configured_device(config_name, overrides):
    """The device the runs will actually ask for, after any override."""
    device = read_config_value(config_name, "model", "device", default="cuda:0")
    for override in overrides:
        if override.startswith("model.device="):
            device = override.split("=", 1)[1]
    return str(device)


def preflight_device(device):
    """Confirm CUDA is really usable before committing to a long queue.

    SOTAttack falls back to CPU when CUDA is missing, so without this check a
    smoke test would quietly pass on CPU and the real run would crawl for days.
    """
    if "cuda" not in device:
        print(f"  Device: {device} (no CUDA requested)")
        return True

    probe = (
        "import json, torch\n"
        "info = {'available': torch.cuda.is_available(),\n"
        "        'count': torch.cuda.device_count() if torch.cuda.is_available() else 0,\n"
        "        'names': [torch.cuda.get_device_name(i)\n"
        "                  for i in range(torch.cuda.device_count())]\n"
        "                 if torch.cuda.is_available() else []}\n"
        "print('PROBE' + json.dumps(info))\n"
    )
    try:
        out = subprocess.run([sys.executable, "-c", probe], cwd=HERE,
                             capture_output=True, text=True, timeout=300)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"  Device: cannot probe CUDA ({exc})")
        return False

    line = next((l for l in out.stdout.splitlines() if l.startswith("PROBE")), None)
    if line is None:
        print("  Device: CUDA probe failed - could not import torch.")
        detail = (out.stderr or out.stdout).strip().splitlines()
        for l in detail[-5:]:
            print(f"    {l}")
        return False

    import json
    info = json.loads(line[len("PROBE"):])
    if not info["available"]:
        print(f"  Device: config asks for '{device}' but torch.cuda.is_available() is False.")
        return False

    index = 0
    if ":" in device:
        try:
            index = int(device.split(":", 1)[1])
        except ValueError:
            index = 0
    if index >= info["count"]:
        print(f"  Device: config asks for '{device}' but only {info['count']} GPU(s) present; "
              f"SOTAttack would silently fall back to cuda:0.")
        return False

    print(f"  Device: {device} -> {info['names'][index]} ({info['count']} GPU(s) visible)")
    return True


def run_sot(config_name, attack, seed, out_dir, overrides, log_path, dry_run=False):
    """Run SOTAttack once as a subprocess. Returns (returncode, elapsed_seconds)."""
    cmd = [
        sys.executable, SOT_SCRIPT,
        f"--config-name={config_name}",
        f"attack={attack}",
        f"seed={'null' if seed is None else seed}",
        f"data.output={out_dir}",
        f"hydra.run.dir={os.path.join(out_dir, 'hydra')}",
    ] + list(overrides)

    print(f"  $ {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0, 0.0

    os.makedirs(out_dir, exist_ok=True)
    start = time.time()

    with open(log_path, "ab") as log:
        header = (
            f"\n{'=' * 70}\n"
            f"{datetime.now():%Y-%m-%d %H:%M:%S} | attack={attack} seed={seed_label(seed)}\n"
            f"{' '.join(cmd)}\n"
            f"{'=' * 70}\n"
        )
        log.write(header.encode())
        log.flush()

        proc = subprocess.Popen(
            cmd, cwd=HERE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        # Read raw chunks rather than lines: tqdm redraws with '\r' and would
        # otherwise appear frozen until a run finished.
        try:
            while True:
                chunk = os.read(proc.stdout.fileno(), 8192)
                if not chunk:
                    break
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
                log.write(chunk)
                log.flush()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise
        finally:
            proc.stdout.close()
        returncode = proc.wait()

    return returncode, time.time() - start


def main():
    parser = argparse.ArgumentParser(
        description="Smoke-test then sequentially generate SOTAttack variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-name", default="ensemble_3models",
                        help="Hydra config in config/ to base every run on")
    parser.add_argument("--attacks", default="mifgsm",
                        help="Comma-separated attack methods [fgsm, mifgsm, pgd]")
    parser.add_argument("--seeds", default="default,2026",
                        help="Comma-separated seeds; 'default' means unseeded")
    parser.add_argument("--root", default="remain",
                        help="Root folder holding one subfolder per variant")
    parser.add_argument("--smoke-steps", type=int, default=3,
                        help="Attack steps during the smoke test (keep it small; it only "
                             "has to prove the code path runs end to end)")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Go straight to the full runs")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Run the smoke tests and stop")
    parser.add_argument("--keep-smoke", action="store_true",
                        help="Keep the smoke-test output instead of deleting it")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Abort the queue if a full run fails, instead of "
                             "carrying on with the remaining variants")
    parser.add_argument("--cooldown", type=int, default=10,
                        help="Seconds to wait between runs so the GPU is fully released")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Continue even when the configured CUDA device is unusable "
                             "(the runs would fall back to CPU and take days)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands without running anything")
    parser.add_argument("overrides", nargs="*",
                        help="Extra Hydra overrides passed to every run, "
                             "e.g. data.batch_size=5 model.device=cuda:1")
    args = parser.parse_args()

    # Catch a malformed override here rather than letting Hydra reject it mid-queue.
    bad = [o for o in args.overrides if "=" not in o]
    if bad:
        parser.error(
            f"Not a Hydra override: {', '.join(bad)}. "
            f"Extra overrides are passed as bare key=value arguments, "
            f"e.g. python run_variants.py data.batch_size=5"
        )

    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    seeds = [parse_seed(s) for s in args.seeds.split(",") if s.strip()]
    variants = [(a, s) for a in attacks for s in seeds]
    if not variants:
        parser.error("No variants to run: check --attacks and --seeds.")

    root = os.path.abspath(os.path.join(HERE, args.root)) \
        if not os.path.isabs(args.root) else args.root

    # An explicit data.batch_size override wins over the value in the config file.
    batch_size = read_config_value(args.config_name, "data", "batch_size", default=1)
    num_samples = read_config_value(args.config_name, "data", "num_samples", default="?")
    for ov in args.overrides:
        if ov.startswith("data.batch_size="):
            batch_size = int(ov.split("=", 1)[1])
        if ov.startswith("data.num_samples="):
            num_samples = int(ov.split("=", 1)[1])

    print("=" * 70)
    print("  SOTAttack variant runner")
    print("=" * 70)
    print(f"  config      : {args.config_name}")
    print(f"  variants    : {len(variants)}")
    for attack, seed in variants:
        print(f"                - {attack}, seed={seed_label(seed)} "
              f"-> {os.path.join(args.root, attack, 'seed' + seed_label(seed))}")
    print(f"  batch size  : {batch_size}")
    print(f"  num samples : {num_samples} per variant")
    if args.overrides:
        print(f"  overrides   : {' '.join(args.overrides)}")
    print(f"  smoke test  : {'skipped' if args.skip_smoke else f'1 batch ({batch_size} images), {args.smoke_steps} steps'}")

    # ------------------------------------------------------------ device check
    device = resolve_configured_device(args.config_name, args.overrides)
    if args.dry_run:
        print(f"  device      : {device} (not probed in a dry run)")
    elif not preflight_device(device):
        if args.allow_cpu:
            print("  --allow-cpu given: continuing on CPU anyway.")
        else:
            print("=" * 70)
            print("\n  Aborted: the GPU this run is configured for is not usable, and every")
            print("  variant would silently fall back to CPU. Fix the GPU/driver, point at")
            print("  another device with model.device=cuda:N, or pass --allow-cpu to")
            print("  proceed on CPU on purpose.")
            return 1
    print("=" * 70)

    # ---------------------------------------------------------------- smoke test
    if not args.skip_smoke:
        print("\n\n" + "#" * 70)
        print("#  PHASE 1/2 - SMOKE TEST (one batch per variant)")
        print("#" * 70)

        smoke_root = os.path.join(root, "_smoke")
        failures = []

        for position, (attack, seed) in enumerate(variants, start=1):
            name = f"{attack}, seed={seed_label(seed)}"
            print(f"\n--- smoke {position}/{len(variants)}: {name} ---")

            out_dir = os.path.join(smoke_root, f"{attack}_seed{seed_label(seed)}")
            log_path = os.path.join(smoke_root, "smoke.log")
            if not args.dry_run:
                os.makedirs(smoke_root, exist_ok=True)

            returncode, elapsed = run_sot(
                args.config_name, attack, seed, out_dir,
                list(args.overrides) + [
                    f"data.num_samples={batch_size}",
                    f"optim.steps={args.smoke_steps}",
                ],
                log_path, dry_run=args.dry_run,
            )

            produced = 0 if args.dry_run else count_pngs(os.path.join(out_dir, "img"))
            if args.dry_run:
                print(f"  (dry run, nothing executed)")
            elif returncode != 0:
                failures.append((name, f"exit code {returncode}"))
                print(f"  FAILED: exit code {returncode} after {fmt_duration(elapsed)}")
            elif produced == 0:
                failures.append((name, "ran but produced no images"))
                print(f"  FAILED: no images written after {fmt_duration(elapsed)}")
            else:
                print(f"  OK: {produced} image(s) in {fmt_duration(elapsed)}")

            print_gpu_memory()

        if failures:
            print("\n" + "!" * 70)
            print("!  SMOKE TEST FAILED - no full run was started.")
            for name, why in failures:
                print(f"!    {name}: {why}")
            print(f"!  Check the log: {os.path.join(smoke_root, 'smoke.log')}")
            print("!" * 70)
            return 1

        if not args.dry_run:
            print("\n  All smoke tests passed.")
            if args.keep_smoke:
                print(f"  Smoke output kept in {smoke_root}")
            else:
                shutil.rmtree(smoke_root, ignore_errors=True)
                print("  Smoke output removed.")

        if args.smoke_only:
            print("\n  --smoke-only given: stopping before the full runs.")
            return 0

    # ---------------------------------------------------------------- full runs
    print("\n\n" + "#" * 70)
    print("#  PHASE 2/2 - FULL GENERATION (sequential)")
    print("#" * 70)

    results = []
    queue_start = time.time()

    for position, (attack, seed) in enumerate(variants, start=1):
        name = f"{attack}, seed={seed_label(seed)}"
        out_dir = os.path.join(root, attack, f"seed{seed_label(seed)}")
        log_path = os.path.join(out_dir, "run.log")
        if not args.dry_run:
            os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"  RUN {position}/{len(variants)}: {name}")
        print(f"  output: {out_dir}")
        print(f"  log   : {log_path}")
        print(f"{'=' * 70}")

        returncode, elapsed = run_sot(
            args.config_name, attack, seed, out_dir,
            list(args.overrides), log_path, dry_run=args.dry_run,
        )

        produced = 0 if args.dry_run else count_pngs(os.path.join(out_dir, "img"))
        status = "ok" if returncode == 0 else f"FAILED (exit {returncode})"
        results.append((name, status, produced, elapsed, out_dir))

        print(f"\n  {name}: {status}, {produced} image(s), {fmt_duration(elapsed)}")

        if returncode != 0 and args.stop_on_error:
            print("  --stop-on-error given: abandoning the remaining variants.")
            break

        # Give the driver a moment to reclaim the memory before the next variant.
        if position < len(variants) and not args.dry_run and args.cooldown > 0:
            print_gpu_memory()
            print(f"  Cooling down {args.cooldown}s before the next variant...")
            time.sleep(args.cooldown)

    # ---------------------------------------------------------------- summary
    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, status, produced, elapsed, out_dir in results:
        print(f"  {name:<28s} {status:<20s} {produced:>5d} img  {fmt_duration(elapsed)}")
        print(f"    {out_dir}")
    print(f"\n  Total wall clock: {fmt_duration(time.time() - queue_start)}")
    print("=" * 70)

    failed = [r for r in results if r[1] != "ok"]
    if failed:
        print(f"\n  {len(failed)} variant(s) failed. Re-run the same command to resume "
              f"them; finished images are skipped automatically.")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted. Re-run the same command to resume where it stopped.")
        sys.exit(130)
