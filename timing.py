#!/usr/bin/env python3
"""Seconds per image of every run, from the save times of its PNGs.

    python timing.py            # prints a table and writes <out>/eval/timing.csv

The time of one image is the gap between two consecutive saves; the median gap is reported, so pauses
(an interrupted run resumed later) do not count. Compare only runs that had a GPU to themselves.
"""
import argparse
import csv
import glob
import os
import statistics


def save_times(out, run):
    for folder in (os.path.join(out, "work", run, "img"), os.path.join(out, "images", run)):
        times = sorted(os.path.getmtime(f) for f in glob.glob(os.path.join(folder, "**", "*.png"), recursive=True))
        if len(times) > 1:
            return times
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ablation_out")
    args = ap.parse_args()
    runs = {os.path.basename(d) for sub in ("work", "images") for d in glob.glob(os.path.join(args.out, sub, "*"))
            if os.path.isdir(d) and not os.path.basename(d).startswith((".", "_"))}
    rows = []
    for run in sorted(runs):
        times = save_times(args.out, run)
        gaps = [b - a for a, b in zip(times, times[1:])]
        if gaps:
            rows.append((run, len(times), round(statistics.median(gaps), 1)))
    print(f"{'run':<22} {'images':>6} {'s/image (median)':>17}")
    for run, n, sec in rows:
        print(f"{run:<22} {n:>6} {sec:>17.1f}")
    os.makedirs(os.path.join(args.out, "eval"), exist_ok=True)
    with open(os.path.join(args.out, "eval", "timing.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("run", "images", "median_seconds_per_image"), *rows])


if __name__ == "__main__":
    main()
