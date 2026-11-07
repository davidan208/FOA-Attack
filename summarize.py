#!/usr/bin/env python3
"""Summarise the lambda logs of SOTAttack.py (TASK_lambda_logging.md, sections 6 and 7).

    python summarize.py <output>/logs [--steps 300] [--diag-every 10] [--out <dir>]

Reads every lam_log_*.jsonl in the folder, prints the checks of section 6 and the tables of
section 7, and writes summary.txt, hist_one_minus_lambda.png and lambda_over_steps.png to --out
(default: the logs folder). Rows are deduplicated on (img, t, enc) / (img, t), keeping the last,
so a resumed run is counted once.
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

ENC_NAMES = {0: "B/16", 1: "B/32", 2: "g/14"}
ONE_TOL = 1e-6  # kappa = 1e-8 puts lambda a hair below 1 when r ~ 1
W_TOL = 1e-4
EARLY = 50


def load(log_dir):
    lam, grad = {}, {}
    files = sorted(glob.glob(os.path.join(log_dir, "lam_log_*.jsonl")))
    if not files:
        raise SystemExit(f"no lam_log_*.jsonl in {log_dir}")
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("type") == "grad":
                    grad[(row["img"], row["t"])] = row
                else:
                    lam[(row["img"], row["t"], row["enc"])] = row
    return list(lam.values()), list(grad.values()), len(files)


def pct(mask):
    return 100.0 * float(np.mean(mask)) if len(mask) else float("nan")


def f4(x):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def text_table(headers, rows):
    cells = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[j]) for r in cells) for j in range(len(headers))]
    line = lambda r: "  ".join(c.rjust(w) for c, w in zip(r, widths))
    return "\n".join([line(cells[0]), "  ".join("-" * w for w in widths)] + [line(r) for r in cells[1:]])


def lam_block(rows):
    lam = np.array([r["lam"] for r in rows])
    r = np.array([r["r"] for r in rows])
    one = 1.0 - lam
    q = np.percentile(r, [5, 25, 50, 75, 95]) if len(r) else [float("nan")] * 5
    return {
        "mean": float(np.mean(one)) if len(one) else float("nan"),
        "median": float(np.median(one)) if len(one) else float("nan"),
        "eq1": pct(lam >= 1.0 - ONE_TOL), "lt095": pct(lam < 0.95), "lt09": pct(lam < 0.9),
        "q": [float(x) for x in q],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--diag-every", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or args.log_dir
    os.makedirs(out_dir, exist_ok=True)

    rows, grad, n_files = load(args.log_dir)
    encs = sorted({r["enc"] for r in rows})
    imgs = sorted({r["img"] for r in rows})
    name = lambda e: ENC_NAMES.get(e, str(e))
    out = []

    # ---- Section 6 checks
    per_img = defaultdict(int)
    for r in rows:
        per_img[r["img"]] += 1
    grad_img = defaultdict(int)
    for g in grad:
        grad_img[g["img"]] += 1
    exp_rows = args.steps * len(encs)
    exp_grad = math.ceil(args.steps / args.diag_every) if grad else 0
    bad_rows = {i: c for i, c in per_img.items() if c != exp_rows}
    bad_grad = {i: grad_img[i] for i in imgs if grad and grad_img[i] != exp_grad}
    t0 = [r for r in rows if r["t"] == 0]
    t0_bad = sorted({(r["img"], r["enc"]) for r in t0 if abs(r["lam"] - 1.0) > ONE_TOL})
    out_of_range = [r for r in rows if r["lam"] < 0.0 or r["lam"] > 1.0]
    w_sum = defaultdict(float)
    for r in rows:
        w_sum[(r["img"], r["t"])] += r["w"]
    w_bad = [k for k, v in w_sum.items() if abs(v - len(encs)) > W_TOL]

    ok = lambda b: "OK " if b else "FAIL"
    out.append(f"Logs: {args.log_dir} ({n_files} files, images {imgs[0]}-{imgs[-1]})")
    out.append("")
    out.append("== Section 6 checks ==")
    out.append(f"[{ok(not bad_rows)}] {exp_rows} lambda rows per image"
               + (f"  -> off: {dict(sorted(bad_rows.items())[:10])}" if bad_rows else ""))
    if grad:
        out.append(f"[{ok(not bad_grad)}] {exp_grad} grad rows per image"
                   + (f"  -> off: {dict(sorted(bad_grad.items())[:10])}" if bad_grad else ""))
    else:
        out.append("[ -- ] no grad rows (diag_grad off)")
    out.append(f"[{ok(not t0_bad)}] lambda(t=0) = 1 for every image/enc (tol {ONE_TOL:g})"
               + (f"  -> {len(t0_bad)}/{len(t0)} not, e.g. (img, enc) {t0_bad[:6]}" if t0_bad else ""))
    out.append(f"[{ok(not out_of_range)}] 0 <= lambda <= 1"
               + (f"  -> {len(out_of_range)} rows outside, min lambda {min(r['lam'] for r in out_of_range):.4f}"
                  if out_of_range else ""))
    out.append(f"[{ok(not w_bad)}] sum_i w_i = {len(encs)} every step (tol {W_TOL:g})"
               + (f"  -> {len(w_bad)} steps off" if w_bad else ""))

    # ---- Section 7 tables
    groups = [(name(e), [r for r in rows if r["enc"] == e]) for e in encs] + [("pooled", rows)]
    out += ["", "== 1 - lambda and r (all steps) =="]
    body = []
    for g, rs in groups:
        b = lam_block(rs)
        body.append([g, len(rs), f4(b["mean"]), f4(b["median"]), f"{b['eq1']:.1f}", f"{b['lt095']:.1f}",
                     f"{b['lt09']:.1f}"] + [f4(x) for x in b["q"]])
    out.append(text_table(["enc", "rows", "mean 1-lam", "median 1-lam", "%lam=1", "%lam<0.95", "%lam<0.9",
                           "r p5", "r p25", "r p50", "r p75", "r p95"], body))

    out += ["", f"== t < {EARLY} vs t >= {EARLY} =="]
    body = []
    for g, rs in groups:
        a = lam_block([r for r in rs if r["t"] < EARLY])
        b = lam_block([r for r in rs if r["t"] >= EARLY])
        body.append([g, f4(a["mean"]), f4(b["mean"]), f4(a["median"]), f4(b["median"]),
                     f"{a['eq1']:.1f}", f"{b['eq1']:.1f}", f"{a['lt095']:.1f}", f"{b['lt095']:.1f}",
                     f4(a["q"][2]), f4(b["q"][2])])
    out.append(text_table(["enc", "mean 1-lam <50", ">=50", "median <50", ">=50", "%lam=1 <50", ">=50",
                           "%lam<0.95 <50", ">=50", "r p50 <50", ">=50"], body))

    out += ["", "== Encoder weights w =="]
    per_step = defaultdict(list)
    for r in rows:
        per_step[(r["img"], r["t"])].append(r["w"])
    spread = [max(v) - min(v) for v in per_step.values()]
    all_w = [r["w"] for r in rows]
    out.append(text_table(["steps (img x t)", "mean(max-min)", "median(max-min)", "global min w", "global max w"],
                          [[len(per_step), f4(float(np.mean(spread))), f4(float(np.median(spread))),
                            f4(min(all_w)), f4(max(all_w))]]))
    out.append("mean w per enc: " + ", ".join(
        f"{name(e)} {f4(float(np.mean([r['w'] for r in rows if r['enc'] == e])))}" for e in encs))

    if grad:
        out += ["", "== Gradient diagnostic (per image, every %d steps) ==" % args.diag_every,
                "flip = share of pixels whose update sign changes without the (1-lam)*m term; "
                "share = ||g_full - g_base|| / ||g_full||;",
                "zero_base = pixels with g_base exactly 0 (outside the base crop); flip_nz = flip on the other pixels."]
        body = []
        for label, gs in (("all", grad), (f"t < {EARLY}", [g for g in grad if g["t"] < EARLY]),
                          (f"t >= {EARLY}", [g for g in grad if g["t"] >= EARLY])):
            vals = {k: np.array([g[k] for g in gs if k in g]) for k in ("flip", "share", "zero_base", "flip_nz")}
            stat = lambda k, fn: f4(float(fn(vals[k]))) if len(vals[k]) else "nan"
            body.append([label, len(gs), stat("flip", np.mean), stat("flip", np.median), stat("share", np.mean),
                         stat("share", np.median), stat("zero_base", np.mean), stat("flip_nz", np.mean)])
        out.append(text_table(["steps", "rows", "mean flip", "median flip", "mean share", "median share",
                               "mean zero_base", "mean flip_nz"], body))

    text = "\n".join(out) + "\n"
    print(text)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = {e: 1.0 - np.array([r["lam"] for r in rows if r["enc"] == e]) for e in encs}
    hi = max(1e-3, max(float(np.max(v)) for v in vals.values()))
    bins = np.linspace(min(0.0, min(float(np.min(v)) for v in vals.values())), hi, 41)
    fig, axes = plt.subplots(1, len(encs), figsize=(4.2 * len(encs), 3.6), sharey=True)
    for ax, e in zip(np.atleast_1d(axes), encs):
        ax.hist(vals[e], bins=bins, color="C%d" % e)
        ax.set_yscale("log")
        ax.set_title(f"{name(e)}  (λ<1 on {100 * np.mean(vals[e] > ONE_TOL):.1f}% of steps)")
        ax.set_xlabel("1 − λ")
    np.atleast_1d(axes)[0].set_ylabel("image × step (log)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_one_minus_lambda.png"), dpi=200)
    plt.close(fig)

    def by_step(rs, key):
        acc = defaultdict(list)
        for r in rs:
            acc[r["t"]].append(r[key])
        ts = sorted(acc)
        return ts, [float(np.mean(acc[t])) for t in ts]

    fig, axes = plt.subplots(1, 3 if grad else 2, figsize=(15 if grad else 10, 3.6))
    for e in encs:
        t, v = by_step([dict(r, one=1.0 - r["lam"]) for r in rows if r["enc"] == e], "one")
        axes[0].plot(t, v, label=name(e))
        t, v = by_step([r for r in rows if r["enc"] == e], "w")
        axes[1].plot(t, v, label=name(e))
    axes[0].set(title="mean 1 − λ per step", xlabel="t")
    axes[1].set(title="mean w per step", xlabel="t")
    axes[1].axhline(1.0, color="gray", lw=0.8, ls="--")
    for ax in axes[:2]:
        ax.legend()
    if grad:
        for k in ("flip", "share", "flip_nz"):
            t, v = by_step([g for g in grad if k in g], k)
            axes[2].plot(t, v, marker="o", ms=2.5, label=k)
        axes[2].set(title="gradient diagnostic per step", xlabel="t")
        axes[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lambda_over_steps.png"), dpi=200)
    print(f"Wrote summary.txt, hist_one_minus_lambda.png, lambda_over_steps.png to {out_dir}")


if __name__ == "__main__":
    main()
