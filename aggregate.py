#!/usr/bin/env python3
"""Tables of the ablation study from the judged runs (after evaluate.py).

    python aggregate.py [--out ablation_out] [--num 100]

Per run: Avg KMR, KMR_a/b/c (share of images with >= 25/50/100 % keywords matched),
ASR@0.3/0.5/0.7 and AvgSim, all from the DeepSeek judge. Per variant: mean +/- std over seeds.
Images whose description Gemini blocked are left out of the metrics and counted in the
"Blocked" column (blocked.csv lists them); a blocked target drops that pair from every run.
Delta vs the reference: difference of per-image means over seeds, 95 % paired bootstrap CI over
images. Writes summary.md, tables.tex, per_run.csv and lambda_curve.png to <out>/eval/.
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np

from ablation_runs import ALL_VARIANTS

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = [("avg_kmr", "Avg KMR"), ("kmr_a", "KMR_a"), ("kmr_b", "KMR_b"), ("kmr_c", "KMR_c"),
           ("asr03", "ASR@0.3"), ("asr05", "ASR@0.5"), ("asr07", "ASR@0.7"), ("avgsim", "AvgSim")]
N_ENC = 3


def passes_per_step(variant):
    """Encoder-view passes per step as counted in the thesis: N(M+2) vs 2N."""
    script, overrides, _ = ALL_VARIANTS[variant]
    if script != "SOTAttack.py":
        return 2 * N_ENC
    m = 5
    for o in overrides:
        if o.startswith("optim.num_crops="):
            m = int(o.split("=")[1])
    return N_ENC * (m + 2)


def steps_of(variant):
    for o in ALL_VARIANTS[variant][1]:
        if o.startswith("optim.steps="):
            return int(o.split("=")[1])
    return 300


def load_run(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[r["img"]] = r
    return {i: r for i, r in rows.items() if "error" not in r}


def run_metrics(rows):
    rows = {i: r for i, r in rows.items() if not r.get("blocked")}
    kmr = np.array([r["kmr"] for r in rows.values()])
    score = np.array([r["score"] for r in rows.values()])
    return {
        "avg_kmr": 100 * kmr.mean(), "kmr_a": 100 * (kmr >= 0.25).mean(), "kmr_b": 100 * (kmr >= 0.5).mean(),
        "kmr_c": 100 * (kmr >= 1.0).mean(), "asr03": 100 * (score >= 0.3).mean(),
        "asr05": 100 * (score >= 0.5).mean(), "asr07": 100 * (score >= 0.7).mean(), "avgsim": score.mean(),
    }


def mean_std(values, digits):
    v = np.array(values, dtype=float)
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


def paired_delta(per_image_a, per_image_b, scale, rng, n_boot=10000):
    """mean(a - b) over common images with a 95 % bootstrap CI."""
    common = sorted(set(per_image_a) & set(per_image_b))
    d = np.array([per_image_a[i] - per_image_b[i] for i in common]) * scale
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d.mean(), lo, hi


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def tex_table(caption, headers, rows):
    esc = lambda s: str(s).replace("±", r"$\pm$").replace("_", r"\_").replace("%", r"\%").replace("λ", r"$\lambda$").replace("Δ", r"$\Delta$")
    lines = [r"\begin{table}[t]\centering\small", rf"\caption{{{esc(caption)}}}",
             r"\begin{tabular}{l" + "c" * (len(headers) - 1) + "}", r"\toprule",
             " & ".join(esc(h) for h in headers) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="ablation_out")
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--ref", default="goc", help="reference variant for the deltas")
    args = ap.parse_args()
    out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    eval_root = os.path.join(out, "eval")
    judge_dir = os.path.join(eval_root, "judge")

    runs = defaultdict(dict)  # variant -> seed -> rows
    incomplete = []
    for name in sorted(os.listdir(judge_dir)) if os.path.isdir(judge_dir) else []:
        m = re.fullmatch(r"(.+)_s(\d+)\.jsonl", name)
        if not m or m.group(1) not in ALL_VARIANTS:
            continue
        rows = {i: r for i, r in load_run(os.path.join(judge_dir, name)).items() if i < args.num}
        if len(rows) < args.num:
            incomplete.append(f"{name[:-6]} ({len(rows)}/{args.num})")
            continue  # a partial run would bias the means
        runs[m.group(1)][int(m.group(2))] = rows
    if not runs:
        raise SystemExit(f"no fully judged run in {judge_dir}")

    per_run = {v: {s: run_metrics(rows) for s, rows in seeds.items()} for v, seeds in runs.items()}
    # Per-image mean over seeds, for the paired comparison.
    # (images blocked in any seed of a variant are left out of that variant's paired comparison)
    per_image = {v: {key: {i: float(np.mean([rows[i][key] for rows in seeds.values()])) for i in range(args.num)
                           if not any(rows[i].get("blocked") for rows in seeds.values())}
                     for key in ("kmr", "score")} for v, seeds in runs.items()}
    n_blocked = {v: sum(1 for rows in seeds.values() for r in rows.values() if r.get("blocked") == "adversarial")
                 for v, seeds in runs.items()}
    blocked_targets = sorted({i for seeds in runs.values() for rows in seeds.values()
                              for i, r in rows.items() if r.get("blocked") == "target"})
    rng = np.random.default_rng(0)

    def row(variant, label, extra=(), ref=None):
        ref = ref or args.ref
        if variant not in per_run:
            return None
        seeds = per_run[variant]
        cells = [label, len(seeds), *extra, n_blocked[variant]]
        cells += [mean_std([m[k] for m in seeds.values()], 3 if k == "avgsim" else 1) for k, _ in METRICS]
        if ref in per_image and variant != ref:
            d, lo, hi = paired_delta(per_image[variant]["kmr"], per_image[ref]["kmr"], 100, rng)
            cells.append(f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]")
            d, lo, hi = paired_delta(per_image[variant]["score"], per_image[ref]["score"], 1, rng)
            cells.append(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        else:
            cells += ["–", "–"]
        return cells

    def metric_heads_for(ref):
        return ["Blocked"] + [h for _, h in METRICS] + [f"ΔKMR vs {ref} [95% CI]", "ΔAvgSim [95% CI]"]

    metric_heads = metric_heads_for(args.ref)
    cost = lambda v: (passes_per_step(v), steps_of(v), passes_per_step(v) * steps_of(v))
    tables = [
        ("Fixed vs adaptive λ (adaptive encoder weights, M = 5)", ["Variant", "Seeds"] + metric_heads, [
            row("foa_T300", "λ = 1 (FOA-Attack)"), row("baseonly", "λ = 1 (in SOTAttack)"),
            row("fixlam05", "λ = 0.5"), row("fixlam03", "λ = 0.3"), row("multionly", "λ = 0"),
            row("goc", "adaptive λ (SOTAttack)")]),
        ("Adaptive λ × encoder re-weighting", ["Variant", "Seeds"] + metric_heads, [
            row("goc", "adaptive λ, adaptive w"), row("uniw", "adaptive λ, uniform w"),
            row("fixlam05", "λ = 0.5, adaptive w"), row("fixlam05_uniw", "λ = 0.5, uniform w")]),
        ("Compute-matched FOA-Attack", ["Method", "Seeds", "Passes/step", "Steps", "Total passes"] + metric_heads, [
            row(v, label, cost(v)) for v, label in [
                ("foa_T300", "FOA-Attack"), ("foa_T1050", "FOA-Attack"), ("M1", "SOTAttack M = 1"), ("goc", "SOTAttack M = 5")]]),
        ("Number of extra views M (adaptive λ)", ["Variant", "Seeds", "Passes/step"] + metric_heads, [
            row(v, label, (passes_per_step(v),)) for v, label in [
                ("M1", "M = 1"), ("M3", "M = 3"), ("goc", "M = 5"), ("M7", "M = 7")]]),
        ("Fixed λ = 0.5 vs FOA-Attack at equal compute (reference: FOA-Attack T = 1050)",
         ["Method", "Seeds", "Total passes"] + metric_heads_for("foa_T1050"), [
            row(v, label, (passes_per_step(v) * steps_of(v),), ref="foa_T1050") for v, label in [
                ("foa_T1050", "FOA-Attack T = 1050"), ("fixlam05", "λ = 0.5, M = 5"),
                ("fix05_M1", "λ = 0.5, M = 1"), ("goc", "adaptive λ, M = 5")]]),
        ("Number of extra views M (λ = 0.5, reference: M = 5)", ["Variant", "Seeds", "Passes/step"] + metric_heads_for("fixlam05"), [
            row(v, label, (passes_per_step(v),), ref="fixlam05") for v, label in [
                ("fix05_M1", "M = 1"), ("fix05_M3", "M = 3"), ("fixlam05", "M = 5"), ("fix05_M7", "M = 7")]]),
    ]

    md = [f"# Ablation summary ({args.num} pairs, judge: DeepSeek)", ""]
    tex = []
    for title, headers, rows in tables:
        rows = [r for r in rows if r is not None]
        if not rows:
            continue
        md += [f"## {title}", "", md_table(headers, rows), ""]
        tex.append(tex_table(title, headers, rows))
    md.append("Passes/step counts encoder-view evaluations as in the thesis (N(M+2) vs 2N, N = 3); "
              "the FOA-Attack script also runs one logging-only pass on the full image.")
    md += ["", "Blocked = adversarial images (summed over seeds) whose description Gemini refused; "
           "they are left out of the metrics, see blocked.csv."]
    if blocked_targets:
        md += ["", f"**Target images blocked by Gemini (pair dropped from every run):** {blocked_targets}"]
    if incomplete:
        md += ["", "**Excluded (not fully judged):** " + ", ".join(incomplete)]
    text = "\n".join(md) + "\n"
    print(text)
    with open(os.path.join(eval_root, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(eval_root, "tables.tex"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(tex) + "\n")
    with open(os.path.join(eval_root, "per_run.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed"] + [k for k, _ in METRICS])
        for v in per_run:
            for s in sorted(per_run[v]):
                w.writerow([v, s] + [f"{per_run[v][s][k]:.4f}" for k, _ in METRICS])

    with open(os.path.join(eval_root, "blocked.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed", "img", "side", "reason"])
        for v in runs:
            for s in sorted(runs[v]):
                for i, r in sorted(runs[v][s].items()):
                    if r.get("blocked"):
                        w.writerow([v, s, i, r["blocked"], r.get("reason", "")])

    curve = [(1.0, "foa_T300"), (0.5, "fixlam05"), (0.3, "fixlam03"), (0.0, "multionly")]
    pts = [(lam, [m["avg_kmr"] for m in per_run[v].values()]) for lam, v in curve if v in per_run]
    if len(pts) >= 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3.4))
        xs = [p[0] for p in pts]
        ys = [np.mean(p[1]) for p in pts]
        es = [np.std(p[1], ddof=1) if len(p[1]) > 1 else 0.0 for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label="fixed λ")
        if args.ref in per_run:
            ref = [m["avg_kmr"] for m in per_run[args.ref].values()]
            ax.axhline(np.mean(ref), color="C1", linestyle="--", label="adaptive λ")
            if len(ref) > 1:
                ax.axhspan(np.mean(ref) - np.std(ref, ddof=1), np.mean(ref) + np.std(ref, ddof=1), color="C1", alpha=0.15)
        ax.set_xlabel("λ (1 = base view only, FOA-Attack)")
        ax.set_ylabel("Avg KMR (%)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(eval_root, "lambda_curve.png"), dpi=200)
    print(f"Wrote summary.md, tables.tex, per_run.csv, blocked.csv" + (", lambda_curve.png" if len(pts) >= 2 else "") + f" to {eval_root}")


if __name__ == "__main__":
    main()
