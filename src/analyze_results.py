"""
Statistical analysis of the full evaluation (evaluate_full_v2.py output).

Reads results_full_<n>images_<tag>.jsonl and produces, for every
method x target model:

  1. Protection rate (PR) with Wilson 95% confidence interval
  2. Identity-level results: PR per identity -> mean +/- SD across identities,
     plus a cluster-bootstrap 95% CI (resamples IDENTITIES, because the
     5 images of one person are not independent)
  3. McNemar exact tests on the SAME images: ensemble vs single / lowkey /
     fawkes / noise, with Holm correction across all comparisons
  4. Threshold sensitivity: PR at cosine-distance thresholds 0.3-0.7
     (recomputed from the saved distances, no re-run needed)
  5. Image quality: SSIM / LPIPS mean +/- SD and % of images with SSIM >= 0.90
  6. ArcFace (buffalo_l) no-face check, and PR with those cases excluded
  7. Pilot-subset check (first 20 images)

Only numpy is required (Wilson, McNemar and bootstrap are implemented here).

Run from the project root:
  python src\\analyze_results.py
  python src\\analyze_results.py --results experiments\\eval_full\\results_full_500images_vggreal_rgb01.jsonl
Outputs go to experiments\\eval_full\\analysis_<results file name>\\
(CSV tables + report.md).
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

METHODS = ['ensemble', 'single', 'lowkey', 'fawkes', 'noise']
LABEL = {'ensemble': 'Ours (3-model ensemble)', 'single': 'Single (FaceNet)',
         'lowkey': 'LowKey-style', 'fawkes': 'Fawkes-style', 'noise': 'Random noise (control)'}
TARGETS = ['facenet', 'vggface', 'arcface', 'adaface', 'elasticface', 'sphereface']
ROLE = {'facenet': 'proxy (white-box)', 'vggface': 'proxy (white-box)',
        'arcface': 'eval-only (buffalo_l)', 'adaface': 'held-out',
        'elasticface': 'held-out', 'sphereface': 'held-out'}
THRESHOLD = 0.3
SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7]
N_BOOT = 2000
SEED = 42


# ── statistics helpers ─────────────────────────────────────────────────────
def wilson(k, n, z=1.959964):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value. b = A protected & B not; c = reverse."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def cluster_bootstrap(values_by_id, rng, n_boot=N_BOOT):
    """95% CI of the image-level mean, resampling identities with replacement."""
    groups = [np.asarray(v, float) for v in values_by_id.values()]
    k = len(groups)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, k)
        sample = np.concatenate([groups[j] for j in pick])
        stats[b] = sample.mean()
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ── io ─────────────────────────────────────────────────────────────────────
def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    recs.sort(key=lambda r: r['index'])
    return recs


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def pct(x):
    return f"{100 * x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(
        'experiments', 'eval_full', 'results_full_500images_vggreal_rgb01.jsonl'))
    ap.add_argument('--pilot_n', type=int, default=20)
    args = ap.parse_args()

    recs = load(args.results)
    methods = [m for m in METHODS if all(m in r for r in recs)]
    n = len(recs)
    ids = sorted({r['identity'] for r in recs})
    out_dir = os.path.join(os.path.dirname(args.results), 'analysis_' +
                           os.path.splitext(os.path.basename(args.results))[0])
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    md = []   # markdown report lines

    print(f"Loaded {n} images, {len(ids)} identities, methods: {methods}")
    md += [f"# Evaluation analysis", "",
           f"Source: `{args.results}`  ", f"Images: **{n}**, identities: **{len(ids)}**, "
           f"PR threshold: cosine distance > {THRESHOLD}", ""]

    # ── 1 + 2. PR with Wilson CI and identity-level stats ─────────────────
    rows, md_rows = [], []
    for t in TARGETS:
        for m in methods:
            x = np.array([r[m][f'pr_{t}'] for r in recs])
            k = int(x.sum())
            lo, hi = wilson(k, n)
            by_id = defaultdict(list)
            for r in recs:
                by_id[r['identity']].append(r[m][f'pr_{t}'])
            id_means = np.array([np.mean(v) for v in by_id.values()])
            blo, bhi = cluster_bootstrap(by_id, rng)
            d = np.array([r[m][f'dist_{t}'] for r in recs])
            rows.append([t, ROLE[t], m, n, k, round(k / n, 4), round(lo, 4), round(hi, 4),
                         round(id_means.mean(), 4), round(id_means.std(ddof=1), 4),
                         round(blo, 4), round(bhi, 4),
                         round(d.mean(), 4), round(d.std(ddof=1), 4)])
            md_rows.append((t, m, k / n, lo, hi, id_means.mean(), id_means.std(ddof=1), blo, bhi))
    write_csv(os.path.join(out_dir, 'protection_rates.csv'),
              ['target', 'role', 'method', 'n', 'protected', 'pr', 'wilson_lo', 'wilson_hi',
               'id_mean_pr', 'id_sd_pr', 'boot_lo', 'boot_hi', 'mean_dist', 'sd_dist'], rows)

    print("\n" + "=" * 96)
    print("1. PROTECTION RATE (%)  [Wilson 95% CI]   |   identity-level mean ± SD")
    print("=" * 96)
    md += ["## 1. Protection rate", "",
           "PR % [Wilson 95% CI]; second line: mean ± SD across identities "
           "[cluster-bootstrap 95% CI].", ""]
    md.append("| Target | " + " | ".join(LABEL[m] for m in methods) + " |")
    md.append("|---|" + "---|" * len(methods))
    for t in TARGETS:
        cells, line = [], f"{t:<12}"
        for m in methods:
            (_, _, p, lo, hi, im, isd, blo, bhi) = next(x for x in md_rows if x[0] == t and x[1] == m)
            line += f" {m[:8]:>8}: {pct(p):>5} [{pct(lo)}-{pct(hi)}]"
            cells.append(f"{pct(p)} [{pct(lo)}–{pct(hi)}]<br>{pct(im)} ± {pct(isd)} "
                         f"[{pct(blo)}–{pct(bhi)}]")
        print(line)
        md.append(f"| {t}<br>*{ROLE[t]}* | " + " | ".join(cells) + " |")
    md.append("")

    # ── 3. McNemar: ensemble vs each other method, per target ─────────────
    comps = [(t, other) for t in TARGETS for other in methods if other != 'ensemble'
             if 'ensemble' in methods]
    raw = []
    for t, other in comps:
        a = np.array([r['ensemble'][f'pr_{t}'] for r in recs])
        b_ = np.array([r[other][f'pr_{t}'] for r in recs])
        b = int(((a == 1) & (b_ == 0)).sum())   # ours protects, other doesn't
        c = int(((a == 0) & (b_ == 1)).sum())   # other protects, ours doesn't
        raw.append((t, other, b, c, mcnemar_exact(b, c), a.mean() - b_.mean()))
    adj = holm([x[4] for x in raw])
    rows = [[t, other, b, c, round(diff, 4), f"{p:.3g}", f"{pa:.3g}",
             'ours better' if (pa < 0.05 and b > c) else
             'other better' if (pa < 0.05 and c > b) else 'no significant difference']
            for (t, other, b, c, p, diff), pa in zip(raw, adj)]
    write_csv(os.path.join(out_dir, 'mcnemar_tests.csv'),
              ['target', 'compared_with', 'ours_only', 'other_only', 'pr_difference',
               'p_exact', 'p_holm', 'verdict'], rows)
    print("\n" + "=" * 96)
    print("3. McNEMAR (exact, Holm-corrected): Ours vs each method on the SAME images")
    print("=" * 96)
    print(f"{'target':<12}{'vs':<9}{'ours only':>10}{'other only':>11}{'ΔPR':>8}"
          f"{'p (Holm)':>11}   verdict")
    md += ["## 2. Paired comparison (exact McNemar, Holm-corrected)", "",
           "*ours only* = images protected by ours but not the other method; "
           "*other only* = the reverse.", "",
           "| Target | vs | ours only | other only | ΔPR (pts) | p (Holm) | Verdict |",
           "|---|---|---|---|---|---|---|"]
    for row in rows:
        t, other, b, c, diff, p, pa, verdict = row
        print(f"{t:<12}{other:<9}{b:>10}{c:>11}{100 * diff:>+8.1f}{pa:>11}   {verdict}")
        md.append(f"| {t} | {LABEL[other]} | {b} | {c} | {100 * diff:+.1f} | {pa} | {verdict} |")
    md.append("")

    # ── 4. Threshold sensitivity ──────────────────────────────────────────
    rows = []
    for t in TARGETS:
        for m in methods:
            d = np.array([r[m][f'dist_{t}'] for r in recs])
            rows.append([t, m] + [round(float((d > th).mean()), 4) for th in SWEEP])
    write_csv(os.path.join(out_dir, 'threshold_sensitivity.csv'),
              ['target', 'method'] + [f'pr_at_{th}' for th in SWEEP], rows)
    print("\n" + "=" * 96)
    print("4. THRESHOLD SENSITIVITY: PR (%) at different distance thresholds")
    print("=" * 96)
    print(f"{'target':<12}{'method':<10}" + "".join(f"{th:>8}" for th in SWEEP))
    md += ["## 3. Threshold sensitivity", "",
           "PR (%) when 'protected' requires cosine distance above each threshold.", "",
           "| Target | Method | " + " | ".join(str(th) for th in SWEEP) + " |",
           "|---|---|" + "---|" * len(SWEEP)]
    for row in rows:
        print(f"{row[0]:<12}{row[1]:<10}" + "".join(f"{100 * v:>8.1f}" for v in row[2:]))
        md.append(f"| {row[0]} | {LABEL[row[1]]} | " + " | ".join(pct(v) for v in row[2:]) + " |")
    md.append("")

    # ── 5. Quality ────────────────────────────────────────────────────────
    rows = []
    for m in methods:
        s = np.array([r[m]['ssim'] for r in recs])
        l = np.array([r[m]['lpips'] for r in recs])
        tm = np.array([r[m]['time'] for r in recs])
        rows.append([m, round(s.mean(), 4), round(s.std(ddof=1), 4),
                     round(float((s >= 0.90).mean()), 4),
                     round(l.mean(), 4), round(l.std(ddof=1), 4),
                     round(tm.mean(), 2), round(tm.std(ddof=1), 2)])
    write_csv(os.path.join(out_dir, 'quality.csv'),
              ['method', 'ssim_mean', 'ssim_sd', 'frac_ssim_ge_0.90', 'lpips_mean', 'lpips_sd',
               'time_mean_s', 'time_sd_s'], rows)
    print("\n" + "=" * 96)
    print("5. IMAGE QUALITY AND COST")
    print("=" * 96)
    print(f"{'method':<10}{'SSIM':>16}{'SSIM>=0.90':>12}{'LPIPS':>18}{'time (s)':>14}")
    md += ["## 4. Image quality and cost", "",
           "| Method | SSIM (mean ± SD) | % images SSIM ≥ 0.90 | LPIPS (mean ± SD) | Time s/img |",
           "|---|---|---|---|---|"]
    for m, sm, ssd, frac, lm, lsd, tm, tsd in rows:
        print(f"{m:<10}{sm:>9.4f} ± {ssd:.4f}{100 * frac:>11.1f}%{lm:>11.4f} ± {lsd:.4f}"
              f"{tm:>9.2f} ± {tsd:.2f}")
        md.append(f"| {LABEL[m]} | {sm:.4f} ± {ssd:.4f} | {pct(frac)} | {lm:.4f} ± {lsd:.4f} | "
                  f"{tm:.2f} ± {tsd:.2f} |")
    md.append("")

    # ── 6. ArcFace no-face check ──────────────────────────────────────────
    nf_orig = sum(1 for r in recs if not r.get('arcface_face_detected_original', True))
    md += ["## 5. ArcFace (buffalo_l) face-detection check", ""]
    print("\n" + "=" * 96)
    print("6. ArcFace (buffalo_l) NO-FACE CHECK")
    print("=" * 96)
    print(f"Original images with no face detected: {nf_orig}")
    md.append(f"Original images with no face detected: **{nf_orig}**  ")
    for m in methods:
        nf = [r for r in recs if not r[m].get('arcface_face_detected', True)
              or not r.get('arcface_face_detected_original', True)]
        keep = [r for r in recs if r not in nf]
        pr_all = np.mean([r[m]['pr_arcface'] for r in recs])
        pr_keep = np.mean([r[m]['pr_arcface'] for r in keep]) if keep else float('nan')
        print(f"  {m:<9} no-face cases: {len(nf):>3}   PR all: {100 * pr_all:5.1f}%   "
              f"PR excluding them: {100 * pr_keep:5.1f}%  (n={len(keep)})")
        md.append(f"{LABEL[m]}: {len(nf)} no-face cases; PR {pct(pr_all)}% → "
                  f"{pct(pr_keep)}% when excluded (n={len(keep)})  ")
    md.append("")

    # ── 7. Pilot subset ───────────────────────────────────────────────────
    k = min(args.pilot_n, n)
    print("\n" + "=" * 96)
    print(f"7. PILOT SUBSET CHECK (first {k} images)")
    print("=" * 96)
    md += [f"## 6. First {k} images (pilot subset)", "",
           "| Target | " + " | ".join(LABEL[m] for m in methods) + " |",
           "|---|" + "---|" * len(methods)]
    for t in TARGETS:
        vals = [np.mean([r[m][f'pr_{t}'] for r in recs[:k]]) for m in methods]
        print(f"{t:<12}" + "".join(f"{m[:8]:>10}: {100 * v:5.1f}%" for m, v in zip(methods, vals)))
        md.append(f"| {t} | " + " | ".join(pct(v) for v in vals) + " |")
    md.append("")

    with open(os.path.join(out_dir, 'report.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(md))
    print(f"\nSaved tables + report.md to: {out_dir}")


if __name__ == '__main__':
    main()
