"""
analyze_with_official_lowkey.py  --  run the existing analysis
(analyze_results.py) with the OFFICIAL LowKey baseline included.

Why this file exists
--------------------
analyze_results.py expects ONE .jsonl in which every record already holds
every method, and its METHODS list is fixed. The official LowKey results
live in their own file (one method per record). This script:

  1. merges the two files into a NEW .jsonl, matching records by 'index'
     (and checking 'img_path' agrees), adding 'official_lowkey' as one
     more method alongside ensemble / single / lowkey / fawkes / noise;
  2. registers that method in analyze_results' METHODS and LABEL;
  3. calls analyze_results.main() on the merged file.

So every statistic -- Wilson CIs, identity-level cluster bootstrap, exact
McNemar with Holm correction, threshold sweep, quality, ArcFace no-face
check -- is computed by YOUR existing, already-committed code. Nothing is
reimplemented here, so the two analyses cannot drift apart.

Neither input file is modified. Output goes to a NEW folder:
  experiments/eval_full/analysis_results_merged_<n>images_<tag>/

Caveat to carry into the thesis: the 'time' column for official_lowkey was
measured on a Colab Tesla T4, the other methods on the local GTX 1650.
Protection rates and SSIM/LPIPS are hardware-independent and fully
comparable; the time row is NOT. The script prints this reminder.

Run from the project root:
  python src\\analyze_with_official_lowkey.py
"""
import argparse
import json
import os
import sys

import analyze_results

NEW_METHOD = 'official_lowkey'
NEW_LABEL = 'LowKey (official code)'


def read_jsonl(path):
    recs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs[r['index']] = r
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(
        'experiments', 'eval_full', 'results_full_500images_vggreal_rgb01.jsonl'))
    ap.add_argument('--lowkey_results', default=os.path.join(
        'experiments', 'eval_full',
        'results_official_lowkey_500images_vggreal_rgb01.jsonl'))
    ap.add_argument('--pilot_n', type=int, default=20)
    args = ap.parse_args()

    for p in (args.results, args.lowkey_results):
        if not os.path.exists(p):
            raise SystemExit(f"ERROR: file not found: {p}")

    base = read_jsonl(args.results)
    lk = read_jsonl(args.lowkey_results)
    print(f"Main results   : {args.results}  ({len(base)} images)")
    print(f"Official LowKey: {args.lowkey_results}  ({len(lk)} images)")

    shared = sorted(set(base) & set(lk))
    if not shared:
        raise SystemExit("ERROR: the two files share no image indices.")
    only_base, only_lk = len(base) - len(shared), len(lk) - len(shared)
    if only_base or only_lk:
        print(f"NOTE: using the {len(shared)} images present in both "
              f"({only_base} only in main, {only_lk} only in LowKey).")

    # Merge, verifying each pair really is the same image.
    merged = []
    for i in shared:
        b, l = base[i], lk[i]
        if b['img_path'] != l['img_path']:
            raise SystemExit(
                f"ERROR: index {i} refers to different images:\n"
                f"  {b['img_path']}\n  {l['img_path']}")
        if NEW_METHOD in b:
            raise SystemExit(f"ERROR: '{NEW_METHOD}' already present in the "
                             f"main results file at index {i}.")
        rec = dict(b)
        rec[NEW_METHOD] = l[NEW_METHOD]
        merged.append(rec)

    stem = os.path.splitext(os.path.basename(args.results))[0]
    tag = stem.replace('results_full_', '')
    out_path = os.path.join(os.path.dirname(args.results),
                            f'results_merged_{tag}.jsonl')
    with open(out_path, 'w') as f:
        for rec in merged:
            f.write(json.dumps(rec) + '\n')
    print(f"Merged file    : {out_path}  ({len(merged)} images)\n")

    # Make the existing analysis aware of the extra method. Order matters
    # only for display; 'ensemble' stays first so McNemar compares against it.
    if NEW_METHOD not in analyze_results.METHODS:
        analyze_results.METHODS.append(NEW_METHOD)
    analyze_results.LABEL[NEW_METHOD] = NEW_LABEL

    # Run YOUR analysis on the merged file.
    sys.argv = [sys.argv[0], '--results', out_path,
                '--pilot_n', str(args.pilot_n)]
    analyze_results.main()

    print("\n" + "!" * 96)
    print("TIMING CAVEAT: 'time' for LowKey (official code) was measured on a "
          "Google Colab Tesla T4;")
    print("the other methods were timed on the local GTX 1650 Max-Q. Do NOT "
          "compare that row directly.")
    print("Protection rates, SSIM and LPIPS are hardware-independent and are "
          "directly comparable.")
    print("!" * 96)


if __name__ == '__main__':
    main()
