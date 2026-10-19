"""
evaluate_official_lowkey.py  --  O5: score the OFFICIAL LowKey images with
the exact evaluation pipeline of evaluate_full_v2.py.

What it does
------------
The official LowKey code (Cherepanova et al., ICLR 2021, OpenReview
supplementary, run unchanged on a Colab Tesla T4) produced one
<name>_attacked.png per evaluation image. This script scores those images
exactly the way evaluate_full_v2.py scored every other method:

  - images loaded with the same load_image_tensor() (originals AND LowKey)
  - same six targets: facenet, vggface (real, same vgg_config), arcface
    (buffalo_l ONNX), adaface, elasticface, sphereface
  - same cosine distance, same THRESHOLD (0.3), same SSIM / LPIPS functions
  - same arcface_face_detected flag
  - images matched to the 500-image run by index and filename

Nothing is regenerated, and no existing file is modified. The original
results file is only READ. Output goes to NEW files:
  results_official_lowkey_<N>images_<tag>.jsonl   (one line per image)
  summary_official_lowkey_<N>images_<tag>.json    (table vs all methods)

Pipeline check (before scoring)
-------------------------------
To prove this script scores images identically to evaluate_full_v2.py,
it first re-scores the SAVED 'ensemble' PNGs of the first few images and
compares the distances with those stored in the results file. Small
differences (~0.01) are expected, because the stored values were computed
on the un-rounded tensor and the PNG is rounded to 8-bit. Large
differences mean something does not match -- stop and investigate.

Timing: the 'time' field is LowKey's own generation time from
lowkey_log.csv, measured on a Tesla T4 (Colab), NOT on the local GTX 1650.
Report it with that hardware label.

Resumable: run the same command again to continue after a stop.

Run from the project root:
  python src\\evaluate_official_lowkey.py
"""
import argparse
import csv
import json
import math
import os

import numpy as np
import torch

# Reuse the EXACT settings and helpers from the v2 evaluation (imported,
# not copied, so they cannot drift apart). Importing does not run its main().
from evaluate_full_v2 import (
    VGG_CONFIGS, THRESHOLD, TARGETS, METHODS,
    cosine_distance, arcface_onnx_embed, device,
)
from models import (
    load_facenet, load_arcface,
    load_vggface_pytorch, get_vggface_pytorch_embedding,
    load_adaface, get_adaface_embedding,
    load_elasticface, get_elasticface_embedding,
    load_sphereface, get_sphereface_embedding,
)
from poison import load_image_tensor, compute_ssim, compute_lpips
from vggface_real import load_vggface_real, get_vggface_real_embedding

NEW_METHOD = 'official_lowkey'


def read_jsonl(path):
    """Read a JSONL file WITHOUT modifying it. Skips any half-written line."""
    recs = {}
    if not os.path.exists(path):
        return recs
    with open(path, 'r') as f:
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


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a proportion, returned in percent."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(
        'experiments', 'eval_full', 'results_full_500images_vggreal_rgb01.jsonl'),
        help='results file of the main 500-image run (read only)')
    ap.add_argument('--lowkey_dir', default=os.path.join(
        'experiments', 'lowkey_official_500'),
        help='folder with <name>_attacked.png files and lowkey_log.csv')
    ap.add_argument('--vggface', choices=['real', 'imagenet'], default='real')
    ap.add_argument('--vgg_config', choices=list(VGG_CONFIGS), default='rgb01')
    ap.add_argument('--check', type=int, default=5,
                    help='images used for the pipeline check (0 = skip)')
    args = ap.parse_args()

    # ── Make sure the settings match the results file we compare against ──
    tag = (f'vggreal_{args.vgg_config}' if args.vggface == 'real'
           else 'vggimagenet')
    if tag not in os.path.basename(args.results):
        raise SystemExit(
            f"ERROR: settings give tag '{tag}', but the results file is\n"
            f"  {args.results}\nUse the --vggface / --vgg_config that "
            f"produced that file.")

    base = read_jsonl(args.results)
    if not base:
        raise SystemExit(f"ERROR: no records found in {args.results}")
    n = len(base)
    out_dir = os.path.dirname(args.results)
    out_path = os.path.join(out_dir, f'results_{NEW_METHOD}_{n}images_{tag}.jsonl')
    summary_path = os.path.join(out_dir, f'summary_{NEW_METHOD}_{n}images_{tag}.json')
    saved_img_dir = os.path.join(out_dir, f'images_{n}_{tag}')
    print(f"Base results : {args.results}  ({n} images)")
    print(f"LowKey images: {args.lowkey_dir}")
    print(f"Output       : {out_path}")

    # LowKey generation times (Colab T4)
    lk_time = {}
    log_path = os.path.join(args.lowkey_dir, 'lowkey_log.csv')
    if os.path.exists(log_path):
        with open(log_path, newline='') as f:
            for row in csv.DictReader(f):
                if row['status'] == 'ok':
                    lk_time[row['image']] = float(row['seconds'])
    else:
        print("WARNING: lowkey_log.csv not found -- 'time' will be null.")

    # ── Models: identical loading to evaluate_full_v2.py (targets only) ────
    print("\nLoading target models...")
    facenet = load_facenet()
    if args.vggface == 'real':
        vggface = load_vggface_real()
        vgg_bgr, vgg_mode = VGG_CONFIGS[args.vgg_config]
    else:
        vggface = load_vggface_pytorch()
    arcface_onnx = load_arcface()
    adaface = load_adaface()
    elasticface = load_elasticface()
    sphereface = load_sphereface()
    if adaface is None or elasticface is None or sphereface is None:
        raise SystemExit("ERROR: all three held-out models must load.")
    for m in (facenet, vggface, adaface, elasticface, sphereface):
        m.requires_grad_(False)

    def facenet_fn(x): return facenet(x * 2 - 1)
    if args.vggface == 'real':
        def vggface_fn(x):
            return get_vggface_real_embedding(vggface, x, bgr=vgg_bgr,
                                              normalize_mode=vgg_mode)
    else:
        def vggface_fn(x): return get_vggface_pytorch_embedding(vggface, x)

    def embed_all(t):
        with torch.no_grad():
            e = {
                'facenet':     facenet_fn(t).squeeze(0),
                'vggface':     vggface_fn(t).squeeze(0),
                'adaface':     get_adaface_embedding(adaface, t).squeeze(0),
                'elasticface': get_elasticface_embedding(elasticface, t).squeeze(0),
                'sphereface':  get_sphereface_embedding(sphereface, t).squeeze(0),
            }
        e['arcface'], detected = arcface_onnx_embed(arcface_onnx, t)
        return e, detected

    def score(img, x):
        emb_o, det_o = embed_all(img)
        emb_p, det_p = embed_all(x)
        r = {'ssim': round(float(compute_ssim(img, x)), 4),
             'lpips': round(float(compute_lpips(img, x)), 4),
             'arcface_face_detected': det_p}
        for t in TARGETS:
            d = cosine_distance(emb_o[t], emb_p[t])
            r[f'pr_{t}'] = int(d > THRESHOLD)
            r[f'dist_{t}'] = round(float(d), 4)
        return r, det_o

    # ── Pipeline check against stored ensemble results ─────────────────────
    if args.check > 0:
        print(f"\nPipeline check on {args.check} saved 'ensemble' images...")
        diffs, flips, checked = [], 0, 0
        for i in sorted(base)[:args.check]:
            p = os.path.join(saved_img_dir, f'{i:04d}_ensemble.png')
            if not os.path.exists(p):
                continue
            rec = base[i]
            r, _ = score(load_image_tensor(rec['img_path']), load_image_tensor(p))
            for t in TARGETS:
                diffs.append(abs(r[f'dist_{t}'] - rec['ensemble'][f'dist_{t}']))
                flips += int(r[f'pr_{t}'] != rec['ensemble'][f'pr_{t}'])
            checked += 1
        if checked == 0:
            print(f"  skipped: no saved images found in {saved_img_dir}")
        else:
            print(f"  {checked} images x {len(TARGETS)} targets: "
                  f"mean |diff| {np.mean(diffs):.4f}, max |diff| {np.max(diffs):.4f}, "
                  f"protected/not flips {flips}")
            if np.max(diffs) > 0.05:
                raise SystemExit(
                    "STOP: distances differ by more than 0.05 from the stored "
                    "results, so this pipeline does not match evaluate_full_v2.py. "
                    "Paste this output for investigation.")
            print("  OK: pipeline matches (small differences = 8-bit PNG rounding).")

    # ── Score every official LowKey image ──────────────────────────────────
    done = read_jsonl(out_path)
    if done:
        print(f"\nResuming: {len(done)} images already scored.")
    # If a previous run stopped mid-line, start on a fresh line.
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, 'rb') as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b'\n'
    else:
        needs_newline = False

    missing = []
    with open(out_path, 'a') as out:
        if needs_newline:
            out.write('\n')
        for i in sorted(base):
            if i in done:
                continue
            rec = base[i]
            name = os.path.basename(rec['img_path'])
            stem = os.path.splitext(name)[0]
            lk_path = os.path.join(args.lowkey_dir, f'{stem}_attacked.png')
            if not os.path.exists(lk_path):
                missing.append(name)
                continue

            r, det_o = score(load_image_tensor(rec['img_path']),
                             load_image_tensor(lk_path))
            r = {'time': lk_time.get(name), **r}
            if det_o != rec['arcface_face_detected_original']:
                print(f"  note: ArcFace detection on original {name} differs "
                      f"from stored run")
            record = {'index': i, 'identity': rec['identity'],
                      'img_path': rec['img_path'], 'lowkey_path': lk_path,
                      'arcface_face_detected_original': det_o,
                      NEW_METHOD: r}
            out.write(json.dumps(record) + '\n')
            out.flush()
            os.fsync(out.fileno())
            done[i] = record

            p = r
            print(f"[{len(done)}/{n}] {rec['identity']:<28} "
                  f"PR FN/VGG/Arc/Ada/Ela/Sph: "
                  f"{p['pr_facenet']}{p['pr_vggface']}{p['pr_arcface']}"
                  f"{p['pr_adaface']}{p['pr_elasticface']}{p['pr_sphereface']} "
                  f"| SSIM {p['ssim']:.3f} LPIPS {p['lpips']:.4f}")

    if missing:
        print(f"\nWARNING: {len(missing)} LowKey image(s) not found, e.g. {missing[:3]}")

    # ── Summary: official LowKey next to every method of the main run ──────
    idx = sorted(i for i in done if i in base)
    methods = [m for m in METHODS if all(m in base[i] for i in idx)]
    rows = {m: [base[i][m] for i in idx] for m in methods}
    rows[NEW_METHOD] = [done[i][NEW_METHOD] for i in idx]
    order = methods + [NEW_METHOD]

    summary = {'n_images': len(idx),
               'n_identities': len({base[i]['identity'] for i in idx}),
               'vggface': args.vggface, 'vgg_config': args.vgg_config,
               'threshold': THRESHOLD,
               'note': ("official_lowkey = authors' attack_dir_warp.py, unchanged "
                        "settings (eps 0.05, 50 iters, LPIPS, Gaussian smoothing); "
                        "time measured on Tesla T4 (Colab). Other methods' time "
                        "measured on the local GPU.")}
    for m in order:
        rs = rows[m]
        s = {}
        for k in ['ssim', 'lpips', 'time']:
            v = [r[k] for r in rs if r.get(k) is not None]
            s[k] = ({'mean': float(np.mean(v)), 'std': float(np.std(v))}
                    if v else None)
        for t in TARGETS:
            k = sum(r[f'pr_{t}'] for r in rs)
            s[f'pr_{t}'] = round(100 * k / len(rs), 1)
            s[f'pr_{t}_ci95'] = wilson_ci(k, len(rs))
        s['arcface_no_face'] = int(sum(1 for r in rs if not r['arcface_face_detected']))
        summary[m] = s

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    w = 17
    print("\n" + "=" * (16 + w * len(order)))
    print(f"O5 COMPARISON -- {summary['n_images']} images, "
          f"{summary['n_identities']} identities, threshold {THRESHOLD}")
    print("=" * (16 + w * len(order)))
    print(f"{'':<16}" + "".join(f"{m:>{w}}" for m in order))
    for t in TARGETS:
        print(f"{'PR ' + t:<16}" + "".join(
            f"{summary[m][f'pr_{t}']:>{w-1}.1f}%" for m in order))
    for k in ['ssim', 'lpips', 'time']:
        print(f"{k.upper():<16}" + "".join(
            (f"{summary[m][k]['mean']:>{w}.4f}" if summary[m][k] else f"{'n/a':>{w}}")
            for m in order))
    print(f"{'ArcFace no-face':<16}" + "".join(
        f"{summary[m]['arcface_no_face']:>{w}}" for m in order))
    print("\nNote: official_lowkey time is on a Tesla T4 (Colab); others on the local GPU.")
    print(f"Summary saved to {summary_path}")
    print(f"Per-image results: {out_path}")


if __name__ == '__main__':
    main()
