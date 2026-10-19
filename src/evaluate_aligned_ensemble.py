"""
evaluate_aligned_ensemble.py  --  does FACE ALIGNMENT explain the gap
between our ensemble and official LowKey on ArcFace-family targets?

The question
------------
Official LowKey reaches 97.2% PR on ArcFace (buffalo_l) where our ensemble
reaches 29.6%, on the same 500 images. Two differences could explain it:
  (a) proxy composition -- LowKey's four proxies are ALL angular-margin
      networks (2 ArcFace-loss + 2 CosFace-loss); ours has one ArcFace
      proxy out of three;
  (b) alignment and resolution -- LowKey detects and aligns the face, then
      perturbs through that warp at 112x112; we perturb the whole
      unaligned 160x160 frame.

This script isolates (b). The proxy ensemble, PGD settings, budget, seeds
and scoring are IDENTICAL to evaluate_full_v2.py. ONLY the spatial
treatment changes. If ArcFace PR rises sharply, alignment is a large part
of the answer; if it barely moves, proxy composition is.

Three conditions, all on the same images with the same seed (42 + i):

  baseline_160   our method exactly as published: perturb the whole
                 160x160 frame. Recomputed here (not copied) so all three
                 numbers come from one run on one machine.

  aligned_crop   perturb the MTCNN-aligned 112x112 face crop, and score
                 the crop against the unperturbed crop. The cleanest test
                 of the hypothesis, but the output is a crop, not a
                 usable photograph.

  aligned_paste  perturb the same aligned crop, resize it back and paste
                 it into the 160x160 frame, then score the FULL frame
                 against the original frame. This is what a deployable
                 version would have to do, and it includes the resampling
                 loss that step causes.

Comparing aligned_crop with aligned_paste separates "does alignment help
the attack?" from "does the benefit survive being put back into a photo?"

Detector choice: MTCNN (facenet-pytorch), NOT buffalo_l. Using the
ArcFace target's own detector to prepare the attack would make the
experiment circular. MTCNN is also the detector LowKey uses.

Honest naming: aligned_paste uses a crop-and-paste of the detected box,
not LowKey's full affine warp with landmark-based similarity transform.
It is a close approximation, and the thesis should describe it as such.

Nothing existing is modified. New outputs only:
  results_aligned_<n>images_<tag>.jsonl
  summary_aligned_<n>images_<tag>.json

Resumable: re-run the same command to continue after a stop.

Run from the project root (about 20-25 min for 100 images on a GTX 1650):
  python src\\evaluate_aligned_ensemble.py --n_images 100
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from PIL import Image
from facenet_pytorch import MTCNN

# Same settings and helpers as the main evaluation -- imported, not copied.
from evaluate_full_v2 import (
    VGG_CONFIGS, THRESHOLD, TARGETS, EPS,
    cosine_distance, arcface_onnx_embed, seed_all, device,
)
from models import (
    load_facenet, load_arcface,
    load_arcface_pytorch, get_arcface_pytorch_embedding,
    load_adaface, get_adaface_embedding,
    load_elasticface, get_elasticface_embedding,
    load_sphereface, get_sphereface_embedding,
)
from poison import (pgd_ensemble_poison, load_image_tensor,
                    compute_ssim, compute_lpips)
from dataset import build_dataset
from vggface_real import load_vggface_real, get_vggface_real_embedding

CONDITIONS = ['baseline_160', 'aligned_crop', 'aligned_paste']
CROP = 112          # aligned crop size (what the FR models consume)
FRAME = 160         # the frame size our published method works at


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def read_jsonl(path):
    recs = {}
    if not os.path.exists(path):
        return recs
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
    ap.add_argument('--n_images', type=int, default=100)
    ap.add_argument('--output_dir', default=os.path.join('experiments', 'eval_full'))
    ap.add_argument('--vgg_config', choices=list(VGG_CONFIGS), default='rgb01')
    ap.add_argument('--margin', type=int, default=0,
                    help='MTCNN margin in pixels around the detected box')
    args = ap.parse_args()

    n = args.n_images
    tag = f'vggreal_{args.vgg_config}'
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'results_aligned_{n}images_{tag}.jsonl')
    summary_path = os.path.join(args.output_dir, f'summary_aligned_{n}images_{tag}.json')
    print(f"Output: {out_path}")

    # ── Models: same loading as evaluate_full_v2.py ───────────────────────
    print("\nLoading proxy models...")
    facenet = load_facenet()
    vggface = load_vggface_real()
    vgg_bgr, vgg_mode = VGG_CONFIGS[args.vgg_config]
    arcface_pt = load_arcface_pytorch()
    if arcface_pt is None:
        raise SystemExit("ERROR: facexlib ArcFace proxy is required.")

    print("\nLoading evaluation targets...")
    arcface_onnx = load_arcface()
    adaface = load_adaface()
    elasticface = load_elasticface()
    sphereface = load_sphereface()
    if adaface is None or elasticface is None or sphereface is None:
        raise SystemExit("ERROR: all three held-out models must load.")
    for m in (facenet, vggface, adaface, elasticface, sphereface):
        m.requires_grad_(False)

    # post_process=False -> MTCNN returns 0-255 float, so we can scale to
    # [0,1] ourselves and match load_image_tensor's convention exactly.
    mtcnn = MTCNN(image_size=CROP, margin=args.margin, post_process=False,
                  select_largest=True, device=device)

    def facenet_fn(x): return facenet(x * 2 - 1)
    def vggface_fn(x):
        return get_vggface_real_embedding(vggface, x, bgr=vgg_bgr,
                                          normalize_mode=vgg_mode)
    def arcface_fn(x): return get_arcface_pytorch_embedding(arcface_pt, x)

    proxies = [(facenet_fn, 1/3), (vggface_fn, 1/3), (arcface_fn, 1/3)]

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

    def score(clean, adv):
        eo, _ = embed_all(clean)
        ep, det = embed_all(adv)
        r = {'ssim': round(float(compute_ssim(clean, adv)), 4),
             'lpips': round(float(compute_lpips(clean, adv)), 4),
             'arcface_face_detected': det}
        for t in TARGETS:
            d = cosine_distance(eo[t], ep[t])
            r[f'pr_{t}'] = int(d > THRESHOLD)
            r[f'dist_{t}'] = round(float(d), 4)
        return r

    data = build_dataset()[:n]
    print(f"\n{len(data)} images, "
          f"{len({e['identity'] for e in data})} identities")

    done = read_jsonl(out_path)
    if done:
        print(f"Resuming: {len(done)} already done.")
    no_face = []

    with open(out_path, 'a') as out:
        for i, entry in enumerate(data):
            if i in done:
                continue
            t_start = time.time()
            pil = Image.open(entry['path']).convert('RGB')

            # The 160x160 frame our published method works at. Built by the
            # SAME function evaluate_full_v2.py uses, so the baseline
            # condition is identical to the published method (and the tensor
            # is contiguous, which facexlib's internal .view() requires).
            frame = load_image_tensor(entry['path'], size=(FRAME, FRAME))

            # MTCNN-aligned crop, taken from the ORIGINAL image (not the
            # downscaled frame) so alignment is not handicapped by our resize.
            with torch.no_grad():
                face = mtcnn(pil)
            if face is None:
                no_face.append(os.path.basename(entry['path']))
                print(f"[{i+1}/{n}] {entry['identity']:<24} NO FACE DETECTED, skipped")
                continue
            crop = (face.unsqueeze(0).to(device) / 255.0).clamp(0, 1).contiguous()

            rec = {'index': i, 'identity': entry['identity'],
                   'img_path': entry['path']}

            # 1. baseline: whole 160x160 frame (our published method)
            seed_all(42 + i)
            t0 = time.time()
            adv_frame = pgd_ensemble_poison(frame, proxies)
            t_base = time.time() - t0
            rec['baseline_160'] = {'time': round(t_base, 2), **score(frame, adv_frame)}

            # 2. aligned crop: perturb the 112x112 aligned face
            seed_all(42 + i)
            t0 = time.time()
            adv_crop = pgd_ensemble_poison(crop, proxies)
            t_crop = time.time() - t0
            rec['aligned_crop'] = {'time': round(t_crop, 2), **score(crop, adv_crop)}

            # 3. aligned paste: put the perturbed crop back into the frame.
            # The crop came from the full-size image, so we rescale it to the
            # frame and overwrite the whole frame -- the honest equivalent of
            # "protect the face region, keep the picture usable". The clean
            # reference is the SAME operation on the unperturbed crop, so the
            # only difference measured is the perturbation, not the resampling.
            clean_paste = F.interpolate(crop, size=(FRAME, FRAME),
                                        mode='bilinear', align_corners=False).clamp(0, 1)
            adv_paste = F.interpolate(adv_crop, size=(FRAME, FRAME),
                                      mode='bilinear', align_corners=False).clamp(0, 1)
            # Re-project into the budget w.r.t. the clean pasted reference:
            # interpolation can push a few pixels past eps.
            delta = torch.clamp(adv_paste - clean_paste, -EPS, EPS)
            adv_paste = torch.clamp(clean_paste + delta, 0, 1)
            rec['aligned_paste'] = {'time': round(t_crop, 2),
                                    **score(clean_paste, adv_paste)}

            out.write(json.dumps(rec) + '\n')
            out.flush()
            os.fsync(out.fileno())
            done[i] = rec

            def bits(c):
                p = rec[c]
                return "".join(str(p[f'pr_{t}']) for t in TARGETS)
            print(f"[{i+1}/{n}] {entry['identity']:<24} "
                  f"base {bits('baseline_160')} | crop {bits('aligned_crop')} | "
                  f"paste {bits('aligned_paste')} | {time.time()-t_start:.0f}s")

    recs = [done[k] for k in sorted(done)]
    if not recs:
        raise SystemExit("No images scored.")

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {'n_images': len(recs),
               'n_identities': len({r['identity'] for r in recs}),
               'n_no_face_skipped': len(no_face), 'no_face': no_face[:10],
               'threshold': THRESHOLD, 'eps': EPS, 'crop': CROP, 'frame': FRAME,
               'note': ('Identical proxies, PGD settings, budget and seeds in all '
                        'three conditions; only the spatial treatment differs. '
                        'aligned_paste is crop-and-paste, not LowKey\'s affine warp.')}
    for c in CONDITIONS:
        s = {}
        for k in ['ssim', 'lpips', 'time']:
            v = [r[c][k] for r in recs]
            s[k] = {'mean': float(np.mean(v)), 'std': float(np.std(v))}
        for t in TARGETS:
            k = sum(r[c][f'pr_{t}'] for r in recs)
            s[f'pr_{t}'] = round(100 * k / len(recs), 1)
            s[f'pr_{t}_ci95'] = wilson_ci(k, len(recs))
            s[f'mean_dist_{t}'] = round(float(np.mean([r[c][f'dist_{t}'] for r in recs])), 4)
        s['arcface_no_face'] = int(sum(1 for r in recs if not r[c]['arcface_face_detected']))
        summary[c] = s

    # Paired tests against the baseline condition.
    tests = []
    for c in ['aligned_crop', 'aligned_paste']:
        for t in TARGETS:
            a = np.array([r['baseline_160'][f'pr_{t}'] for r in recs])
            b_ = np.array([r[c][f'pr_{t}'] for r in recs])
            b = int(((b_ == 1) & (a == 0)).sum())   # aligned protects, baseline doesn't
            cc = int(((b_ == 0) & (a == 1)).sum())
            p = mcnemar_exact(b, cc)
            tests.append({'condition': c, 'target': t, 'aligned_only': b,
                          'baseline_only': cc,
                          'pr_change_pts': round(100 * (b_.mean() - a.mean()), 1),
                          'p_exact': float(f'{p:.3g}')})
    summary['mcnemar_vs_baseline'] = tests

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    w = 16
    print("\n" + "=" * (18 + w * len(CONDITIONS)))
    print(f"ALIGNMENT TEST -- {summary['n_images']} images, "
          f"{summary['n_identities']} identities, threshold {THRESHOLD}")
    if no_face:
        print(f"({len(no_face)} image(s) skipped: MTCNN found no face)")
    print("=" * (18 + w * len(CONDITIONS)))
    print(f"{'':<18}" + "".join(f"{c:>{w}}" for c in CONDITIONS))
    for t in TARGETS:
        print(f"{'PR ' + t:<18}" + "".join(
            f"{summary[c][f'pr_{t}']:>{w-1}.1f}%" for c in CONDITIONS))
    for k in ['ssim', 'lpips', 'time']:
        print(f"{k.upper():<18}" + "".join(
            f"{summary[c][k]['mean']:>{w}.4f}" for c in CONDITIONS))
    print(f"{'ArcFace no-face':<18}" + "".join(
        f"{summary[c]['arcface_no_face']:>{w}}" for c in CONDITIONS))

    print("\nPaired change vs baseline_160 (exact McNemar, uncorrected):")
    print(f"{'condition':<16}{'target':<14}{'aligned only':>13}{'base only':>11}"
          f"{'dPR':>8}{'p':>11}")
    for x in tests:
        print(f"{x['condition']:<16}{x['target']:<14}{x['aligned_only']:>13}"
              f"{x['baseline_only']:>11}{x['pr_change_pts']:>+8.1f}{x['p_exact']:>11}")

    print(f"\nSummary saved to {summary_path}")
    print(f"Per-image results: {out_path}")
    print("\nNOTE: aligned_crop scores a 112x112 crop against the same crop; "
          "its SSIM/LPIPS\nare therefore crop-level and not directly comparable "
          "with the 160x160 conditions.")


if __name__ == '__main__':
    main()
