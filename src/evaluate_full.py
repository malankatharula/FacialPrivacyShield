"""
Full-scale evaluation -- O1/O2 + O3 + O5 in ONE pass (resumable).

Why this file exists
--------------------
evaluate.py, evaluate_transferability.py and evaluate_baselines.py each
regenerate the same perturbations from scratch (the ensemble image is
generated three times per image). For a 500-image run that would cost
~10-11 hours. This script generates each protected image ONCE and
evaluates it against ALL target models, so:
  - runtime is roughly halved (~5-6 h for 500 images, estimate)
  - O1/O2, O3 and O5 are guaranteed to use the IDENTICAL protected images

It does NOT modify any existing file. It reuses the same functions,
seeds (42 + i), PGD settings, thresholds and preprocessing as the pilot
scripts, so `--n_images 20` should reproduce the locked pilot results
(the 20 pilot images are the first 20 of the 500).

Methods generated per image (each seeded with 42 + i, as in the pilot):
  single   -- FaceNet only                         (evaluate.py)
  ensemble -- FaceNet + VGG-Face + ArcFace (Ours)  (evaluate.py / "ours")
  fawkes   -- Fawkes-style targeted collision      (evaluate_baselines.py)
  lowkey   -- LowKey-style FaceNet + ArcFace       (evaluate_baselines.py)

Targets evaluated for every method:
  facenet, vggface            (proxies -- white-box)
  arcface                     (buffalo_l ONNX -- eval only)
  adaface, elasticface, sphereface   (held-out -- never optimised against)

Extra field (new): arcface_face_detected. If insightface finds no face it
returns a zero embedding, which yields distance 1.0 and is counted as
"protected" without anything being measured. This flag lets the analysis
report and exclude those cases.

Resumable: one JSON line is written per finished image. If the run stops
(crash, restart, Ctrl+C), run the same command again and it skips images
already done.

Run from the project root:
  python src\\evaluate_full.py --n_images 10     # quick speed / smoke test
  python src\\evaluate_full.py --n_images 20     # reproduce pilot (check)
  python src\\evaluate_full.py --n_images 500    # main run
Optional: --save_images  (saves each protected image as PNG, ~4 per image)
"""
import argparse
import json
import os
import random
import tempfile
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Same determinism settings as evaluate_transferability.py / evaluate_baselines.py
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from models import (
    load_facenet, load_arcface,
    load_vggface_pytorch, get_vggface_pytorch_embedding,
    load_arcface_pytorch, get_arcface_pytorch_embedding,
    load_adaface, get_adaface_embedding,
    load_elasticface, get_elasticface_embedding,
    load_sphereface, get_sphereface_embedding,
)
from poison import (
    pgd_ensemble_poison, load_image_tensor, tensor_to_pil,
    compute_ssim, compute_lpips,
)
from dataset import build_dataset
from evaluate_baselines import pgd_fawkes_style

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

THRESHOLD = 0.3          # same as all pilot scripts
METHODS = ['single', 'ensemble', 'fawkes', 'lowkey']
TARGETS = ['facenet', 'vggface', 'arcface', 'adaface', 'elasticface', 'sphereface']


# ── Same metric as the pilot scripts ───────────────────────────────────────
def cosine_distance(a, b):
    a = F.normalize(a.float().flatten(), dim=0)
    b = F.normalize(b.float().flatten(), dim=0)
    return 1 - torch.dot(a, b).item()


def seed_all(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)


# ── ArcFace ONNX: identical computation to the pilot, plus detection flag ──
def arcface_onnx_embed(app, t):
    pil = tensor_to_pil(t)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        pil.save(f.name)
        tmp = f.name
    img_cv = cv2.imread(tmp)
    os.unlink(tmp)
    img_cv = cv2.resize(img_cv, (160, 160))
    faces = app.get(img_cv)
    if len(faces) == 0:
        return torch.zeros(512, device=device), False
    emb = torch.tensor(faces[0].embedding, dtype=torch.float32, device=device)
    return emb / emb.norm(), True


def load_done(path):
    done = {}
    if not os.path.exists(path):
        return done
    bad = 0
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1          # half-written line from a crash/power cut
                continue
            done[r['index']] = r
    if bad:
        # Rewrite the file with only the complete records, so new lines
        # don't get glued onto the broken fragment. Those images are redone.
        print(f"  (removed {bad} incomplete line(s) from a previous run; "
              f"those images will be redone)")
        with open(path, 'w') as f:
            for k in sorted(done):
                f.write(json.dumps(done[k]) + '\n')
    return done


def summarise(records, out_path):
    summary = {'n_images': len(records),
               'n_identities': len({r['identity'] for r in records})}
    for m in METHODS:
        s = {}
        for k in ['time', 'ssim', 'lpips']:
            v = [r[m][k] for r in records]
            s[k] = {'mean': float(np.mean(v)), 'std': float(np.std(v))}
        for t in TARGETS:
            s[f'pr_{t}'] = float(np.mean([r[m][f'pr_{t}'] for r in records]) * 100)
        s['arcface_no_face'] = int(sum(1 for r in records
                                       if not r[m]['arcface_face_detected']))
        summary[m] = s
    summary['arcface_no_face_original'] = int(
        sum(1 for r in records if not r['arcface_face_detected_original']))
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print(f"SUMMARY -- {summary['n_images']} images, "
          f"{summary['n_identities']} identities")
    print("=" * 78)
    print(f"{'':<16}" + "".join(f"{m:>15}" for m in METHODS))
    for t in TARGETS:
        print(f"{'PR ' + t:<16}" + "".join(
            f"{summary[m][f'pr_{t}']:>14.1f}%" for m in METHODS))
    for k in ['ssim', 'lpips', 'time']:
        print(f"{k.upper():<16}" + "".join(
            f"{summary[m][k]['mean']:>15.4f}" for m in METHODS))
    print(f"\nArcFace no-face cases -- original images: "
          f"{summary['arcface_no_face_original']}, protected: "
          + ", ".join(f"{m} {summary[m]['arcface_no_face']}" for m in METHODS))
    print(f"Summary saved to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_images', type=int, default=500)
    ap.add_argument('--output_dir', default=os.path.join('experiments', 'eval_full'))
    ap.add_argument('--save_images', action='store_true')
    args = ap.parse_args()

    n = args.n_images
    os.makedirs(args.output_dir, exist_ok=True)
    img_dir = os.path.join(args.output_dir, f'images_{n}')
    if args.save_images:
        os.makedirs(img_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f'results_full_{n}images.jsonl')
    summary_path = os.path.join(args.output_dir, f'summary_full_{n}images.json')

    # ── Models ────────────────────────────────────────────────────────────
    print("Loading proxy models...")
    facenet = load_facenet()
    vggface = load_vggface_pytorch()
    arcface_pt = load_arcface_pytorch()
    if arcface_pt is None:
        raise SystemExit("ERROR: facexlib ArcFace proxy is required (no ONNX fallback).")

    print("\nLoading ArcFace ONNX (eval only)...")
    arcface_onnx = load_arcface()

    print("\nLoading held-out targets...")
    adaface = load_adaface()
    elasticface = load_elasticface()
    sphereface = load_sphereface()
    if adaface is None or elasticface is None or sphereface is None:
        raise SystemExit("ERROR: all three held-out models must load. See messages above.")

    def facenet_fn(x): return facenet(x * 2 - 1)
    def vggface_fn(x): return get_vggface_pytorch_embedding(vggface, x)
    def arcface_fn(x): return get_arcface_pytorch_embedding(arcface_pt, x)

    proxies = {
        'single':   [(facenet_fn, 1.0)],
        'ensemble': [(facenet_fn, 1/3), (vggface_fn, 1/3), (arcface_fn, 1/3)],
        'lowkey':   [(facenet_fn, 0.5), (arcface_fn, 0.5)],
    }

    def embed_all(t):
        """Embeddings from every target model. Returns (dict, arcface_detected)."""
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

    # ── Dataset (same call as the pilot; slice keeps pilot as first 20) ────
    all_data = build_dataset()
    eval_data = all_data[:n]
    print(f"Evaluating {len(eval_data)} images, "
          f"{len({e['identity'] for e in eval_data})} identities")

    # Fawkes target pool -- same rule as evaluate_baselines.py
    eval_ids = {e['identity'] for e in eval_data}
    pool_paths = [e['path'] for e in all_data[n:]
                  if e['identity'] not in eval_ids][:50]
    print(f"Fawkes target pool: {len(pool_paths)} images")
    target_embeddings = []
    for p in pool_paths:
        with torch.no_grad():
            target_embeddings.append(facenet_fn(load_image_tensor(p)).detach())

    # ── Resume ────────────────────────────────────────────────────────────
    done = load_done(results_path)
    if done:
        print(f"Resuming: {len(done)} images already done, skipping them.")

    session_times = []
    with open(results_path, 'a') as out:
        for i, entry in enumerate(eval_data):
            if i in done:
                continue
            t_start = time.time()
            identity = entry['identity']
            img = load_image_tensor(entry['path'])

            # Generate every protected version (same seed per method, as pilot)
            protected, gen_time = {}, {}
            for m in ['single', 'ensemble', 'fawkes', 'lowkey']:
                seed_all(42 + i)
                t0 = time.time()
                if m == 'fawkes':
                    target_emb = random.Random(42 + i).choice(target_embeddings)
                    protected[m] = pgd_fawkes_style(img, facenet_fn, target_emb)
                else:
                    protected[m] = pgd_ensemble_poison(img, proxies[m])
                gen_time[m] = time.time() - t0

            emb_orig, det_orig = embed_all(img)
            record = {'index': i, 'identity': identity, 'img_path': entry['path'],
                      'arcface_face_detected_original': det_orig}

            for m in METHODS:
                x = protected[m]
                emb_p, det_p = embed_all(x)
                r = {'time': round(gen_time[m], 2),
                     'ssim': round(float(compute_ssim(img, x)), 4),
                     'lpips': round(float(compute_lpips(img, x)), 4),
                     'arcface_face_detected': det_p}
                for t in TARGETS:
                    d = cosine_distance(emb_orig[t], emb_p[t])
                    r[f'pr_{t}'] = int(d > THRESHOLD)
                    r[f'dist_{t}'] = round(float(d), 4)
                record[m] = r
                if args.save_images:
                    tensor_to_pil(x).save(os.path.join(img_dir, f'{i:04d}_{m}.png'))
            if args.save_images:
                tensor_to_pil(img).save(os.path.join(img_dir, f'{i:04d}_original.png'))

            out.write(json.dumps(record) + '\n')
            out.flush()
            os.fsync(out.fileno())
            done[i] = record

            # Progress + ETA
            session_times.append(time.time() - t_start)
            remaining = n - len(done)
            eta_h = np.mean(session_times) * remaining / 3600
            pr = record['ensemble']
            print(f"[{len(done)}/{n}] {identity:<28} "
                  f"ens PR FN/VGG/Arc/Ada/Ela/Sph: "
                  f"{pr['pr_facenet']}{pr['pr_vggface']}{pr['pr_arcface']}"
                  f"{pr['pr_adaface']}{pr['pr_elasticface']}{pr['pr_sphereface']} "
                  f"| {session_times[-1]:.0f}s/img | ETA {eta_h:.1f}h")

    records = [done[k] for k in sorted(done)]
    summarise(records, summary_path)
    print(f"Per-image results: {results_path}")


if __name__ == '__main__':
    main()
