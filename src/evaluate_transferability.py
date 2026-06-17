"""
O3 — Cross-Architecture Transferability Evaluation

Tests whether perturbations generated against the proxy ensemble
(FaceNet + VGG-Face + facexlib ArcFace) transfer to HELD-OUT models
NEVER seen during PGD optimization:
  - AdaFace   (IR-101, WebFace4M, via CVLFace/HuggingFace)
  - ElasticFace (IR-101, ElasticFace-Cos+, MS1MV2)

This is the core O3 measurement: does the ensemble proxy improve
protection on held-out, unseen FR architectures compared to a
single-proxy baseline?

Reuses the EXACT same PGD calls (same seeds, same proxy weights) as
evaluate.py, so the poisoned images are identical to your locked
20-image pilot -- only the EVALUATION TARGETS differ.

Run:  python src\\evaluate_transferability.py
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import time

from models import (
    load_facenet, load_vggface_pytorch, get_vggface_pytorch_embedding,
    load_arcface_pytorch, get_arcface_pytorch_embedding,
    load_adaface, get_adaface_embedding,
    load_elasticface, get_elasticface_embedding,
    load_sphereface, get_sphereface_embedding,
)
from poison import pgd_ensemble_poison, load_image_tensor
from dataset import build_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Determinism: some PGD perturbations land very close to the PR threshold
# (e.g. one image's ensemble distance was 0.3005 vs threshold 0.3 -- a
# difference of 0.0005). cuDNN's default algorithm selection can introduce
# tiny run-to-run floating-point variation that would flip such borderline
# cases. These settings make convolution algorithms deterministic so the
# same seed always produces the same perturbation and the same PR result.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def cosine_distance(a, b):
    a = F.normalize(a.float().flatten(), dim=0)
    b = F.normalize(b.float().flatten(), dim=0)
    return 1 - torch.dot(a, b).item()


def is_misidentified(emb_orig, emb_poisoned, threshold=0.3):
    dist = cosine_distance(emb_orig, emb_poisoned)
    return dist > threshold, dist


def run_transferability_eval(n_images=20, output_dir='experiments/eval'):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading proxy models (same as evaluate.py)...")
    facenet    = load_facenet()
    vggface    = load_vggface_pytorch()
    arcface_pt = load_arcface_pytorch()

    if arcface_pt is None:
        print("ERROR: facexlib ArcFace proxy failed to load.")
        print("This script requires the differentiable ArcFace proxy to")
        print("reproduce the same perturbations as the locked pilot.")
        return

    def facenet_embed(x):
        return facenet(x * 2 - 1)

    def vggface_embed(x):
        return get_vggface_pytorch_embedding(vggface, x)

    def arcface_embed(x):
        return get_arcface_pytorch_embedding(arcface_pt, x)

    ensemble_proxies = [
        (facenet_embed, 1/3),
        (vggface_embed, 1/3),
        (arcface_embed, 1/3),
    ]
    single_proxy = [(facenet_embed, 1.0)]

    # ── Held-out evaluation targets ────────────────────────────────────────
    print("\nLoading AdaFace (held-out target)...")
    adaface = load_adaface()

    print("\nLoading ElasticFace (held-out target)...")
    elasticface = load_elasticface()

    print("\nLoading SphereFace (held-out target)...")
    sphereface = load_sphereface()

    held_out = {}
    if adaface is not None:
        held_out['adaface'] = (adaface, get_adaface_embedding)
    if elasticface is not None:
        held_out['elasticface'] = (elasticface, get_elasticface_embedding)
    if sphereface is not None:
        held_out['sphereface'] = (sphereface, get_sphereface_embedding)

    if not held_out:
        print("ERROR: no held-out models available.")
        return

    print(f"\nHeld-out targets active: {list(held_out.keys())}")

    print(f"\nLoading dataset ({n_images} images)...")
    dataset = build_dataset()[:n_images]

    results = []

    for i, entry in enumerate(dataset):
        img_path = entry['path']
        identity = entry['identity']
        print(f"\n[{i+1}/{n_images}] {identity}")

        img = load_image_tensor(img_path)

        # Same seeding as evaluate.py -- reproduces identical perturbations
        torch.manual_seed(42 + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + i)
        t0 = time.time()
        img_single = pgd_ensemble_poison(img, single_proxy)
        t_single = time.time() - t0

        torch.manual_seed(42 + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + i)
        t0 = time.time()
        img_ensemble = pgd_ensemble_poison(img, ensemble_proxies)
        t_ensemble = time.time() - t0

        result = {
            'identity': identity,
            'img_path': img_path,
            'single':   {'time': round(float(t_single), 2)},
            'ensemble': {'time': round(float(t_ensemble), 2)},
        }

        # ── Evaluate against each held-out target ──────────────────────────
        for name, (model, embed_fn) in held_out.items():
            emb_orig     = embed_fn(model, img).squeeze(0)
            emb_single   = embed_fn(model, img_single).squeeze(0)
            emb_ensemble = embed_fn(model, img_ensemble).squeeze(0)

            mis_single,   dist_single   = is_misidentified(emb_orig, emb_single)
            mis_ensemble, dist_ensemble = is_misidentified(emb_orig, emb_ensemble)

            result['single'][f'pr_{name}']   = int(mis_single)
            result['single'][f'dist_{name}'] = round(float(dist_single), 4)
            result['ensemble'][f'pr_{name}']   = int(mis_ensemble)
            result['ensemble'][f'dist_{name}'] = round(float(dist_ensemble), 4)

            print(f"  [{name:>11}] Single   — PR: {int(mis_single)}  dist: {dist_single:.4f}")
            print(f"  [{name:>11}] Ensemble — PR: {int(mis_ensemble)}  dist: {dist_ensemble:.4f}")

        print(f"  (time: single {t_single:.1f}s, ensemble {t_ensemble:.1f}s)")
        results.append(result)

    # ── Summary ──
    print("\n" + "="*60)
    print("O3 TRANSFERABILITY SUMMARY -- held-out, unseen architectures")
    print("="*60)

    summary_lines = []
    for name in held_out.keys():
        pr_single   = np.mean([r['single'][f'pr_{name}']   for r in results]) * 100
        pr_ensemble = np.mean([r['ensemble'][f'pr_{name}'] for r in results]) * 100
        dist_single   = np.mean([r['single'][f'dist_{name}']   for r in results])
        dist_ensemble = np.mean([r['ensemble'][f'dist_{name}'] for r in results])

        verdict = ('IMPROVEMENT' if pr_ensemble > pr_single
                   else 'no improvement' if pr_ensemble == pr_single
                   else 'DECLINE')

        print(f"\n{name.upper()}:")
        print(f"  Single   PR: {pr_single:.1f}%   (mean dist {dist_single:.4f})")
        print(f"  Ensemble PR: {pr_ensemble:.1f}%   (mean dist {dist_ensemble:.4f})")
        print(f"  -> {pr_ensemble:.1f}% vs {pr_single:.1f}%  ({verdict})")

        summary_lines.append(f"{name}: {pr_ensemble:.1f}% vs {pr_single:.1f}% ({verdict})")

    print(f"\n{'='*60}")
    print("Overall O3 summary:")
    for line in summary_lines:
        print(f"  {line}")

    out_path = os.path.join(output_dir, f'results_transferability_{n_images}images.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    run_transferability_eval(n_images=20)