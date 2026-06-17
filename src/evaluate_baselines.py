"""
O5 — Baseline Comparison: Ours vs Fawkes-style vs LowKey-style

Implements lightweight PyTorch approximations of Fawkes and LowKey
under IDENTICAL conditions (same dataset, same hardware, same evaluation
protocol, same target models) as your main evaluate.py.

Fawkes-style:
  - Single proxy: FaceNet only (matches Fawkes' original proxy choice)
  - Targeted feature collision: PGD moves embedding TOWARD a randomly
    sampled target identity (different person) rather than just away from
    original. This is Fawkes' actual published mechanism (Shan et al., 2020).
  - Same ε=8/255, α=2/255, 40 iterations, quality penalty.

LowKey-style:
  - Two-model ensemble: FaceNet + ArcFace (matches LowKey's published
    proxy setup: Cherepanova et al., 2021).
  - Untargeted divergence loss (same as your main pipeline).
  - Same ε=8/255, α=2/255, 40 iterations, quality penalty.

Your method (Ensemble, 3-model):
  - FaceNet + VGG-Face + ArcFace (re-run from evaluate.py for side-by-side).

All three evaluated against the SAME target models:
  FaceNet (proxy, white-box), ArcFace ONNX (buffalo_l, eval-only),
  VGG-Face (proxy for ours/lowkey, eval for fawkes).

Run:  python src\\evaluate_baselines.py
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import random

from models import (
    load_facenet, load_arcface, get_arcface_embedding,
    load_vggface_pytorch, get_vggface_pytorch_embedding,
    load_arcface_pytorch, get_arcface_pytorch_embedding,
)
from poison import (
    pgd_ensemble_poison, load_image_tensor,
    compute_ssim, compute_lpips, loss_fn_lpips
)
from dataset import build_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ── Fawkes-style: targeted feature collision ────────────────────────────────
def pgd_fawkes_style(
    img_tensor,        # [1,3,H,W] original image to protect
    facenet_fn,        # FaceNet embedding function
    target_emb,        # [1,512] embedding of the target (different) identity
    eps=8/255,
    alpha=2/255,
    n_iter=40,
    quality_weight=0.5
):
    """
    Fawkes-style feature collision:
    PGD minimizes cosine distance between x_adv and target_emb,
    causing the model to confuse the protected image with a different person.

    Unlike the main pipeline (which maximizes distance from self),
    this moves the embedding TOWARD a target identity -- the published
    Fawkes mechanism.
    """
    from poison import loss_fn_lpips

    x = img_tensor.clone().detach().to(device)
    x_adv = x.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0, 1).detach()

    target = target_emb.detach().to(device)

    for _ in range(n_iter):
        x_adv.requires_grad_(True)

        emb_adv = facenet_fn(x_adv)

        # minimize cosine distance to target (feature collision)
        cos_sim_to_target = F.cosine_similarity(
            emb_adv, target, dim=-1
        ).mean()
        # MINIMIZE distance to target = maximize similarity = NEGATE
        # We do gradient DESCENT on total_loss (which uses .backward() + sign)
        # so we MINIMIZE: -cos_sim_to_target => moves toward target
        loss = -cos_sim_to_target

        # quality penalty
        quality_penalty = loss_fn_lpips(x * 2 - 1, x_adv * 2 - 1)
        total_loss = loss + quality_weight * quality_penalty

        total_loss.backward()

        with torch.no_grad():
            grad = x_adv.grad.sign()
            x_adv = x_adv - alpha * grad   # gradient DESCENT
            delta = torch.clamp(x_adv - x, -eps, eps)
            x_adv = torch.clamp(x + delta, 0, 1).detach()

    return x_adv


# ── Evaluation helpers (same as evaluate.py) ───────────────────────────────
def cosine_distance(a, b):
    a = F.normalize(a.float().flatten(), dim=0)
    b = F.normalize(b.float().flatten(), dim=0)
    return 1 - torch.dot(a, b).item()


def is_misidentified(emb_orig, emb_poisoned, threshold=0.3):
    dist = cosine_distance(emb_orig, emb_poisoned)
    return dist > threshold, dist


def evaluate_image(img, img_protected, arcface_onnx, facenet, vggface):
    """Compute PR against FaceNet, ArcFace ONNX, VGG-Face."""
    import tempfile, cv2, os

    results = {}

    # FaceNet
    def fn_embed(x): return facenet(x * 2 - 1)
    emb_o = fn_embed(img).squeeze(0)
    emb_p = fn_embed(img_protected).squeeze(0)
    mis, dist = is_misidentified(emb_o, emb_p)
    results['pr_facenet'] = int(mis)
    results['dist_facenet'] = round(float(dist), 4)

    # ArcFace ONNX (buffalo_l — eval-only, unseen)
    from poison import tensor_to_pil
    def arc_embed(t):
        pil = tensor_to_pil(t)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            pil.save(f.name); tmp = f.name
        img_cv = cv2.imread(tmp); os.unlink(tmp)
        img_cv = cv2.resize(img_cv, (160, 160))
        faces = arcface_onnx.get(img_cv)
        if not faces: return torch.zeros(1, 512, device=device)
        emb = torch.tensor(faces[0].embedding, dtype=torch.float32, device=device)
        return (emb / emb.norm()).unsqueeze(0)
    emb_o = arc_embed(img).squeeze(0)
    emb_p = arc_embed(img_protected).squeeze(0)
    mis, dist = is_misidentified(emb_o, emb_p)
    results['pr_arcface'] = int(mis)
    results['dist_arcface'] = round(float(dist), 4)

    # VGG-Face
    emb_o = get_vggface_pytorch_embedding(vggface, img).squeeze(0)
    emb_p = get_vggface_pytorch_embedding(vggface, img_protected).squeeze(0)
    mis, dist = is_misidentified(emb_o, emb_p)
    results['pr_vggface'] = int(mis)
    results['dist_vggface'] = round(float(dist), 4)

    # Quality metrics
    results['ssim']  = round(float(compute_ssim(img, img_protected)), 4)
    results['lpips'] = round(float(compute_lpips(img, img_protected)), 4)

    return results


def run_baseline_eval(n_images=20, output_dir='experiments/eval'):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading models...")
    facenet      = load_facenet()
    arcface_onnx = load_arcface()
    vggface      = load_vggface_pytorch()
    arcface_pt   = load_arcface_pytorch()

    if arcface_pt is None:
        print("ERROR: facexlib ArcFace proxy needed.")
        return

    def facenet_fn(x):  return facenet(x * 2 - 1)
    def vggface_fn(x):  return get_vggface_pytorch_embedding(vggface, x)
    def arcface_fn(x):  return get_arcface_pytorch_embedding(arcface_pt, x)

    # Pre-compute a pool of target embeddings for Fawkes collision
    # (use a random OTHER identity's first image from the dataset)
    print(f"\nLoading dataset ({n_images} images)...")
    all_data = build_dataset()
    eval_data = all_data[:n_images]

    # Build target pool: one embedding per identity NOT in eval set
    eval_identities = set(e['identity'] for e in eval_data)
    target_pool_paths = [
        e['path'] for e in all_data[n_images:]
        if e['identity'] not in eval_identities
    ][:50]  # first 50 non-eval identities as target pool

    print(f"Target pool for Fawkes collision: {len(target_pool_paths)} images")
    target_embeddings = []
    for p in target_pool_paths:
        t = load_image_tensor(p)
        with torch.no_grad():
            emb = facenet_fn(t)
        target_embeddings.append(emb.detach())

    results = []

    for i, entry in enumerate(eval_data):
        img_path = entry['path']
        identity = entry['identity']
        print(f"\n[{i+1}/{n_images}] {identity}")

        img = load_image_tensor(img_path)

        # Pick a random target for Fawkes (different identity)
        rng = random.Random(42 + i)
        target_emb = rng.choice(target_embeddings)

        # ── Method 1: Fawkes-style (FaceNet, targeted collision) ──────────
        torch.manual_seed(42 + i)
        if torch.cuda.is_available(): torch.cuda.manual_seed(42 + i)
        t0 = time.time()
        img_fawkes = pgd_fawkes_style(img, facenet_fn, target_emb)
        t_fawkes = time.time() - t0

        # ── Method 2: LowKey-style (FaceNet + ArcFace, untargeted) ───────
        torch.manual_seed(42 + i)
        if torch.cuda.is_available(): torch.cuda.manual_seed(42 + i)
        t0 = time.time()
        img_lowkey = pgd_ensemble_poison(
            img,
            [(facenet_fn, 0.5), (arcface_fn, 0.5)],
        )
        t_lowkey = time.time() - t0

        # ── Method 3: Ours (FaceNet + VGG-Face + ArcFace, untargeted) ────
        torch.manual_seed(42 + i)
        if torch.cuda.is_available(): torch.cuda.manual_seed(42 + i)
        t0 = time.time()
        img_ours = pgd_ensemble_poison(
            img,
            [(facenet_fn, 1/3), (vggface_fn, 1/3), (arcface_fn, 1/3)],
        )
        t_ours = time.time() - t0

        print(f"  Fawkes-style  time: {t_fawkes:.1f}s")
        print(f"  LowKey-style  time: {t_lowkey:.1f}s")
        print(f"  Ours          time: {t_ours:.1f}s")

        # Evaluate all three against same target models
        r_fawkes = evaluate_image(img, img_fawkes, arcface_onnx, facenet, vggface)
        r_lowkey = evaluate_image(img, img_lowkey, arcface_onnx, facenet, vggface)
        r_ours   = evaluate_image(img, img_ours,   arcface_onnx, facenet, vggface)

        r_fawkes['time'] = round(float(t_fawkes), 2)
        r_lowkey['time'] = round(float(t_lowkey), 2)
        r_ours['time']   = round(float(t_ours), 2)

        for name, r in [('fawkes', r_fawkes), ('lowkey', r_lowkey), ('ours', r_ours)]:
            print(f"  [{name:>6}] PR(FN):{r['pr_facenet']} "
                  f"PR(Arc):{r['pr_arcface']} PR(VGG):{r['pr_vggface']} "
                  f"SSIM:{r['ssim']:.4f} LPIPS:{r['lpips']:.4f}")

        results.append({
            'identity': identity,
            'img_path': img_path,
            'fawkes': r_fawkes,
            'lowkey': r_lowkey,
            'ours':   r_ours,
        })

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("O5 BASELINE COMPARISON SUMMARY")
    print("="*65)
    print(f"\n{'Metric':<20} {'Fawkes-style':>14} {'LowKey-style':>14} {'Ours (3-model)':>15}")
    print("-"*65)

    metrics = [
        ('PR FaceNet',  'pr_facenet',  True,  100),
        ('PR ArcFace',  'pr_arcface',  True,  100),
        ('PR VGG-Face', 'pr_vggface',  True,  100),
        ('SSIM',        'ssim',        False, 1),
        ('LPIPS',       'lpips',       False, 1),
        ('Time (s/img)', 'time',       False, 1),
    ]

    summary = {}
    for label, key, is_pr, scale in metrics:
        vals = {
            m: np.mean([r[m][key] for r in results]) * scale
            for m in ['fawkes', 'lowkey', 'ours']
        }
        summary[key] = vals
        if is_pr:
            print(f"{label:<20} {vals['fawkes']:>13.1f}% "
                  f"{vals['lowkey']:>13.1f}% {vals['ours']:>14.1f}%")
        else:
            print(f"{label:<20} {vals['fawkes']:>14.4f} "
                  f"{vals['lowkey']:>14.4f} {vals['ours']:>15.4f}")

    print("="*65)

    out_path = os.path.join(output_dir, f'results_baselines_{n_images}images.json')
    with open(out_path, 'w') as f:
        json.dump({'results': results, 'summary': summary}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    run_baseline_eval(n_images=20)