import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import tempfile
import cv2
from PIL import Image

from models import (
    load_facenet, load_arcface,
    load_vggface_pytorch, get_vggface_pytorch_embedding
)
from poison import (
    pgd_ensemble_poison, load_image_tensor,
    tensor_to_pil, compute_ssim, compute_lpips
)
from dataset import build_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_py(x):
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    return x

# ── Protection Rate Measurement ────────────────────────────────────────────
def cosine_distance(a, b):
    a = F.normalize(a.float().flatten(), dim=0)
    b = F.normalize(b.float().flatten(), dim=0)
    return 1 - torch.dot(a, b).item()

def is_misidentified(emb_orig, emb_poisoned, threshold=0.3):
    dist = cosine_distance(emb_orig, emb_poisoned)
    return dist > threshold, dist

# ── Target model embedding extractors ─────────────────────────────────────
def get_target_facenet_emb(facenet, img_tensor):
    x_norm = img_tensor * 2 - 1
    with torch.no_grad():
        emb = facenet(x_norm)
    return emb.squeeze(0)

def get_target_arcface_emb(arcface, img_tensor):
    pil = tensor_to_pil(img_tensor)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        pil.save(f.name)
        tmp = f.name
    img_cv = cv2.imread(tmp)
    os.unlink(tmp)
    img_cv = cv2.resize(img_cv, (160, 160))
    faces = arcface.get(img_cv)
    if len(faces) == 0:
        return torch.zeros(512).to(device)
    emb = torch.tensor(faces[0].embedding, dtype=torch.float32).to(device)
    return emb / emb.norm()

def get_target_vggface_emb(vggface, img_tensor):
    with torch.no_grad():
        emb = get_vggface_pytorch_embedding(vggface, img_tensor)
    return emb.squeeze(0)

# ── Main Evaluation ────────────────────────────────────────────────────────
def run_evaluation(n_images=20, output_dir='experiments/eval'):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading proxy models...")
    facenet = load_facenet()
    arcface = load_arcface()
    vggface = load_vggface_pytorch()

    def facenet_embed(x):
        return facenet(x * 2 - 1)

    def vggface_embed(x):
        return get_vggface_pytorch_embedding(vggface, x)

    def arcface_embed(x):
        pil = tensor_to_pil(x)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            pil.save(f.name)
            tmp = f.name
        img_cv = cv2.imread(tmp)
        os.unlink(tmp)
        img_cv = cv2.resize(img_cv, (160, 160))
        faces = arcface.get(img_cv)
        if len(faces) == 0:
            return torch.zeros(1, 512).to(device)
        emb = torch.tensor(faces[0].embedding, dtype=torch.float32).to(device)
        return (emb / emb.norm()).unsqueeze(0)

    ensemble_proxies = [
        (facenet_embed, 1/3),
        (vggface_embed, 1/3),
        (arcface_embed, 1/3),
    ]
    single_proxy = [(facenet_embed, 1.0)]

    print(f"Loading dataset ({n_images} images)...")
    dataset = build_dataset()[:n_images]

    results = []

    for i, entry in enumerate(dataset):
        img_path = entry['path']
        identity = entry['identity']
        print(f"\n[{i+1}/{n_images}] {identity}")

        img = load_image_tensor(img_path)

        t0 = time.time()
        img_single = pgd_ensemble_poison(img, single_proxy)
        t_single = time.time() - t0

        t0 = time.time()
        img_ensemble = pgd_ensemble_poison(img, ensemble_proxies)
        t_ensemble = time.time() - t0

        ssim_single  = float(compute_ssim(img, img_single))
        lpips_single = float(compute_lpips(img, img_single))
        ssim_ensemble  = float(compute_ssim(img, img_ensemble))
        lpips_ensemble = float(compute_lpips(img, img_ensemble))

        emb_orig_fn      = get_target_facenet_emb(facenet, img)
        emb_single_fn    = get_target_facenet_emb(facenet, img_single)
        emb_ensemble_fn  = get_target_facenet_emb(facenet, img_ensemble)
        mis_single_fn,   dist_single_fn   = is_misidentified(emb_orig_fn, emb_single_fn)
        mis_ensemble_fn, dist_ensemble_fn = is_misidentified(emb_orig_fn, emb_ensemble_fn)

        emb_orig_arc      = get_target_arcface_emb(arcface, img)
        emb_single_arc    = get_target_arcface_emb(arcface, img_single)
        emb_ensemble_arc  = get_target_arcface_emb(arcface, img_ensemble)
        mis_single_arc,   dist_single_arc   = is_misidentified(emb_orig_arc, emb_single_arc)
        mis_ensemble_arc, dist_ensemble_arc = is_misidentified(emb_orig_arc, emb_ensemble_arc)

        emb_orig_vgg      = get_target_vggface_emb(vggface, img)
        emb_single_vgg    = get_target_vggface_emb(vggface, img_single)
        emb_ensemble_vgg  = get_target_vggface_emb(vggface, img_ensemble)
        mis_single_vgg,   dist_single_vgg   = is_misidentified(emb_orig_vgg, emb_single_vgg)
        mis_ensemble_vgg, dist_ensemble_vgg = is_misidentified(emb_orig_vgg, emb_ensemble_vgg)

        result = {
            'identity': identity,
            'img_path': img_path,
            'single': {
                'time':         round(float(t_single), 2),
                'ssim':         round(ssim_single, 4),
                'lpips':        round(lpips_single, 4),
                'pr_facenet':   int(mis_single_fn),
                'dist_facenet': round(float(dist_single_fn), 4),
                'pr_arcface':   int(mis_single_arc),
                'dist_arcface': round(float(dist_single_arc), 4),
                'pr_vggface':   int(mis_single_vgg),
                'dist_vggface': round(float(dist_single_vgg), 4),
            },
            'ensemble': {
                'time':         round(float(t_ensemble), 2),
                'ssim':         round(ssim_ensemble, 4),
                'lpips':        round(lpips_ensemble, 4),
                'pr_facenet':   int(mis_ensemble_fn),
                'dist_facenet': round(float(dist_ensemble_fn), 4),
                'pr_arcface':   int(mis_ensemble_arc),
                'dist_arcface': round(float(dist_ensemble_arc), 4),
                'pr_vggface':   int(mis_ensemble_vgg),
                'dist_vggface': round(float(dist_ensemble_vgg), 4),
            }
        }
        results.append(result)

        print(f"  Single  — SSIM: {ssim_single:.4f} LPIPS: {lpips_single:.4f} "
              f"PR(FN): {int(mis_single_fn)} PR(Arc): {int(mis_single_arc)} "
              f"PR(VGG): {int(mis_single_vgg)} Time: {t_single:.1f}s")
        print(f"  Ensemble — SSIM: {ssim_ensemble:.4f} LPIPS: {lpips_ensemble:.4f} "
              f"PR(FN): {int(mis_ensemble_fn)} PR(Arc): {int(mis_ensemble_arc)} "
              f"PR(VGG): {int(mis_ensemble_vgg)} Time: {t_ensemble:.1f}s")

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for condition in ['single', 'ensemble']:
        times  = [r[condition]['time']  for r in results]
        ssims  = [r[condition]['ssim']  for r in results]
        lpipss = [r[condition]['lpips'] for r in results]
        pr_fn  = [r[condition]['pr_facenet']  for r in results]
        pr_arc = [r[condition]['pr_arcface']  for r in results]
        pr_vgg = [r[condition]['pr_vggface']  for r in results]

        print(f"\n{condition.upper()} PROXY:")
        print(f"  Time:        {np.mean(times):.1f}s ± {np.std(times):.1f}s")
        print(f"  SSIM:        {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
        print(f"  LPIPS:       {np.mean(lpipss):.4f} ± {np.std(lpipss):.4f}")
        print(f"  PR FaceNet:  {np.mean(pr_fn)*100:.1f}%")
        print(f"  PR ArcFace:  {np.mean(pr_arc)*100:.1f}%")
        print(f"  PR VGG-Face: {np.mean(pr_vgg)*100:.1f}%")

    out_path = os.path.join(output_dir, f'results_{n_images}images.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    run_evaluation(n_images=20)