import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import lpips

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Image utilities ────────────────────────────────────────────────────────
def load_image_tensor(img_path, size=(160, 160)):
    img = Image.open(img_path).convert('RGB').resize(size)
    t = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return t  # [1, 3, H, W] in [0, 1]

def tensor_to_pil(t):
    t = t.squeeze(0).clamp(0, 1).cpu()
    return transforms.ToPILImage()(t)

# ── Perceptual quality metrics ─────────────────────────────────────────────
loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

def compute_ssim(img1, img2):
    from skimage.metrics import structural_similarity as ssim
    i1 = img1.squeeze(0).permute(1,2,0).cpu().numpy()
    i2 = img2.squeeze(0).permute(1,2,0).cpu().numpy()
    return ssim(i1, i2, channel_axis=2, data_range=1.0)

def compute_lpips(img1, img2):
    with torch.no_grad():
        return loss_fn_lpips(img1*2-1, img2*2-1).item()

# ── Ensemble PGD poisoning ─────────────────────────────────────────────────
def pgd_ensemble_poison(
    img_tensor,          # [1,3,H,W] original image
    proxy_models,        # list of (model_fn, weight) tuples
    eps=8/255,           # L-inf perturbation budget
    alpha=2/255,         # step size
    n_iter=40,           # PGD iterations
    weights=None         # per-model weights (None = uniform)
):
    """
    Generate adversarial perturbation via PGD over ensemble of proxy models.
    Maximizes feature-space divergence across all proxies simultaneously.
    """
    if weights is None:
        weights = [1.0 / len(proxy_models)] * len(proxy_models)

    x = img_tensor.clone().detach().to(device)
    x_adv = x.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0, 1).detach()

    for i in range(n_iter):
        x_adv.requires_grad_(True)
        
        total_loss = torch.tensor(0.0, device=device)
        
        for (model_fn, model_weight), w in zip(proxy_models, weights):
            emb_orig = model_fn(x)
            emb_adv  = model_fn(x_adv)
            # maximize cosine distance between original and poisoned embeddings
            cos_sim = F.cosine_similarity(emb_orig, emb_adv, dim=-1).mean()
            loss = cos_sim * w  # minimize cosine similarity = maximize divergence
            total_loss = total_loss + loss

        total_loss.backward()
        
        with torch.no_grad():
            grad = x_adv.grad.sign()
            x_adv = x_adv - alpha * grad  # gradient descent on similarity
            # project back into epsilon ball
            delta = torch.clamp(x_adv - x, -eps, eps)
            x_adv = torch.clamp(x + delta, 0, 1).detach()

    return x_adv

# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from models import load_facenet, load_arcface, load_vggface_pytorch, get_vggface_pytorch_embedding
    import time, tempfile, os, cv2

    test_img = r'C:\projects\FacialPrivacyShield\data\lfw-dataset\lfw-deepfunneled\lfw-deepfunneled\Richard_Myers\Richard_Myers_0004.jpg'

    print("Loading proxy models...")
    facenet = load_facenet()
    arcface = load_arcface()
    vggface = load_vggface_pytorch()

    def facenet_embed(x):
        x_norm = x * 2 - 1
        return facenet(x_norm)

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
        emb = emb / emb.norm()
        return emb.unsqueeze(0)

    img = load_image_tensor(test_img)

    print("\n--- Single Proxy (FaceNet only) ---")
    t0 = time.time()
    img_single = pgd_ensemble_poison(img, [(facenet_embed, 1.0)])
    t1 = time.time()
    print(f"Time:  {t1-t0:.1f}s")
    print(f"SSIM:  {compute_ssim(img, img_single):.4f}")
    print(f"LPIPS: {compute_lpips(img, img_single):.4f}")

    print("\n--- Ensemble (FaceNet + VGG-Face + ArcFace) ---")
    t0 = time.time()
    img_ensemble = pgd_ensemble_poison(img, [
        (facenet_embed, 1/3),
        (vggface_embed, 1/3),
        (arcface_embed, 1/3),
    ])
    t1 = time.time()
    print(f"Time:  {t1-t0:.1f}s")
    print(f"SSIM:  {compute_ssim(img, img_ensemble):.4f}")
    print(f"LPIPS: {compute_lpips(img, img_ensemble):.4f}")

    tensor_to_pil(img).save('experiments/original.jpg')
    tensor_to_pil(img_single).save('experiments/poisoned_single.jpg')
    tensor_to_pil(img_ensemble).save('experiments/poisoned_ensemble.jpg')
    print("\nSaved original, poisoned_single, poisoned_ensemble to experiments/")