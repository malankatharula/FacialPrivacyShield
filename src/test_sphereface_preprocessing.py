"""
Verify SphereFace (sphere20a) loads correctly and determine the correct
input preprocessing, using the same same-identity/different-identity
discrimination test as AdaFace/ElasticFace.

Tests 6 configs: {RGB, BGR} x {[0,1], [-1,1], mean127 (original convention)}.
SphereFace input is NON-SQUARE: 112x96 (height x width).

Requires sphere20a_20171020.pth at data/sphereface/sphere20a_20171020.pth
(extracted from the .7z archive -- see models.py for the download link).

Run:  python src\\test_sphereface_preprocessing.py
"""
import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import load_sphereface, get_sphereface_embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LFW_BASE = r'C:\projects\FacialPrivacyShield\data\lfw-dataset\lfw-deepfunneled\lfw-deepfunneled'
SAME_A = os.path.join(LFW_BASE, 'Richard_Myers', 'Richard_Myers_0001.jpg')
SAME_B = os.path.join(LFW_BASE, 'Richard_Myers', 'Richard_Myers_0002.jpg')
DIFF   = os.path.join(LFW_BASE, 'Carlos_Moya', 'Carlos_Moya_0001.jpg')


def load_img_tensor(path, size=112):
    img = Image.open(path).convert('RGB').resize((size, size))
    return transforms.ToTensor()(img).unsqueeze(0).to(device)  # [1,3,112,112] in [0,1], RGB


CONFIGS = [
    ("RGB, [0,1]",     False, '[0,1]'),
    ("RGB, [-1,1]",    False, '[-1,1]'),
    ("RGB, mean127",   False, 'mean127'),
    ("BGR, [0,1]",     True,  '[0,1]'),
    ("BGR, [-1,1]",    True,  '[-1,1]'),
    ("BGR, mean127",   True,  'mean127'),
]


if __name__ == '__main__':
    print("Loading SphereFace model...")
    model = load_sphereface()
    if model is None:
        print("\nCannot proceed without weights. See instructions above.")
        sys.exit(1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.1f}M (sphere20a w/ AngleLinear fc6, expect ~28-30M)")

    print("\nLoading test images...")
    img_a = load_img_tensor(SAME_A)
    img_b = load_img_tensor(SAME_B)
    img_c = load_img_tensor(DIFF)

    print(f"\n{'Config':<15} {'Same-ID sim':>12} {'Diff-ID sim':>12} {'Gap':>8}")
    print("-" * 50)

    for name, bgr, norm_mode in CONFIGS:
        emb_a = get_sphereface_embedding(model, img_a, bgr=bgr, normalize_mode=norm_mode).squeeze(0)
        emb_b = get_sphereface_embedding(model, img_b, bgr=bgr, normalize_mode=norm_mode).squeeze(0)
        emb_c = get_sphereface_embedding(model, img_c, bgr=bgr, normalize_mode=norm_mode).squeeze(0)

        emb_a = F.normalize(emb_a, dim=-1)
        emb_b = F.normalize(emb_b, dim=-1)
        emb_c = F.normalize(emb_c, dim=-1)

        sim_same = F.cosine_similarity(emb_a, emb_b, dim=0).item()
        sim_diff = F.cosine_similarity(emb_a, emb_c, dim=0).item()
        gap = sim_same - sim_diff

        print(f"{name:<15} {sim_same:>12.4f} {sim_diff:>12.4f} {gap:>8.4f}")

    print("\nThe config with HIGH same-ID sim, LOW diff-ID sim, and LARGEST gap is correct.")