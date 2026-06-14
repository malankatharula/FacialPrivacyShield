"""
Verify ElasticFace (IResNet-100) loads correctly and determine the correct
input preprocessing, using the same same-identity/different-identity
discrimination test as AdaFace.

Requires: 295672backbone.pth downloaded manually and placed at
  data/elasticface/295672backbone.pth

Run:  python src\\test_elasticface_preprocessing.py
"""
import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import load_elasticface, get_elasticface_embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LFW_BASE = r'C:\projects\FacialPrivacyShield\data\lfw-dataset\lfw-deepfunneled\lfw-deepfunneled'
SAME_A = os.path.join(LFW_BASE, 'Richard_Myers', 'Richard_Myers_0001.jpg')
SAME_B = os.path.join(LFW_BASE, 'Richard_Myers', 'Richard_Myers_0002.jpg')
DIFF   = os.path.join(LFW_BASE, 'Carlos_Moya', 'Carlos_Moya_0001.jpg')


def load_img_tensor(path, size=112):
    img = Image.open(path).convert('RGB').resize((size, size))
    return transforms.ToTensor()(img).unsqueeze(0).to(device)  # [1,3,112,112] in [0,1], RGB


CONFIGS = [
    ("BGR, [-1,1]", True,  '[-1,1]'),
    ("BGR, [0,1]",  True,  '[0,1]'),
    ("RGB, [-1,1]", False, '[-1,1]'),
    ("RGB, [0,1]",  False, '[0,1]'),
]


if __name__ == '__main__':
    print("Loading ElasticFace model...")
    model = load_elasticface()
    if model is None:
        print("\nCannot proceed without weights. See instructions above.")
        sys.exit(1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.1f}M (expect ~65.2M for IResNet-100)")

    print("\nLoading test images...")
    img_a = load_img_tensor(SAME_A)
    img_b = load_img_tensor(SAME_B)
    img_c = load_img_tensor(DIFF)

    print(f"\n{'Config':<15} {'Same-ID sim':>12} {'Diff-ID sim':>12} {'Gap':>8}")
    print("-" * 50)

    for name, bgr, norm_range in CONFIGS:
        emb_a = get_elasticface_embedding(model, img_a, bgr=bgr, normalize_range=norm_range).squeeze(0)
        emb_b = get_elasticface_embedding(model, img_b, bgr=bgr, normalize_range=norm_range).squeeze(0)
        emb_c = get_elasticface_embedding(model, img_c, bgr=bgr, normalize_range=norm_range).squeeze(0)

        emb_a = F.normalize(emb_a, dim=-1)
        emb_b = F.normalize(emb_b, dim=-1)
        emb_c = F.normalize(emb_c, dim=-1)

        sim_same = F.cosine_similarity(emb_a, emb_b, dim=0).item()
        sim_diff = F.cosine_similarity(emb_a, emb_c, dim=0).item()
        gap = sim_same - sim_diff

        print(f"{name:<15} {sim_same:>12.4f} {sim_diff:>12.4f} {gap:>8.4f}")

    print("\nThe config with HIGH same-ID sim, LOW diff-ID sim, and LARGEST gap is correct.")