"""
Determine the correct input preprocessing for AdaFace (CVLFace IR101-WebFace4M).

Loads two images of the SAME identity and one image of a DIFFERENT identity.
For each of the 3 candidate preprocessing configs, computes embeddings and checks:
  - same-identity cosine similarity should be HIGH (close to 1)
  - different-identity cosine similarity should be LOWER

The config where (same_sim - diff_sim) is largest and same_sim is high
is the correct preprocessing.

Run:  python src\\test_adaface_preprocessing.py
"""
import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reuse the loader from test_adaface.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_adaface import load_adaface

# ── Pick three LFW images: two of same identity, one different ────────────
LFW_BASE = r'C:\projects\FacialPrivacyShield\data\lfw-dataset\lfw-deepfunneled\lfw-deepfunneled'

# Use Richard_Myers (has 5 images) for same-identity pair, and Carlos_Moya for different
SAME_A = os.path.join(LFW_BASE, 'Richard_Myers', 'Richard_Myers_0001.jpg')
SAME_B = os.path.join(LFW_BASE, 'Richard_Myers', 'Richard_Myers_0002.jpg')
DIFF   = os.path.join(LFW_BASE, 'Carlos_Moya', 'Carlos_Moya_0001.jpg')


def load_img_tensor(path, size=112):
    img = Image.open(path).convert('RGB').resize((size, size))
    t = transforms.ToTensor()(img).unsqueeze(0).to(device)  # [1,3,112,112] in [0,1], RGB
    return t


def to_bgr(x):
    return x[:, [2, 1, 0], :, :]


CONFIGS = {
    "RGB, [-1,1]": lambda x: x * 2 - 1,
    "BGR, [-1,1]": lambda x: to_bgr(x) * 2 - 1,
    "RGB, [0,1]":  lambda x: x,
}


if __name__ == '__main__':
    print("Loading AdaFace model...")
    model = load_adaface()

    print("Loading test images...")
    img_a = load_img_tensor(SAME_A)
    img_b = load_img_tensor(SAME_B)
    img_c = load_img_tensor(DIFF)

    print(f"\n{'Config':<15} {'Same-ID sim':>12} {'Diff-ID sim':>12} {'Gap':>8}")
    print("-" * 50)

    for name, fn in CONFIGS.items():
        with torch.no_grad():
            emb_a = model(fn(img_a))
            emb_b = model(fn(img_b))
            emb_c = model(fn(img_c))

        # Normalize for cosine similarity
        emb_a = F.normalize(emb_a, dim=-1)
        emb_b = F.normalize(emb_b, dim=-1)
        emb_c = F.normalize(emb_c, dim=-1)

        sim_same = F.cosine_similarity(emb_a, emb_b).item()
        sim_diff = F.cosine_similarity(emb_a, emb_c).item()
        gap = sim_same - sim_diff

        print(f"{name:<15} {sim_same:>12.4f} {sim_diff:>12.4f} {gap:>8.4f}")

    print("\nThe config with HIGH same-ID sim, LOW diff-ID sim, and LARGEST gap is correct.")
    print("A correct face recognition model: same-ID sim should be > 0.3-0.5,")
    print("diff-ID sim should be noticeably lower (often < 0.3 or even negative).")