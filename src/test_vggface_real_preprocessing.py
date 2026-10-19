"""
Determine the correct input preprocessing for the REAL VGG-Face proxy
(vggface_real.py), with the same same-identity / different-identity test
used for AdaFace, ElasticFace and SphereFace.

Two checks:
  1. The classic triplet (Richard_Myers 0001 vs 0002, Carlos_Moya 0001),
     identical to the other preprocessing tests, for comparability.
  2. A more robust average over 20 identities from the evaluation dataset:
     same-ID pairs  = image 1 vs image 2 of the same person
     diff-ID pairs  = image 1 of person k vs image 1 of person k+1

The ImageNet VGG-16 currently in models.py is included as a reference row,
to show (for the thesis) how well an object classifier separates faces.

Config with HIGH same-ID sim, LOW diff-ID sim and the LARGEST gap wins.

Run from the project root:  python src\\test_vggface_real_preprocessing.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vggface_real import load_vggface_real, get_vggface_real_embedding
from models import load_vggface_pytorch, get_vggface_pytorch_embedding
from poison import load_image_tensor          # 160x160 RGB [0,1], as in the pipeline
from dataset import build_dataset

LFW_BASE = r'C:\projects\FacialPrivacyShield\data\lfw-dataset\lfw-deepfunneled\lfw-deepfunneled'
SAME_A = os.path.join(LFW_BASE, 'Richard_Myers', 'Richard_Myers_0001.jpg')
SAME_B = os.path.join(LFW_BASE, 'Richard_Myers', 'Richard_Myers_0002.jpg')
DIFF   = os.path.join(LFW_BASE, 'Carlos_Moya', 'Carlos_Moya_0001.jpg')

CONFIGS = [
    ("BGR, [0,1]",    True,  '[0,1]'),     # DeepFace default
    ("BGR, raw255",   True,  'raw255'),
    ("BGR, vggmean",  True,  'vggmean'),   # original VGG-Face convention
    ("RGB, [0,1]",    False, '[0,1]'),
    ("RGB, raw255",   False, 'raw255'),
    ("RGB, vggmean",  False, 'vggmean'),
]


def sim(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def score(embed, triplet, pairs_same, pairs_diff):
    a, b, c = (embed(t) for t in triplet)
    s_same, s_diff = sim(a, b), sim(a, c)
    m_same = sum(sim(embed(x), embed(y)) for x, y in pairs_same) / len(pairs_same)
    m_diff = sum(sim(embed(x), embed(y)) for x, y in pairs_diff) / len(pairs_diff)
    return s_same, s_diff, m_same, m_diff


if __name__ == '__main__':
    model = load_vggface_real()
    imagenet_vgg = load_vggface_pytorch()

    triplet = [load_image_tensor(p) for p in (SAME_A, SAME_B, DIFF)]

    # 20 identities x first two images (dataset is grouped 5 images per identity)
    data = build_dataset()
    firsts, seconds = [], []
    for k in range(20):
        firsts.append(load_image_tensor(data[5 * k]['path']))
        seconds.append(load_image_tensor(data[5 * k + 1]['path']))
    pairs_same = list(zip(firsts, seconds))
    pairs_diff = [(firsts[k], firsts[k + 1]) for k in range(19)]

    rows = []
    with torch.no_grad():
        for name, bgr, mode in CONFIGS:
            fn = lambda t, bgr=bgr, mode=mode: get_vggface_real_embedding(
                model, t, bgr=bgr, normalize_mode=mode)
            rows.append((f"VGG-Face {name}", *score(fn, triplet, pairs_same, pairs_diff)))
        fn = lambda t: get_vggface_pytorch_embedding(imagenet_vgg, t)
        rows.append(("ImageNet VGG-16 (current)", *score(fn, triplet, pairs_same, pairs_diff)))

    print(f"\n{'':<28}{'--- classic triplet ---':^30}{'--- 20-identity average ---':^32}")
    print(f"{'Config':<28}{'same':>9}{'diff':>9}{'gap':>9}   {'same':>9}{'diff':>9}{'gap':>9}")
    print("-" * 90)
    for name, s_same, s_diff, m_same, m_diff in rows:
        print(f"{name:<28}{s_same:>9.4f}{s_diff:>9.4f}{s_same - s_diff:>9.4f}   "
              f"{m_same:>9.4f}{m_diff:>9.4f}{m_same - m_diff:>9.4f}")
    print("\nPick the VGG-Face config with the largest 20-identity gap.")
