"""
Real VGG-Face (Parkhi et al., 2015) as a native, differentiable PyTorch proxy.

Why this file exists
--------------------
models.py's load_vggface_pytorch() loads torchvision VGG-16 with
ImageNet weights (an object classifier, not a face recogniser). This file
provides the ACTUAL VGG-Face face-recognition network -- the same weights
DeepFace uses for its 'VGG-Face' model -- so the ensemble really is
FaceNet + VGG-Face + ArcFace as described in the proposal.

Weights: DeepFace's official PyTorch export of VGG-Face
  https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.pth
(numerically checked against DeepFace's original Keras .h5 weights).
Architecture: same as deepface/models/facial_recognition/pytorch/VGGFace.py
(DeepFace, MIT licence). Embedding = 4096-d fc7 output, then L2-normalised,
exactly as DeepFace's VGG-Face represent().

Input to get_vggface_real_embedding(): [B,3,H,W] RGB in [0,1] (the output
of load_image_tensor). It is resized to 224x224 and converted to the
model's expected format. Gradients flow through (needed for PGD).

This file does NOT modify models.py. It is used by evaluate_full.py when
run with --vggface real (the default).
"""
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VGGFACE_URLS = [   # tried in order; same file on both hosts
    'https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.pth',
    'https://huggingface.co/serengil/deepface/resolve/main/vgg_face_weights.pth',
]
VGGFACE_SIZE = 580022362   # bytes; used to detect incomplete downloads
_PROJECT_WEIGHTS = os.path.join('data', 'vggface', 'vgg_face_weights.pth')
_DEEPFACE_WEIGHTS = os.path.join(os.path.expanduser('~'), '.deepface',
                                 'weights', 'vgg_face_weights.pth')

# Mean pixel of the VGG-Face training set, in BGR order (0-255 scale)
_VGG_MEAN_BGR = (93.5940, 104.7624, 129.1863)


class VGGFaceNet(nn.Module):
    """VGG-Face (VGG-16 layout, fc layers as convolutions). Input is BGR."""

    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.fc8 = nn.Conv2d(4096, 2622, 1)   # identity classifier; loaded, not used
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """x: [B,3,224,224] BGR, already normalised. Returns [B,4096] fc7 features."""
        r = F.relu
        x = self.pool(r(self.conv1_2(r(self.conv1_1(x)))))
        x = self.pool(r(self.conv2_2(r(self.conv2_1(x)))))
        x = self.pool(r(self.conv3_3(r(self.conv3_2(r(self.conv3_1(x)))))))
        x = self.pool(r(self.conv4_3(r(self.conv4_2(r(self.conv4_1(x)))))))
        x = self.pool(r(self.conv5_3(r(self.conv5_2(r(self.conv5_1(x)))))))
        x = r(self.fc6(x))
        x = r(self.fc7(x))          # dropout layers are inactive in eval mode
        return torch.flatten(x, 1)


def _resume_download(url, tmp):
    """Download url into tmp, continuing from whatever tmp already holds."""
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url, headers={'Range': f'bytes={have}-'})
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:      # server ignored Range -> start over
            have = 0
        with open(tmp, 'ab' if have else 'wb') as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                have += len(chunk)
                print(f"\r  {have/1e6:.0f}/{VGGFACE_SIZE/1e6:.0f} MB", end="")
    print()


def _find_or_download_weights(weights_path=None):
    if weights_path:
        return weights_path
    for p in (_PROJECT_WEIGHTS, _DEEPFACE_WEIGHTS):
        if os.path.exists(p) and os.path.getsize(p) == VGGFACE_SIZE:
            return p
    os.makedirs(os.path.dirname(_PROJECT_WEIGHTS), exist_ok=True)
    tmp = _PROJECT_WEIGHTS + '.part'
    print(f"[VGG-Face] Downloading weights (~580 MB) to {_PROJECT_WEIGHTS} "
          f"(resumes automatically if interrupted) ...")
    for attempt in range(1, 9):
        url = VGGFACE_URLS[(attempt - 1) // 4]   # 4 tries on GitHub, then 4 on HF
        try:
            _resume_download(url, tmp)
        except Exception as e:
            print(f"\n  attempt {attempt} interrupted ({type(e).__name__}); resuming...")
        if os.path.exists(tmp) and os.path.getsize(tmp) == VGGFACE_SIZE:
            os.replace(tmp, _PROJECT_WEIGHTS)
            print("[VGG-Face] Download complete.")
            return _PROJECT_WEIGHTS
    raise RuntimeError(
        "VGG-Face weights could not be downloaded. Download this file in your "
        f"browser:\n  {VGGFACE_URLS[0]}\nand save it as {_PROJECT_WEIGHTS}")


def load_vggface_real(weights_path=None):
    """Load real VGG-Face in eval mode on `device`, parameters frozen."""
    path = _find_or_download_weights(weights_path)
    print(f"[VGG-Face] Loading real VGG-Face weights from {path} ...")
    model = VGGFaceNet()
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)   # fails loudly on any mismatch
    model = model.eval().to(device)
    model.requires_grad_(False)   # gradients still flow to the INPUT image
    print("[VGG-Face] Loaded (all keys matched).")
    return model


def get_vggface_real_embedding(model, x, bgr=True, normalize_mode='[0,1]'):
    """
    x: [B,3,H,W] RGB in [0,1]. Returns [B,4096] L2-normalised, differentiable.

    normalize_mode:
      '[0,1]'   -> leave in [0,1]                (DeepFace's default 'base')
      'raw255'  -> x*255                          (DeepFace 'raw')
      'vggmean' -> x*255 - VGG-Face mean pixel    (original VGG-Face / DeepFace 'VGGFace')
    Defaults (BGR, [0,1]) match DeepFace's default pipeline; confirm with
    test_vggface_real_preprocessing.py before relying on results.
    """
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    if bgr:
        x = x[:, [2, 1, 0], :, :]
    if normalize_mode == 'raw255':
        x = x * 255
    elif normalize_mode == 'vggmean':
        order = _VGG_MEAN_BGR if bgr else _VGG_MEAN_BGR[::-1]
        mean = torch.tensor(order, device=x.device).view(1, 3, 1, 1)
        x = x * 255 - mean
    emb = model(x)
    return emb / emb.norm(dim=-1, keepdim=True)