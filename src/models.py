import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── FaceNet ────────────────────────────────────────────────────────────────
def load_facenet():
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return model

def get_facenet_embedding(model, img_path):
    mtcnn = MTCNN(image_size=160, device=device)
    img = Image.open(img_path).convert('RGB')
    face = mtcnn(img)
    if face is None:
        face = torch.tensor(
            np.array(img.resize((160,160)), dtype=np.float32) / 127.5 - 1
        ).permute(2,0,1).unsqueeze(0).to(device)
    else:
        face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(face)
    return emb

# ── DeepFace / VGG-Face ───────────────────────────────────────────────────
def get_deepface_embedding(img_path):
    from deepface import DeepFace
    result = DeepFace.represent(
        img_path=img_path,
        model_name='VGG-Face',
        enforce_detection=False
    )
    emb = np.array(result[0]['embedding'], dtype=np.float32)
    return torch.tensor(emb).to(device)

# ── VGG-Face (PyTorch native — differentiable) ─────────────────────────────
def load_vggface_pytorch():
    import torchvision.models as tvm
    model = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])
    model = model.eval().to(device)
    return model

def get_vggface_pytorch_embedding(model, x):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
    x_resized = torch.nn.functional.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
    x_norm = (x_resized - mean) / std
    emb = model(x_norm)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb

# ── ArcFace ONNX (insightface) — used for PR evaluation ONLY ──────────────
# NOTE: Sits OUTSIDE the PyTorch gradient graph. For PR evaluation only.
# Do NOT use as a proxy inside pgd_ensemble_poison.
def load_arcface():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(160, 160))
    return app

def get_arcface_embedding(app, img_path):
    import cv2
    img = cv2.imread(img_path)
    img = cv2.resize(img, (160, 160))
    faces = app.get(img)
    if len(faces) == 0:
        return torch.zeros(512).to(device)
    emb = torch.tensor(faces[0].embedding, dtype=torch.float32).to(device)
    emb = emb / emb.norm()
    return emb

# ── ArcFace PyTorch Native (facexlib) — DIFFERENTIABLE proxy ──────────────
# Uses facexlib's authoritative ArcFace (IR-SE-50) architecture with
# pretrained weights auto-downloaded from a reliable mirror.
# This is a fully native nn.Module — gradients flow cleanly during PGD.
#
# This is an INDEPENDENT ArcFace checkpoint (ir_se50), distinct from the
# buffalo_l ONNX used for evaluation. That separation is intentional and
# good for the transferability story: the proxy and the eval target are
# not the same exact weights.

class ArcFaceProxy(nn.Module):
    """
    facexlib ArcFace (Backbone IR-SE-50) wrapped with input preprocessing.
    Input:  [B, 3, H, W] float in [0, 1]
    Output: [B, 512] L2-normalised embedding (fully differentiable)
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone   # facexlib Backbone, outputs l2-normed [B,512]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # facexlib ArcFace expects 112x112, RGB, normalized to [-1, 1]
        x = torch.nn.functional.interpolate(
            x, size=(112, 112), mode='bilinear', align_corners=False
        )
        x = x * 2.0 - 1.0           # [0,1] → [-1,1]
        emb = self.backbone(x)       # already l2-normalised by Backbone.forward
        return emb


def load_arcface_pytorch() -> nn.Module:
    """
    Load facexlib's differentiable ArcFace (IR-SE-50) as a PGD proxy.
    Weights auto-download to facexlib/weights on first run.

    Returns: ArcFaceProxy in eval mode on device, or None on failure.
    """
    try:
        from facexlib.recognition import init_recognition_model
    except ImportError:
        print("[ArcFace PyTorch] facexlib not installed.")
        print("  Run: pip install facexlib")
        print("[ArcFace PyTorch] Falling back to ONNX proxy (zero gradients).")
        return None

    try:
        print("[ArcFace PyTorch] Loading facexlib ArcFace IR-SE-50 "
              "(auto-downloads weights on first run)...")
        # init_recognition_model hardcodes .to('cuda'); pass device through
        backbone = init_recognition_model('arcface', device=str(device))
        backbone = backbone.eval()

        wrapper = ArcFaceProxy(backbone).eval().to(device)

        # Forward sanity check
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 160, 160).to(device)
            out = wrapper(dummy)
        assert out.shape == (1, 512), f"Bad output shape: {out.shape}"
        print(f"[ArcFace PyTorch] Forward pass OK. Output: {out.shape}")

        # Gradient flow check — this MUST pass for PGD to work.
        # IMPORTANT: create the tensor directly on `device` with requires_grad.
        # Doing torch.randn(..., requires_grad=True).to(device) would make the
        # tensor a NON-LEAF node (the .to() is recorded in the graph), so its
        # .grad stays None even when gradients flow correctly. That bug would
        # produce a false "0.0000" reading.
        test_in = torch.randn(1, 3, 160, 160, device=device, requires_grad=True)
        test_out = wrapper(test_in)
        test_out.sum().backward()
        grad_sum = test_in.grad.abs().sum().item() if test_in.grad is not None else 0.0
        has_grad = grad_sum > 1e-8
        print(f"[ArcFace PyTorch] Gradient flow check: "
              f"{'PASS ✓' if has_grad else 'FAIL ✗'} (grad magnitude: {grad_sum:.4f})")
        if not has_grad:
            print("[ArcFace PyTorch] Falling back to ONNX proxy.")
            return None

        return wrapper

    except Exception as e:
        import traceback
        print(f"[ArcFace PyTorch] Failed: {e}")
        traceback.print_exc()
        print("[ArcFace PyTorch] Falling back to ONNX proxy.")
        return None


def get_arcface_pytorch_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """x: [1,3,H,W] in [0,1] → [1,512] normalised, gradients intact."""
    return model(x)


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    test_img = r'C:\projects\FacialPrivacyShield\data\lfw-dataset\lfw-deepfunneled\lfw-deepfunneled\Richard_Myers\Richard_Myers_0004.jpg'

    print("\n[1/4] Loading FaceNet...")
    facenet = load_facenet()
    emb_fn = get_facenet_embedding(facenet, test_img)
    print(f"FaceNet embedding shape: {emb_fn.shape}, norm: {emb_fn.norm():.4f}")

    print("\n[2/4] Testing DeepFace VGG-Face...")
    emb_df = get_deepface_embedding(test_img)
    print(f"DeepFace embedding shape: {emb_df.shape}, norm: {emb_df.norm():.4f}")

    print("\n[3/4] Testing ArcFace ONNX (evaluation only)...")
    arcface_onnx = load_arcface()
    emb_arc_onnx = get_arcface_embedding(arcface_onnx, test_img)
    print(f"ArcFace ONNX embedding shape: {emb_arc_onnx.shape}, norm: {emb_arc_onnx.norm():.4f}")

    print("\n[4/4] Testing ArcFace PyTorch facexlib (differentiable proxy)...")
    arcface_pt = load_arcface_pytorch()
    if arcface_pt is not None:
        from poison import load_image_tensor
        img_t = load_image_tensor(test_img)
        emb_arc_pt = get_arcface_pytorch_embedding(arcface_pt, img_t)
        print(f"ArcFace PyTorch embedding shape: {emb_arc_pt.shape}, "
              f"norm: {emb_arc_pt.norm():.4f}")
        print("  ✓ Differentiable ArcFace proxy ready for PGD ensemble.")
        print("  (Note: this is ir_se50, independent from buffalo_l ONNX eval target)")
    else:
        print("ArcFace PyTorch load failed — check output above.")

    print("\nAll models tested.")


# ── AdaFace (CVLFace IR101-WebFace4M) — HELD-OUT EVALUATION TARGET ─────────
# Used ONLY to measure transferability (O3). NOT used as a PGD proxy.
# Forward-pass only (no gradients needed).
#
# Source: minchul/cvlface_adaface_ir101_webface4m on HuggingFace Hub.
# Requires: transformers==4.46.3 (NOT 5.x — needs torch<2.4 compat),
#           huggingface_hub<1.0, omegaconf, fvcore.
#
# Preprocessing (empirically verified): RGB, [0,1] range, 112x112 input.
# This matches load_image_tensor()'s output directly -- only a resize needed.

_ADAFACE_REPO = 'minchul/cvlface_adaface_ir101_webface4m'

import os as _os
_ADAFACE_SAVE_PATH = _os.path.join(
    _os.path.expanduser('~'), '.cvlface_models', 'adaface_ir101_webface4m'
)


def _download_adaface(repo_id, path, hf_token=None):
    """Download AdaFace model files from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download
    _os.makedirs(path, exist_ok=True)
    files_path = _os.path.join(path, 'files.txt')
    if not _os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=hf_token,
                         local_dir=path, local_dir_use_symlinks=False)
    with open(files_path, 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = _os.path.join(path, file)
        if not _os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=hf_token,
                             local_dir=path, local_dir_use_symlinks=False)


def load_adaface(repo_id: str = _ADAFACE_REPO, save_path: str = None):
    """
    Load AdaFace (IR101-WebFace4M) as a held-out evaluation target.

    Returns: nn.Module in eval mode on device. Output: [B, 512] embedding
    (NOT pre-normalised -- caller should L2-normalise if needed for cosine
    distance, consistent with other models in this file).

    Input format: x of shape [B, 3, H, W] in [0, 1], RGB.
    Use get_adaface_embedding(model, x) which handles the resize to 112x112.
    """
    import sys
    # Avoid hanging on HF's xet transfer backend (known to stall on some networks)
    _os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

    from transformers import AutoModel

    if save_path is None:
        save_path = _ADAFACE_SAVE_PATH

    print(f"[AdaFace] Loading {repo_id} (held-out eval target)...")
    _download_adaface(repo_id, save_path)

    cwd = _os.getcwd()
    _os.chdir(save_path)
    sys.path.insert(0, save_path)

    # CVLFace's wrapper.py does `from models import get_model`, expecting its
    # OWN models/ package (downloaded into save_path). But this very file is
    # named src/models.py and is already cached in sys.modules['models'] (and
    # possibly models.iresnet etc. from a prior call). Without eviction,
    # Python's import cache would hand wrapper.py THIS module instead of
    # CVLFace's, causing "cannot import name 'get_model' from 'models'".
    # Temporarily evict our 'models'/'models.*' entries so the dynamic import
    # re-resolves via the CVLFace path inserted above, then restore them.
    saved_modules = {k: v for k, v in sys.modules.items()
                     if k == 'models' or k.startswith('models.')}
    for k in saved_modules:
        del sys.modules[k]

    try:
        model = AutoModel.from_pretrained(save_path, trust_remote_code=True)
    finally:
        _os.chdir(cwd)
        sys.path.pop(0)
        # Restore our models module(s) so subsequent `from models import ...`
        # elsewhere in this process still resolves to src/models.py.
        for k in list(sys.modules.keys()):
            if k == 'models' or k.startswith('models.'):
                del sys.modules[k]
        for k, v in saved_modules.items():
            sys.modules[k] = v

    model = model.eval().to(device)
    print("[AdaFace] Loaded.")
    return model


def get_adaface_embedding(model, x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, 3, H, W] in [0, 1], RGB (matches load_image_tensor output).
    Returns: [B, 512] embedding (forward pass only, no_grad).
    """
    x_resized = torch.nn.functional.interpolate(
        x, size=(112, 112), mode='bilinear', align_corners=False
    )
    with torch.no_grad():
        emb = model(x_resized)
    return emb


# ── ElasticFace (IResNet-100, ElasticFace-Cos+) — HELD-OUT EVAL TARGET ─────
# O3 cross-architecture transferability target #2.
# Used ONLY for evaluation (forward pass, no gradients).
#
# Architecture: standard InsightFace iresnet100 (src/iresnet.py) -- the SAME
# file ElasticFace's training code imports, so state_dict keys should match
# directly via load_state_dict() with strict=True.
#
# Weights: 295672backbone.pth from the official ElasticFace repo
# (fdbtrs/ElasticFace, ElasticFace-Cos+ variant), CC BY-NC-SA 4.0.
# Google Drive folder (download manually, see instructions):
#   https://drive.google.com/drive/folders/19LXrjVNt60JBZP7JqsvOSWMwGLGrcJl5
#
# Place the downloaded file at: data/elasticface/295672backbone.pth
#
# Preprocessing: BGR (per ElasticFace/CR-FIQA evaluation scripts).
# Normalisation TBD -- verify empirically like AdaFace (RGB[0,1] won there;
# do NOT assume BGR[-1,1] is correct without checking).

_ELASTICFACE_WEIGHTS_PATH = _os.path.join(
    'data', 'elasticface', '295672backbone.pth'
)


def load_elasticface(weights_path: str = None):
    """
    Load ElasticFace-Cos+ (IResNet-100) as a held-out evaluation target.

    Requires 295672backbone.pth downloaded manually from the Google Drive
    folder linked above and placed at data/elasticface/295672backbone.pth
    (relative to project root) or pass an explicit weights_path.

    Returns: nn.Module in eval mode on device, or None if weights not found.
    Output: [B, 512] embedding (not L2-normalised).
    """
    if weights_path is None:
        weights_path = _ELASTICFACE_WEIGHTS_PATH

    if not _os.path.exists(weights_path):
        print(f"[ElasticFace] Weights not found at {weights_path}")
        print("[ElasticFace] Download 295672backbone.pth from:")
        print("  https://drive.google.com/drive/folders/19LXrjVNt60JBZP7JqsvOSWMwGLGrcJl5")
        print(f"  and place it at: {weights_path}")
        return None

    from iresnet import iresnet100

    print(f"[ElasticFace] Loading IResNet-100 from {weights_path} ...")
    model = iresnet100()
    state_dict = torch.load(weights_path, map_location=device)

    # Some checkpoints wrap the state_dict in a 'state_dict' or 'model' key
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']

    # Strip potential 'module.' prefix from DataParallel-trained checkpoints
    state_dict = { (k[7:] if k.startswith('module.') else k): v
                    for k, v in state_dict.items() }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    n_total = len(model.state_dict())
    n_loaded = n_total - len(missing)
    print(f"[ElasticFace] Loaded {n_loaded}/{n_total} keys "
          f"({len(missing)} missing, {len(unexpected)} unexpected)")
    if missing:
        print(f"  Missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"  Unexpected (first 5): {unexpected[:5]}")

    model = model.eval().to(device)
    print("[ElasticFace] Loaded.")
    return model


def get_elasticface_embedding(model, x: torch.Tensor, bgr: bool = False,
                               normalize_range: str = '[0,1]') -> torch.Tensor:
    """
    x: [B, 3, H, W] in [0, 1], RGB (matches load_image_tensor output).
    bgr: if True, flip RGB->BGR before feeding the model.
    normalize_range: '[-1,1]' applies x*2-1; '[0,1]' leaves as-is.

    Defaults (bgr=False, normalize_range='[0,1]') are the EMPIRICALLY
    VERIFIED best config (gap=0.6753 vs BGR[-1,1]'s 0.5406 in
    test_elasticface_preprocessing.py) -- despite ElasticFace's official
    docs suggesting BGR. Same convention as AdaFace.

    Returns: [B, 512] embedding (forward pass only, no_grad).
    """
    x_resized = torch.nn.functional.interpolate(
        x, size=(112, 112), mode='bilinear', align_corners=False
    )
    if bgr:
        x_resized = x_resized[:, [2, 1, 0], :, :]
    if normalize_range == '[-1,1]':
        x_resized = x_resized * 2 - 1
    with torch.no_grad():
        emb = model(x_resized)
    return emb