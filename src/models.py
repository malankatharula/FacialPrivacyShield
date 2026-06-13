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