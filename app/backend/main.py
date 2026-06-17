"""
FacialPrivacyShield — FastAPI Backend
O4: RESTful API endpoint for adversarial face image protection.

Endpoints:
  POST /protect   — upload a face image, receive adversarially protected version
  GET  /health    — health check
  GET  /          — serves frontend (when running via Docker)
"""
import io
import os
import sys
import time
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Add project src to path (models/poison live there) ────────────────────
# Resolve src/ relative to this file: app/backend/main.py -> ../../src
# Works regardless of which directory uvicorn is launched from.
_HERE = Path(__file__).resolve().parent          # app/backend/
_PROJECT_ROOT = _HERE.parent.parent              # FacialPrivacyShield/
SRC_DIR = _PROJECT_ROOT / "src"
if not SRC_DIR.exists():
    # Fallback: try one level up (in case of flat layout)
    SRC_DIR = _HERE.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))          # also add project root

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FacialPrivacyShield",
    description="Adversarial image poisoning for facial privacy protection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading (done once at startup) ──────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

_models_loaded = False
_facenet = None
_vggface = None
_arcface_pt = None


def load_models():
    global _models_loaded, _facenet, _vggface, _arcface_pt
    if _models_loaded:
        return
    logger.info("Loading proxy models...")
    from models import (
        load_facenet,
        load_vggface_pytorch,
        load_arcface_pytorch,
    )
    _facenet = load_facenet()
    _vggface = load_vggface_pytorch()
    _arcface_pt = load_arcface_pytorch()
    _models_loaded = True
    logger.info("All models loaded.")


@app.on_event("startup")
async def startup_event():
    load_models()


# ── Proxy functions ────────────────────────────────────────────────────────
def get_proxy_models():
    from models import get_vggface_pytorch_embedding, get_arcface_pytorch_embedding

    def facenet_fn(x):
        return _facenet(x * 2 - 1)

    def vggface_fn(x):
        return get_vggface_pytorch_embedding(_vggface, x)

    def arcface_fn(x):
        return get_arcface_pytorch_embedding(_arcface_pt, x)

    return [
        (facenet_fn, 1 / 3),
        (vggface_fn, 1 / 3),
        (arcface_fn, 1 / 3),
    ]


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(device),
        "models_loaded": _models_loaded,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/protect")
async def protect_image(file: UploadFile = File(...)):
    """
    Upload a face image (JPEG/PNG), receive an adversarially protected version.

    The protection uses PGD-based ensemble proxy optimization over FaceNet,
    VGG-Face, and ArcFace — making the image resistant to unauthorized facial
    recognition training while remaining visually imperceptible to humans.
    """
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    # Validate file size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    if not _models_loaded:
        raise HTTPException(status_code=503, detail="Models still loading. Try again in a moment.")

    try:
        # Load image
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

        from poison import load_image_tensor, pgd_ensemble_poison
        from torchvision import transforms
        import torch.nn.functional as F

        # Keep original dimensions for output
        orig_w, orig_h = pil_img.size

        # Save to temp file for load_image_tensor
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            pil_img.save(tmp.name)
            tmp_path = tmp.name

        img_tensor = load_image_tensor(tmp_path)  # resized to 160x160 for PGD
        os.unlink(tmp_path)

        # Run ensemble PGD protection
        logger.info(f"Running ensemble PGD on uploaded image...")
        t0 = time.time()

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        proxy_models = get_proxy_models()
        img_protected_160 = pgd_ensemble_poison(img_tensor, proxy_models)

        elapsed = time.time() - t0
        logger.info(f"Protection complete in {elapsed:.1f}s")

        # Compute the perturbation at 160x160
        delta = img_protected_160 - img_tensor  # perturbation in [−eps, eps]

        # Upscale perturbation back to original resolution and apply to original image
        orig_tensor = transforms.ToTensor()(pil_img).unsqueeze(0).to(device)
        delta_upscaled = F.interpolate(
            delta, size=(orig_h, orig_w), mode='bilinear', align_corners=False
        )
        img_protected_full = (orig_tensor + delta_upscaled).clamp(0, 1)

        # Convert to PIL at original resolution
        protected_pil = transforms.ToPILImage()(
            img_protected_full.squeeze(0).cpu()
        )

        # Return as JPEG
        output_buffer = io.BytesIO()
        protected_pil.save(output_buffer, format="JPEG", quality=95)
        output_buffer.seek(0)

        return StreamingResponse(
            output_buffer,
            media_type="image/jpeg",
            headers={
                "X-Processing-Time": f"{elapsed:.1f}s",
                "X-Device": str(device),
                "Content-Disposition": "attachment; filename=protected.jpg",
            },
        )

    except Exception as e:
        logger.error(f"Protection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Protection failed: {str(e)}")


# ── Serve React frontend (in Docker, frontend is built to /app/static) ─────
STATIC_DIR = Path(__file__).parent.parent / "frontend" / "dist"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")