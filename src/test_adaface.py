"""
Standalone test: load AdaFace (CVLFace IR101-WebFace4M) from HuggingFace
and run a forward pass to confirm it loads and check the output shape.

This model is used ONLY as a held-out EVALUATION TARGET for O3 (measuring
PR of perturbations generated against FaceNet/VGG/ArcFace proxies).
No gradients needed -- pure forward pass.

Run:  python src\\test_adaface.py
"""
import os
import sys
import torch
import numpy as np
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

REPO_ID = 'minchul/cvlface_adaface_ir101_webface4m'
SAVE_PATH = os.path.join(os.path.expanduser('~'), '.cvlface_models', 'adaface_ir101_webface4m')


def download(repo_id, path, hf_token=None):
    from huggingface_hub import hf_hub_download
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=hf_token,
                         local_dir=path, local_dir_use_symlinks=False)
    with open(files_path, 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            print(f"  Downloading {file} ...")
            hf_hub_download(repo_id, file, token=hf_token,
                             local_dir=path, local_dir_use_symlinks=False)


def load_model_from_local_path(path, hf_token=None):
    from transformers import AutoModel
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    try:
        model = AutoModel.from_pretrained(path, trust_remote_code=True, token=hf_token)
    finally:
        os.chdir(cwd)
        sys.path.pop(0)
    return model


def load_adaface():
    print(f"[AdaFace] Downloading/loading {REPO_ID} ...")
    download(REPO_ID, SAVE_PATH)
    model = load_model_from_local_path(SAVE_PATH)
    model = model.eval().to(device)
    print("[AdaFace] Loaded.")
    return model


if __name__ == '__main__':
    print("\n[1/3] Checking dependencies...")
    try:
        import transformers
        import huggingface_hub
        print(f"  transformers: {transformers.__version__}")
        print(f"  huggingface_hub: {huggingface_hub.__version__}")
    except ImportError as e:
        print(f"  MISSING: {e}")
        print("  Run: pip install transformers huggingface_hub")
        sys.exit(1)

    print("\n[2/3] Loading AdaFace model...")
    try:
        model = load_adaface()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model loaded. Parameters: {n_params/1e6:.1f}M")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n[3/3] Testing forward pass...")
    # Try a few common input conventions and report what works
    test_configs = [
        ("112x112, RGB, [-1,1]",  112, lambda x: x * 2 - 1, False),
        ("112x112, BGR, [-1,1]",  112, lambda x: x * 2 - 1, True),
        ("112x112, RGB, [0,1]",   112, lambda x: x,          False),
    ]

    for name, size, norm_fn, bgr in test_configs:
        try:
            dummy = torch.rand(1, 3, size, size).to(device)
            if bgr:
                dummy = dummy[:, [2, 1, 0], :, :]  # flip channels
            x = norm_fn(dummy)
            with torch.no_grad():
                out = model(x)
            # output might be a tensor or tuple/namedtuple
            if isinstance(out, (tuple, list)):
                out_main = out[0]
            elif hasattr(out, 'last_hidden_state'):
                out_main = out.last_hidden_state
            else:
                out_main = out
            print(f"  [{name}] OK -> output type: {type(out_main)}, "
                  f"shape: {getattr(out_main, 'shape', 'N/A')}")
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")

    print("\nDone. Check which config above succeeded and the embedding shape.")
    print("Expected: shape (1, 512) for a standard face embedding.")