import torch
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
        # fallback: resize without detection
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

# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    test_img = r'C:\projects\FacialPrivacyShield\data\lfw-dataset\lfw-deepfunneled\lfw-deepfunneled\Richard_Myers\Richard_Myers_0004.jpg'
    
    print("\n[1/2] Loading FaceNet...")
    facenet = load_facenet()
    emb_fn = get_facenet_embedding(facenet, test_img)
    print(f"FaceNet embedding shape: {emb_fn.shape}, norm: {emb_fn.norm():.4f}")
    
    print("\n[2/2] Testing DeepFace VGG-Face...")
    emb_df = get_deepface_embedding(test_img)
    print(f"DeepFace embedding shape: {emb_df.shape}, norm: {emb_df.norm():.4f}")
    
    print("\nAll proxy models OK.")