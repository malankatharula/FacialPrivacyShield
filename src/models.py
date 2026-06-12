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

# ── ArcFace ────────────────────────────────────────────────────────────────
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

# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    test_img = r'C:\projects\FacialPrivacyShield\data\lfw-dataset\lfw-deepfunneled\lfw-deepfunneled\Richard_Myers\Richard_Myers_0004.jpg'

    print("\n[1/3] Loading FaceNet...")
    facenet = load_facenet()
    emb_fn = get_facenet_embedding(facenet, test_img)
    print(f"FaceNet embedding shape: {emb_fn.shape}, norm: {emb_fn.norm():.4f}")

    print("\n[2/3] Testing DeepFace VGG-Face...")
    emb_df = get_deepface_embedding(test_img)
    print(f"DeepFace embedding shape: {emb_df.shape}, norm: {emb_df.norm():.4f}")

    print("\n[3/3] Testing ArcFace...")
    arcface = load_arcface()
    emb_arc = get_arcface_embedding(arcface, test_img)
    print(f"ArcFace embedding shape: {emb_arc.shape}, norm: {emb_arc.norm():.4f}")

    print("\nAll 3 proxy models OK.")