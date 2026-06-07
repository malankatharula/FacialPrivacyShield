import os
import random
from PIL import Image
import numpy as np

LFW_DIR = r'C:\projects\FacialPrivacyShield\data\lfw-dataset\lfw-deepfunneled\lfw-deepfunneled'
MIN_IMAGES = 5
SAMPLE_SIZE = 423  # all identities with >= 5 images
IMG_SIZE = (160, 160)  # standard for FaceNet
SEED = 42

def get_identities(lfw_dir, min_images=MIN_IMAGES):
    identities = []
    for name in sorted(os.listdir(lfw_dir)):
        path = os.path.join(lfw_dir, name)
        if os.path.isdir(path):
            imgs = [f for f in os.listdir(path) if f.endswith('.jpg')]
            if len(imgs) >= min_images:
                identities.append((name, imgs))
    return identities

def load_image(path, size=IMG_SIZE):
    img = Image.open(path).convert('RGB').resize(size)
    return np.array(img, dtype=np.float32) / 255.0

def build_dataset(lfw_dir=LFW_DIR, n_identities=SAMPLE_SIZE, 
                  n_images=MIN_IMAGES, seed=SEED):
    random.seed(seed)
    identities = get_identities(lfw_dir)
    selected = random.sample(identities, min(n_identities, len(identities)))
    
    dataset = []
    for name, imgs in selected:
        sampled = random.sample(imgs, n_images)
        for img_file in sampled:
            img_path = os.path.join(lfw_dir, name, img_file)
            dataset.append({
                'identity': name,
                'path': img_path,
            })
    
    print(f"Dataset: {len(selected)} identities x {n_images} images = {len(dataset)} total")
    return dataset

if __name__ == '__main__':
    dataset = build_dataset()
    print("Sample entry:", dataset[0])
    print("Dataset ready.")