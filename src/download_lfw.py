import os
import urllib.request
import tarfile

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
LFW_TGZ = os.path.join(DATA_DIR, "lfw.tgz")
LFW_DIR = os.path.join(DATA_DIR, "lfw")

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(LFW_TGZ):
    print("Downloading LFW dataset (~180MB)...")
    urllib.request.urlretrieve(LFW_URL, LFW_TGZ, 
        reporthook=lambda b, bs, t: print(f"\r{b*bs/1e6:.1f}/{t/1e6:.1f} MB", end=""))
    print("\nDownload complete.")
else:
    print("LFW archive already exists, skipping download.")

if not os.path.exists(LFW_DIR):
    print("Extracting...")
    with tarfile.open(LFW_TGZ, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    print("Extraction complete.")
else:
    print("LFW already extracted.")

# Count identities and images
identities = os.listdir(LFW_DIR)
total_images = sum(len(os.listdir(os.path.join(LFW_DIR, p))) for p in identities)
print(f"LFW: {len(identities)} identities, {total_images} images total.")