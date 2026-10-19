# Copy of the official LowKey script attack_dir_warp.py (Cherepanova et al., ICLR 2021).
# ATTACK SETTINGS AND ALGORITHM ARE UNCHANGED. Only practical additions:
#   1) outputs saved to a separate --out folder
#   2) resume: skips images already done
#   3) images with no detected face are logged and skipped (instead of crashing)
#   4) per-image timing saved to lowkey_log.csv
#   5) images converted to RGB on load
import torch, time, csv
from PIL import Image
import numpy as np
from util.feature_extraction_utils import feature_extractor, normalize_transforms, warp_image, normalize_batch
from backbone.model_irse import IR_50, IR_101, IR_152
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from util.attack_utils import Attack
from util.prepare_utils import prepare_models, prepare_dir_vec, get_ensemble, prepare_data
from align.detector import detect_faces
from align.align_trans import get_reference_facial_points, warp_and_crop_face
import argparse
import torchvision.transforms as transforms
import sys, os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_tensor = transforms.ToTensor()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='folder with original images')
    parser.add_argument('--out', required=True, help='folder to save protected images')
    parser.add_argument('--limit', type=int, default=0, help='only process first N images (0 = all)')
    args = parser.parse_args()
    dir_root, out_root = args.dir, args.out
    os.makedirs(out_root, exist_ok=True)
    print(f"Device: {device}", flush=True)

    # ---- official LowKey settings (unchanged) ----
    eps = 0.05
    n_iters = 50
    input_size = [112, 112]
    attack_type = 'lpips'
    c_tv = None
    c_sim = 0.05
    lr = 0.0025
    net_type = 'alex'
    noise_size = 0.005
    n_starts = 1
    kernel_size_gf = 7
    sigma_gf = 3
    combination = True
    using_subspace = False
    V_reduction_root = './'
    model_backbones = ['IR_152', 'IR_152', 'ResNet_152', 'ResNet_152']
    model_roots = ['models/Backbone_IR_152_Arcface_Epoch_112.pth', 'models/Backbone_IR_152_Cosface_Epoch_70.pth',
                   'models/Backbone_ResNet_152_Arcface_Epoch_65.pth', 'models/Backbone_ResNet_152_Cosface_Epoch_68.pth']
    direction = 1
    crop_size = 112
    scale = crop_size / 112.

    print("Loading the four LowKey models...", flush=True)
    models_attack, V_reduction, dim = prepare_models(model_backbones, input_size, model_roots,
                                                     kernel_size_gf, sigma_gf, combination,
                                                     using_subspace, V_reduction_root)
    print("Models loaded.", flush=True)

    names = sorted(f for f in os.listdir(dir_root) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    if args.limit:
        names = names[:args.limit]

    log_path = os.path.join(out_root, 'lowkey_log.csv')
    new_log = not os.path.exists(log_path)
    logf = open(log_path, 'a', newline='')
    writer = csv.writer(logf)
    if new_log:
        writer.writerow(['image', 'status', 'seconds'])

    for i, img_name in enumerate(names):
        out_path = os.path.join(out_root, os.path.splitext(img_name)[0] + '_attacked.png')
        if os.path.exists(out_path):
            continue
        t0 = time.time()
        try:
            img = Image.open(os.path.join(dir_root, img_name)).convert('RGB')
            reference = get_reference_facial_points(default_square=True) * scale
            h, w, c = np.array(img).shape

            _, landmarks = detect_faces(img)
            if len(landmarks) == 0:
                writer.writerow([img_name, 'no_face', 0]); logf.flush()
                print(f'[{i+1}/{len(names)}] {img_name}: NO FACE DETECTED, skipped', flush=True)
                continue
            facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
            _, tfm = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
            theta = normalize_transforms(tfm, w, h)
            tensor_img = to_tensor(img).unsqueeze(0).to(device)

            V_reduction = None
            dim = 512
            dir_vec_extractor = get_ensemble(models=models_attack, sigma_gf=None, kernel_size_gf=None,
                                             combination=False, V_reduction=V_reduction, warp=True, theta_warp=theta)
            dir_vec = prepare_dir_vec(dir_vec_extractor, tensor_img, dim, combination)

            attack = Attack(models_attack, dim, attack_type, eps, c_sim, net_type, lr,
                            n_iters, noise_size, n_starts, c_tv, sigma_gf, kernel_size_gf,
                            combination, warp=True, theta_warp=theta, V_reduction=V_reduction)
            img_attacked = attack.execute(tensor_img, dir_vec, direction).detach().cpu()

            transforms.ToPILImage()(img_attacked[0]).save(out_path)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            secs = time.time() - t0
            writer.writerow([img_name, 'ok', f'{secs:.2f}'])
            print(f'[{i+1}/{len(names)}] {img_name}: done in {secs:.1f}s', flush=True)
        except Exception as e:
            writer.writerow([img_name, f'error: {e}', 0])
            print(f'[{i+1}/{len(names)}] {img_name}: ERROR {e}', flush=True)
        logf.flush()

    logf.close()
    print('Finished.', flush=True)
