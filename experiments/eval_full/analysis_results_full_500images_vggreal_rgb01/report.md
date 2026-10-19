# Evaluation analysis

Source: `experiments\eval_full\results_full_500images_vggreal_rgb01.jsonl`  
Images: **500**, identities: **100**, PR threshold: cosine distance > 0.3

## 1. Protection rate

PR % [Wilson 95% CI]; second line: mean ± SD across identities [cluster-bootstrap 95% CI].

| Target | Ours (3-model ensemble) | Single (FaceNet) | LowKey-style | Fawkes-style | Random noise (control) |
|---|---|---|---|---|---|
| facenet<br>*proxy (white-box)* | 100.0 [99.2–100.0]<br>100.0 ± 0.0 [100.0–100.0] | 100.0 [99.2–100.0]<br>100.0 ± 0.0 [100.0–100.0] | 100.0 [99.2–100.0]<br>100.0 ± 0.0 [100.0–100.0] | 100.0 [99.2–100.0]<br>100.0 ± 0.0 [100.0–100.0] | 0.0 [0.0–0.8]<br>0.0 ± 0.0 [0.0–0.0] |
| vggface<br>*proxy (white-box)* | 100.0 [99.2–100.0]<br>100.0 ± 0.0 [100.0–100.0] | 11.2 [8.7–14.3]<br>11.2 ± 18.3 [8.0–15.0] | 17.0 [14.0–20.5]<br>17.0 ± 22.8 [12.8–21.4] | 2.2 [1.2–3.9]<br>2.2 ± 6.3 [1.0–3.4] | 0.0 [0.0–0.8]<br>0.0 ± 0.0 [0.0–0.0] |
| arcface<br>*eval-only (buffalo_l)* | 29.6 [25.8–33.7]<br>29.6 ± 32.3 [23.4–36.8] | 15.8 [12.9–19.3]<br>15.8 ± 26.3 [11.0–21.4] | 17.0 [14.0–20.5]<br>17.0 ± 25.8 [12.2–22.2] | 4.4 [2.9–6.6]<br>4.4 ± 10.9 [2.4–6.6] | 0.2 [0.0–1.1]<br>0.2 ± 2.0 [0.0–0.6] |
| adaface<br>*held-out* | 70.4 [66.3–74.2]<br>70.4 ± 25.5 [65.4–75.4] | 24.0 [20.5–27.9]<br>24.0 ± 23.8 [19.8–28.6] | 52.0 [47.6–56.3]<br>52.0 ± 27.3 [46.8–57.6] | 5.8 [4.1–8.2]<br>5.8 ± 13.1 [3.6–8.8] | 0.0 [0.0–0.8]<br>0.0 ± 0.0 [0.0–0.0] |
| elasticface<br>*held-out* | 73.8 [69.8–77.5]<br>73.8 ± 23.9 [69.2–78.6] | 17.6 [14.5–21.2]<br>17.6 ± 18.5 [14.0–21.2] | 54.8 [50.4–59.1]<br>54.8 ± 27.7 [49.6–60.2] | 2.8 [1.7–4.6]<br>2.8 ± 7.0 [1.6–4.2] | 0.0 [0.0–0.8]<br>0.0 ± 0.0 [0.0–0.0] |
| sphereface<br>*held-out* | 14.0 [11.2–17.3]<br>14.0 ± 16.2 [10.8–17.2] | 4.8 [3.2–7.0]<br>4.8 ± 9.9 [3.0–6.8] | 11.4 [8.9–14.5]<br>11.4 ± 14.6 [8.8–14.4] | 0.4 [0.1–1.4]<br>0.4 ± 2.8 [0.0–1.0] | 0.0 [0.0–0.8]<br>0.0 ± 0.0 [0.0–0.0] |

## 2. Paired comparison (exact McNemar, Holm-corrected)

*ours only* = images protected by ours but not the other method; *other only* = the reverse.

| Target | vs | ours only | other only | ΔPR (pts) | p (Holm) | Verdict |
|---|---|---|---|---|---|---|
| facenet | Single (FaceNet) | 0 | 0 | +0.0 | 1 | no significant difference |
| facenet | LowKey-style | 0 | 0 | +0.0 | 1 | no significant difference |
| facenet | Fawkes-style | 0 | 0 | +0.0 | 1 | no significant difference |
| facenet | Random noise (control) | 500 | 0 | +100.0 | 1.47e-149 | ours better |
| vggface | Single (FaceNet) | 444 | 0 | +88.8 | 9.25e-133 | ours better |
| vggface | LowKey-style | 415 | 0 | +83.0 | 4.73e-124 | ours better |
| vggface | Fawkes-style | 489 | 0 | +97.8 | 2.75e-146 | ours better |
| vggface | Random noise (control) | 500 | 0 | +100.0 | 1.47e-149 | ours better |
| arcface | Single (FaceNet) | 72 | 3 | +13.8 | 2.61e-17 | ours better |
| arcface | LowKey-style | 66 | 3 | +12.6 | 1.11e-15 | ours better |
| arcface | Fawkes-style | 126 | 0 | +25.2 | 2.82e-37 | ours better |
| arcface | Random noise (control) | 147 | 0 | +29.4 | 1.46e-43 | ours better |
| adaface | Single (FaceNet) | 234 | 2 | +46.4 | 7.09e-66 | ours better |
| adaface | LowKey-style | 104 | 12 | +18.4 | 1.68e-18 | ours better |
| adaface | Fawkes-style | 325 | 2 | +64.6 | 6.28e-93 | ours better |
| adaface | Random noise (control) | 352 | 0 | +70.4 | 3.92e-105 | ours better |
| elasticface | Single (FaceNet) | 284 | 3 | +56.2 | 4.75e-79 | ours better |
| elasticface | LowKey-style | 112 | 17 | +19.0 | 1.96e-17 | ours better |
| elasticface | Fawkes-style | 356 | 1 | +71.0 | 4.15e-104 | ours better |
| elasticface | Random noise (control) | 369 | 0 | +73.8 | 3.16e-110 | ours better |
| sphereface | Single (FaceNet) | 47 | 1 | +9.2 | 1.74e-12 | ours better |
| sphereface | LowKey-style | 23 | 10 | +2.6 | 0.14 | no significant difference |
| sphereface | Fawkes-style | 68 | 0 | +13.6 | 6.78e-20 | ours better |
| sphereface | Random noise (control) | 70 | 0 | +14.0 | 1.86e-20 | ours better |

## 3. Threshold sensitivity

PR (%) when 'protected' requires cosine distance above each threshold.

| Target | Method | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|
| facenet | Ours (3-model ensemble) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| facenet | Single (FaceNet) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| facenet | LowKey-style | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| facenet | Fawkes-style | 100.0 | 99.8 | 97.6 | 88.6 | 72.2 |
| facenet | Random noise (control) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| vggface | Ours (3-model ensemble) | 100.0 | 100.0 | 99.2 | 93.8 | 80.6 |
| vggface | Single (FaceNet) | 11.2 | 2.8 | 0.2 | 0.0 | 0.0 |
| vggface | LowKey-style | 17.0 | 4.6 | 0.6 | 0.0 | 0.0 |
| vggface | Fawkes-style | 2.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| vggface | Random noise (control) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| arcface | Ours (3-model ensemble) | 29.6 | 9.4 | 2.4 | 0.8 | 0.6 |
| arcface | Single (FaceNet) | 15.8 | 3.2 | 0.6 | 0.4 | 0.4 |
| arcface | LowKey-style | 17.0 | 3.2 | 0.8 | 0.6 | 0.6 |
| arcface | Fawkes-style | 4.4 | 0.4 | 0.2 | 0.2 | 0.2 |
| arcface | Random noise (control) | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| adaface | Ours (3-model ensemble) | 70.4 | 45.6 | 24.6 | 12.0 | 4.6 |
| adaface | Single (FaceNet) | 24.0 | 5.8 | 1.4 | 0.6 | 0.2 |
| adaface | LowKey-style | 52.0 | 27.2 | 13.2 | 5.0 | 1.2 |
| adaface | Fawkes-style | 5.8 | 0.4 | 0.0 | 0.0 | 0.0 |
| adaface | Random noise (control) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| elasticface | Ours (3-model ensemble) | 73.8 | 47.8 | 25.2 | 12.6 | 5.4 |
| elasticface | Single (FaceNet) | 17.6 | 5.8 | 1.6 | 0.0 | 0.0 |
| elasticface | LowKey-style | 54.8 | 29.2 | 13.4 | 5.0 | 1.8 |
| elasticface | Fawkes-style | 2.8 | 0.8 | 0.2 | 0.0 | 0.0 |
| elasticface | Random noise (control) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sphereface | Ours (3-model ensemble) | 14.0 | 5.0 | 2.6 | 0.8 | 0.4 |
| sphereface | Single (FaceNet) | 4.8 | 0.8 | 0.2 | 0.0 | 0.0 |
| sphereface | LowKey-style | 11.4 | 3.6 | 1.0 | 0.2 | 0.0 |
| sphereface | Fawkes-style | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| sphereface | Random noise (control) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## 4. Image quality and cost

| Method | SSIM (mean ± SD) | % images SSIM ≥ 0.90 | LPIPS (mean ± SD) | Time s/img |
|---|---|---|---|---|
| Ours (3-model ensemble) | 0.8320 ± 0.0373 | 1.2 | 0.0256 ± 0.0084 | 9.88 ± 7.03 |
| Single (FaceNet) | 0.8475 ± 0.0376 | 4.8 | 0.0220 ± 0.0079 | 3.38 ± 1.94 |
| LowKey-style | 0.8346 ± 0.0372 | 1.8 | 0.0255 ± 0.0083 | 5.53 ± 3.73 |
| Fawkes-style | 0.8576 ± 0.0371 | 11.2 | 0.0191 ± 0.0070 | 2.16 ± 1.33 |
| Random noise (control) | 0.8922 ± 0.0271 | 44.0 | 0.0318 ± 0.0178 | 0.00 ± 0.00 |

## 5. ArcFace (buffalo_l) face-detection check

Original images with no face detected: **0**  
Ours (3-model ensemble): 0 no-face cases; PR 29.6% → 29.6% when excluded (n=500)  
Single (FaceNet): 0 no-face cases; PR 15.8% → 15.8% when excluded (n=500)  
LowKey-style: 0 no-face cases; PR 17.0% → 17.0% when excluded (n=500)  
Fawkes-style: 0 no-face cases; PR 4.4% → 4.4% when excluded (n=500)  
Random noise (control): 0 no-face cases; PR 0.2% → 0.2% when excluded (n=500)  

## 6. First 20 images (pilot subset)

| Target | Ours (3-model ensemble) | Single (FaceNet) | LowKey-style | Fawkes-style | Random noise (control) |
|---|---|---|---|---|---|
| facenet | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| vggface | 100.0 | 5.0 | 25.0 | 5.0 | 0.0 |
| arcface | 15.0 | 20.0 | 20.0 | 0.0 | 0.0 |
| adaface | 60.0 | 20.0 | 60.0 | 5.0 | 0.0 |
| elasticface | 85.0 | 25.0 | 55.0 | 15.0 | 0.0 |
| sphereface | 20.0 | 10.0 | 15.0 | 5.0 | 0.0 |
