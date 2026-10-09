# Crowd Density Estimation using CNN Density-Map Regression

Mini project: Deep Learning & Computer Vision.
Estimates the number of people in surveillance frames by regressing a **density map**
(sum of the map = crowd count), and exposes the result in a **venue crowd-safety dashboard**.

## 1. Problem
Counting people by detection fails in dense crowds (occlusion, tiny heads). Instead, a CNN
predicts a per-pixel density map; integrating it gives the count and also shows *where* the crowd is dense.

## 2. Pipeline
1. **Data** – public crowd datasets with head-point annotations (ShanghaiTech Part A/B, UCF-QNRF, UCF_CC_50).
   Download ShanghaiTech and keep the `train_data/` and `test_data/` folders.
2. **Pre-processing** (`dataset.py`)
   - Head points -> Gaussian-blurred ground-truth density map (sigma = 4).
   - Random 256x256 crops, horizontal flip, brightness jitter (train only).
   - ImageNet normalisation; density map sum-pooled to 1/8 scale so count is preserved.
3. **Model** (`model.py`) – CSRNet-style: pretrained VGG16-BN front-end (conv1–conv4) + 6 dilated
   convolution layers (dilation 2) + 1x1 conv -> 1-channel density map.
4. **Training** (`train.py`) – MSE loss between predicted and GT density maps, Adam (lr 1e-5), best checkpoint by test MAE.
5. **Evaluation** (`evaluate.py`) – held-out test split; metrics:
   - MAE = mean(|pred - gt|)
   - RMSE = sqrt(mean((pred - gt)^2))
6. **Dashboard** (`dashboard.py`) – Streamlit app: upload frames -> count, density (people/m^2),
   heatmap overlay, capacity %, SAFE/WARNING/CRITICAL status, trend chart and CSV log.

## 3. Run
```bash
pip install -r requirements.txt
python train.py --data_root ./ShanghaiTech/part_B_final --epochs 100
python evaluate.py --data_root ./ShanghaiTech/part_B_final --weights checkpoints/best.pth
streamlit run dashboard.py
```
Tip: no GPU? Use Google Colab (free T4); Part B trains in well under an hour.

## 4. Expected results
CSRNet reports MAE ~68 on ShanghaiTech Part A and ~10.6 on Part B. A short mini-project run
(fewer epochs) will be somewhat worse, which is fine; report your own numbers and a few heatmap screenshots.

## 5. Safety thresholds (dashboard)
| Status   | Rule                                   |
|----------|-----------------------------------------|
| SAFE     | < 80% capacity and < 2.5 people/m^2     |
| WARNING  | >= 80% capacity or >= 2.5 people/m^2    |
| CRITICAL | >= 100% capacity or >= 4 people/m^2     |

## 6. Limitations & future work
- Trained on static datasets; needs fine-tuning on the target venue's camera angle.
- Density per m^2 assumes a known, flat visible floor area (a perspective/homography map would be more accurate).
- Extensions: live RTSP video stream, alerts (SMS/email), temporal smoothing, ROI zones per camera.

## 7. Files
`dataset.py` · `model.py` · `train.py` · `evaluate.py` · `dashboard.py` · `requirements.txt`
