"""Evaluation on held-out data: MAE and RMSE of predicted vs. true counts.

Usage:
    python evaluate.py --data_root ./ShanghaiTech/part_B_final --weights checkpoints/best.pth
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CrowdDataset
from model import CrowdCNN


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    errs = []
    for img, _, gt_count in loader:
        pred = model(img.to(device)).sum().item()
        errs.append(pred - float(gt_count))
    errs = np.array(errs)
    return float(np.abs(errs).mean()), float(np.sqrt((errs ** 2).mean()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--weights", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrowdCNN(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    ds = CrowdDataset(os.path.join(args.data_root, "test_data"), train=False)
    mae, rmse = evaluate(model, DataLoader(ds, batch_size=1), device)
    print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}")
