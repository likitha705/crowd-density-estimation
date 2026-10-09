"""Train the density-map regression model.

Usage:
    python train.py --data_root ./ShanghaiTech/part_B_final --epochs 100
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CrowdDataset
from evaluate import evaluate
from model import CrowdCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    train_ds = CrowdDataset(os.path.join(args.data_root, "train_data"), train=True, crop_size=args.crop)
    test_ds = CrowdDataset(os.path.join(args.data_root, "test_data"), train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)

    model = CrowdCNN(pretrained=True).to(device)
    criterion = nn.MSELoss(reduction="sum")  # pixel-wise Euclidean loss on density maps
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for img, dm, _ in train_dl:
            img, dm = img.to(device), dm.to(device)
            pred = model(img)
            loss = criterion(pred, dm) / img.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        mae, rmse = evaluate(model, test_dl, device)
        print(f"epoch {epoch:03d} | loss {running / len(train_dl):.4f} | test MAE {mae:.2f} | RMSE {rmse:.2f}")
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), os.path.join(args.out, "best.pth"))
            print(f"  saved new best (MAE {mae:.2f})")


if __name__ == "__main__":
    main()
