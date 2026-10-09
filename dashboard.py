"""Venue crowd-safety dashboard (Streamlit).

Run:  streamlit run dashboard.py

Upload a camera frame (or several), set the visible floor area of the camera view and
the venue's safe capacity, and the dashboard shows the estimated count, density heatmap,
people-per-square-metre, a safety status and a running log/trend.
"""
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from dataset import DOWNSAMPLE, preprocess
from model import CrowdCNN

st.set_page_config(page_title="Crowd Safety Dashboard", layout="wide")
st.title("Venue Crowd-Safety Dashboard")


@st.cache_resource
def load_model(weights_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrowdCNN(pretrained=False).to(device)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device


def predict(model, device, img_rgb, max_side=1024):
    h, w = img_rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img_rgb = cv2.resize(img_rgb, (int(w * s), int(h * s)))
    h8, w8 = img_rgb.shape[0] // DOWNSAMPLE * DOWNSAMPLE, img_rgb.shape[1] // DOWNSAMPLE * DOWNSAMPLE
    img_rgb = img_rgb[:h8, :w8]
    x = torch.from_numpy(preprocess(img_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        dm = model(x)[0, 0].cpu().numpy()
    return img_rgb, np.clip(dm, 0, None)


def overlay_heatmap(img_rgb, dm):
    heat = cv2.resize(dm, (img_rgb.shape[1], img_rgb.shape[0]))
    heat = (255 * heat / (heat.max() + 1e-8)).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 0.55, heat, 0.45, 0)


def status_for(count, capacity, density):
    ratio = count / capacity
    if ratio >= 1.0 or density >= 4.0:
        return "CRITICAL", "red"
    if ratio >= 0.8 or density >= 2.5:
        return "WARNING", "orange"
    return "SAFE", "green"


with st.sidebar:
    st.header("Settings")
    weights = st.text_input("Model weights path", "checkpoints/best.pth")
    capacity = st.number_input("Safe capacity of monitored zone (people)", 1, 100000, 200)
    area = st.number_input("Visible floor area (m²)", 1.0, 100000.0, 100.0)
    st.caption("Density > 2.5 p/m² = warning, > 4 p/m² = critical (common crowd-safety rules of thumb).")

if "log" not in st.session_state:
    st.session_state.log = []

try:
    model, device = load_model(weights)
except Exception as e:
    st.error(f"Could not load weights ({e}). Train the model first with train.py.")
    st.stop()

files = st.file_uploader("Upload camera frame(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

for f in files or []:
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img, dm = predict(model, device, img)
    count = float(dm.sum())
    density = count / area
    label, color = status_for(count, capacity, density)
    st.session_state.log.append({"time": time.strftime("%H:%M:%S"), "frame": f.name,
                                 "count": round(count, 1), "density_pm2": round(density, 2), "status": label})

    st.subheader(f.name)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estimated count", f"{count:.0f}")
    c2.metric("Density (people/m²)", f"{density:.2f}")
    c3.metric("Capacity used", f"{100 * count / capacity:.0f}%")
    c4.markdown(f"### Status: :{color}[{label}]")
    left, right = st.columns(2)
    left.image(img, caption="Input frame", use_container_width=True)
    right.image(overlay_heatmap(img, dm), caption="Predicted density heatmap", use_container_width=True)

if st.session_state.log:
    st.divider()
    st.subheader("Log & trend")
    df = pd.DataFrame(st.session_state.log)
    st.line_chart(df.set_index("time")["count"])
    st.dataframe(df, use_container_width=True)
    st.download_button("Download log (CSV)", df.to_csv(index=False), "crowd_log.csv")
