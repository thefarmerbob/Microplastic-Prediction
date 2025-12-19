"""
Minimal ConvLSTM sanity run on chlorophyll data.
- Fetch a few time steps via ERDDAP for the Gulf/Arabian Sea region
- Downsample to 64x64
- Run a forward pass with SA_ConvLSTM_Model
"""
import os
import numpy as np
import xarray as xr
import torch
from pathlib import Path
from urllib import parse, request
import tempfile

from sa_convlstm import SA_ConvLSTM_Model

ERDDAP_BASE = "https://coastwatch.noaa.gov/erddap"
DATASET_ID = "noaacwNPPN20VIIRSSCIDINEOFDaily"
VAR_NAME = "chlor_a"
DATES = ["2025-11-30", "2025-12-02", "2025-12-05"]  # three frames
LON_MIN, LON_MAX = 30, 80
LAT_MIN, LAT_MAX = -10, 35
STRIDE = 4
TARGET_HW = 64

def fetch_frame(date):
    query = (
        f"{VAR_NAME}[({date}T00:00:00Z):1:({date}T00:00:00Z)]"
        f"[(0):1:(0)]"
        f"[({LAT_MAX}):{STRIDE}:({LAT_MIN})]"
        f"[({LON_MIN}):{STRIDE}:({LON_MAX})]"
    )
    encoded = parse.quote(query, safe="[]():,.-+TZ")
    url = f"{ERDDAP_BASE}/griddap/{DATASET_ID}.nc?{encoded}"
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name
    request.urlretrieve(url, tmp_path)
    ds = xr.open_dataset(tmp_path)
    da = ds[VAR_NAME].squeeze().transpose("latitude", "longitude")
    arr = np.array(da)
    arr = np.nan_to_num(arr, nan=0.0)
    os.remove(tmp_path)
    return arr

def resize_bilinear(arr, target_hw):
    import torch.nn.functional as F
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # 1,1,H,W
    t = F.interpolate(t.float(), size=(target_hw, target_hw), mode="bilinear", align_corners=False)
    return t.squeeze(0)  # 1,H,W

def main():
    frames = []
    for d in DATES:
        arr = fetch_frame(d)
        frames.append(resize_bilinear(arr, TARGET_HW))
    X = torch.stack(frames, dim=0).unsqueeze(0)  # batch=1, seq=3, C=1, H, W
    class Args:
        batch_size=1
        gpu_num=1
        img_size=TARGET_HW
        num_layers=1
        frame_num=len(DATES)
        input_dim=1
        hidden_dim=16
        patch_size=4
    model = SA_ConvLSTM_Model(Args())
    model.eval()
    with torch.no_grad():
        out = model(X)
    np.save("convlstm_output.npy", out.detach().cpu().numpy())
    print("Forward pass ok. Output shape:", out.shape)
    print("Saved convlstm_output.npy")

if __name__ == "__main__":
    main()







