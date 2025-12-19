"""
Fetch chlorophyll (chlor_a) frames from ERDDAP for Gulf/Arabian Sea,
downsample, normalize, and save to NPZ for ConvLSTM training.
"""
import numpy as np
import xarray as xr
from urllib import parse, request
from urllib.error import HTTPError, URLError
from pathlib import Path
import tempfile
import argparse
import os

ERDDAP_BASE = "https://coastwatch.noaa.gov/erddap"
DATASET_ID = "noaacwNPPN20VIIRSSCIDINEOFDaily"
VAR_NAME = "chlor_a"

def fetch_frame(date, lon_min, lon_max, lat_min, lat_max, stride):
    query = (
        f"{VAR_NAME}[({date}T00:00:00Z):1:({date}T00:00:00Z)]"
        f"[(0):1:(0)]"
        f"[({lat_max}):{stride}:({lat_min})]"
        f"[({lon_min}):{stride}:({lon_max})]"
    )
    encoded = parse.quote(query, safe="[]():,.-+TZ")
    url = f"{ERDDAP_BASE}/griddap/{DATASET_ID}.nc?{encoded}"
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name
    try:
    request.urlretrieve(url, tmp_path)
    ds = xr.open_dataset(tmp_path)
    da = ds[VAR_NAME].squeeze().transpose("latitude", "longitude")
    arr = np.array(da)
    arr = np.nan_to_num(arr, nan=0.0)
        return arr, da.latitude.values, da.longitude.values
    except (HTTPError, URLError) as e:
        print(f"Skip {date}: HTTP error {e}")
        return None, None, None
    finally:
        if os.path.exists(tmp_path):
    os.remove(tmp_path)

def resize_bilinear(arr, target_hw):
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t.float(), size=(target_hw, target_hw), mode="bilinear", align_corners=False)
    return t.squeeze().numpy()

def main():
    today = str(np.datetime64("today", "D"))
    p = argparse.ArgumentParser()
    # default to a long window (VIIRS daily archive) to maximize training data
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=today)
    p.add_argument("--lon-min", type=float, default=30)
    p.add_argument("--lon-max", type=float, default=80)
    p.add_argument("--lat-min", type=float, default=-10)
    p.add_argument("--lat-max", type=float, default=35)
    # lower stride and higher target to keep more spatial detail
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--target", type=int, default=128)
    p.add_argument("--out", default="chlorophyll_timeseries.npz")
    p.add_argument("--max-fail-streak", type=int, default=200, help="Stop early after this many consecutive failed days.")
    args = p.parse_args()

    dates = np.array(np.arange(np.datetime64(args.start), np.datetime64(args.end) + 1))
    frames = []
    lat_ref, lon_ref = None, None
    fail_streak = 0
    for i, d in enumerate(dates):
        date_str = str(d)
        print(f"Fetching {i+1}/{len(dates)} {date_str}")
        arr, lats, lons = fetch_frame(date_str, args.lon_min, args.lon_max, args.lat_min, args.lat_max, args.stride)
        if arr is None:
            fail_streak += 1
            if fail_streak >= args.max_fail_streak:
                print(f"Stopping after {fail_streak} consecutive failures; try a later --start date.")
                break
            continue
        fail_streak = 0
        arr_ds = resize_bilinear(arr, args.target)
        frames.append(arr_ds)
        if lat_ref is None:
            lat_ref, lon_ref = lats, lons
    if not frames:
        raise SystemExit("No frames downloaded; check date range/ERDDAP availability.")
    data = np.stack(frames, axis=0)  # T,H,W
    data_min = data.min()
    data_max = data.max()
    norm = (data - data_min) / (data_max - data_min + 1e-9)
    np.savez_compressed(
        args.out,
        data=norm.astype(np.float32),
        dates=dates.astype('datetime64[D]'),
        lat=lat_ref,
        lon=lon_ref,
        data_min=np.float32(data_min),
        data_max=np.float32(data_max),
    )
    print(f"Saved {args.out} with shape {data.shape}, norm min/max {norm.min():.4f}/{norm.max():.4f}")

if __name__ == "__main__":
    main()


