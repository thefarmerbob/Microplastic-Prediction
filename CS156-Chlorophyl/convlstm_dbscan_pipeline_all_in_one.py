"""
End-to-end script that:
- fetches chlorophyll-a frames from ERDDAP and stores a normalized NPZ
- trains the SA-ConvLSTM on that NPZ
- evaluates a few samples and runs DBSCAN clustering on top percentile pixels
- saves comparison figures

This is a self-contained consolidation of the existing utilities so it can be
copied into another LLM context. Defaults target the Gulf/Arabian Sea region.
"""

import argparse
import os
import tempfile
from pathlib import Path
from urllib import parse, request
from urllib.error import HTTPError, URLError

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader, Dataset
import xarray as xr

# ---------------------------------------------------------------------------
# Model (from sa_convlstm.py)
# ---------------------------------------------------------------------------


class SA_Memory_Module(nn.Module):
    def __init__(self, input_dim, hidden_dim, patch_size=8):
        super().__init__()
        self.layer_qh = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_kh = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_vh = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_km = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_vm = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1)
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, 1)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.patch_size = patch_size

    def forward(self, h, m):
        batch_size, channel, H, W = h.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        h_patches = h.view(batch_size, channel, patch_h, self.patch_size, patch_w, self.patch_size)
        h_patches = h_patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        h_patches = h_patches.view(batch_size, channel, patch_h * patch_w, self.patch_size * self.patch_size)
        h_patches = h_patches.mean(dim=-1)

        m_patches = m.view(batch_size, channel, patch_h, self.patch_size, patch_w, self.patch_size)
        m_patches = m_patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        m_patches = m_patches.view(batch_size, channel, patch_h * patch_w, self.patch_size * self.patch_size)
        m_patches = m_patches.mean(dim=-1)

        K_h = self.layer_kh(h)
        Q_h = self.layer_qh(h)
        V_h = self.layer_vh(h)

        K_h_patches = K_h.view(batch_size, self.hidden_dim, patch_h * patch_w, -1).mean(dim=-1)
        Q_h_patches = Q_h.view(batch_size, self.hidden_dim, patch_h * patch_w, -1).mean(dim=-1)
        V_h_patches = V_h.view(batch_size, self.hidden_dim, patch_h * patch_w, -1).mean(dim=-1)
        Q_h_patches = Q_h_patches.transpose(1, 2)

        A_h = torch.softmax(torch.bmm(Q_h_patches, K_h_patches), dim=-1)
        Z_h_patches = torch.matmul(A_h, V_h_patches.permute(0, 2, 1))

        K_m = self.layer_km(m)
        V_m = self.layer_vm(m)
        K_m_patches = K_m.view(batch_size, self.hidden_dim, patch_h * patch_w, -1).mean(dim=-1)
        V_m_patches = V_m.view(batch_size, self.hidden_dim, patch_h * patch_w, -1).mean(dim=-1)

        A_m = torch.softmax(torch.bmm(Q_h_patches, K_m_patches), dim=-1)
        Z_m_patches = torch.matmul(A_m, V_m_patches.permute(0, 2, 1))

        Z_h_patches = Z_h_patches.transpose(1, 2).view(batch_size, self.input_dim, patch_h, patch_w)
        Z_m_patches = Z_m_patches.transpose(1, 2).view(batch_size, self.input_dim, patch_h, patch_w)

        Z_h = F.interpolate(Z_h_patches, size=(H, W), mode="bilinear", align_corners=False)
        Z_m = F.interpolate(Z_m_patches, size=(H, W), mode="bilinear", align_corners=False)

        W_z = torch.cat([Z_h, Z_m], dim=1)
        Z = self.layer_z(W_z)

        combined = self.layer_m(torch.cat([Z, h], dim=1))
        mo, mg, mi = torch.chunk(combined, chunks=3, dim=1)
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m
        return new_h, new_m


class SA_Convlstm_cell(nn.Module):
    def __init__(self, input_dim, hid_dim, patch_size=8):
        super().__init__()
        self.input_channels = input_dim
        self.hidden_dim = hid_dim
        self.kernel_size = 3
        self.padding = 1
        self.attention_layer = SA_Memory_Module(hid_dim, hid_dim, patch_size=patch_size)
        self.conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels + self.hidden_dim,
                out_channels=4 * self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
            ),
            nn.GroupNorm(4 * self.hidden_dim, 4 * self.hidden_dim),
        )

    def forward(self, x, hidden):
        c, h, m = hidden
        combined = torch.cat([x, h], dim=1)
        combined_conv = self.conv2d(combined)
        i, f, g, o = torch.chunk(combined_conv, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = torch.mul(f, c) + torch.mul(i, g)
        h_next = torch.mul(o, torch.tanh(c_next))
        h_next, m_next = self.attention_layer(h_next, m)
        return h_next, (c_next, h_next, m_next)


class SA_ConvLSTM_Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size // args.gpu_num
        self.img_size = (args.img_size, args.img_size)
        self.cells, self.bns = [], []
        self.n_layers = args.num_layers
        self.frame_num = args.frame_num
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.patch_size = getattr(args, "patch_size", 8)
        self.linear_conv = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.input_dim, kernel_size=1, stride=1)
        for i in range(self.n_layers):
            input_dim = self.input_dim if i == 0 else self.hidden_dim
            hidden_dim = self.hidden_dim
            self.cells.append(SA_Convlstm_cell(input_dim, hidden_dim, patch_size=self.patch_size))
            self.bns.append(nn.LayerNorm((self.hidden_dim, *self.img_size)))
        self.cells = nn.ModuleList(self.cells)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, X, hidden=None):
        actual_batch_size = X.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size=actual_batch_size, img_size=self.img_size)
        predict = []
        inputs_x = None
        for t in range(X.size(1)):
            inputs_x = X[:, t, :, :, :]
            for i, layer in enumerate(self.cells):
                inputs_x, hidden[i] = layer(inputs_x, hidden[i])
                inputs_x = self.bns[i](inputs_x)
        inputs_x = X[:, -1, :, :, :]
        for _ in range(X.size(1)):
            for i, layer in enumerate(self.cells):
                inputs_x, hidden[i] = layer(inputs_x, hidden[i])
                inputs_x = self.bns[i](inputs_x)
            inputs_x = self.linear_conv(inputs_x)
            predict.append(inputs_x)
        predict = torch.stack(predict, dim=1)
        return torch.sigmoid(predict)

    def init_hidden(self, batch_size, img_size, device=None):
        h, w = img_size
        if device is None:
            device = next(self.parameters()).device
        hidden_state = (
            torch.zeros(batch_size, self.hidden_dim, h, w).to(device),
            torch.zeros(batch_size, self.hidden_dim, h, w).to(device),
            torch.zeros(batch_size, self.hidden_dim, h, w).to(device),
        )
        states = []
        for _ in range(self.n_layers):
            states.append(hidden_state)
        return states


# ---------------------------------------------------------------------------
# Data fetching (from fetch_chlorophyll_npz.py)
# ---------------------------------------------------------------------------

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


def resize_bilinear_np(arr, target_hw):
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t.float(), size=(target_hw, target_hw), mode="bilinear", align_corners=False)
    return t.squeeze().numpy()


def fetch_and_save_npz(
    start="2020-01-01",
    end=None,
    lon_min=30,
    lon_max=80,
    lat_min=-10,
    lat_max=35,
    stride=2,
    target=128,
    out="chlorophyll_timeseries.npz",
    max_fail_streak=200,
):
    if end is None:
        end = str(np.datetime64("today", "D"))
    dates = np.array(np.arange(np.datetime64(start), np.datetime64(end) + 1))
    frames = []
    lat_ref, lon_ref = None, None
    fail_streak = 0
    for i, d in enumerate(dates):
        date_str = str(d)
        print(f"Fetching {i+1}/{len(dates)} {date_str}")
        arr, lats, lons = fetch_frame(date_str, lon_min, lon_max, lat_min, lat_max, stride)
        if arr is None:
            fail_streak += 1
            if fail_streak >= max_fail_streak:
                print(f"Stopping after {fail_streak} consecutive failures; try a later --start date.")
                break
            continue
        fail_streak = 0
        arr_ds = resize_bilinear_np(arr, target)
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
        out,
        data=norm.astype(np.float32),
        dates=dates.astype("datetime64[D]"),
        lat=lat_ref,
        lon=lon_ref,
        data_min=np.float32(data_min),
        data_max=np.float32(data_max),
    )
    print(f"Saved {out} with shape {data.shape}, norm min/max {norm.min():.4f}/{norm.max():.4f}")


# ---------------------------------------------------------------------------
# Dataset and training (from train_convlstm_chlorophyll.py)
# ---------------------------------------------------------------------------


class ChlorophyllSeqDataset(Dataset):
    def __init__(self, data, seq_in=3, seq_out=1):
        self.data = data
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.max_start = data.shape[0] - (seq_in + seq_out) + 1

    def __len__(self):
        return max(0, self.max_start)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_in]
        y = self.data[idx + self.seq_in : idx + self.seq_in + self.seq_out]
        return torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y).unsqueeze(1)


def train_model(
    data,
    data_min,
    data_max,
    epochs=10,
    batch_size=1,
    hidden_dim=64,
    lr=1e-3,
    device="cpu",
):
    seq_in, seq_out = 3, 1
    n = data.shape[0]
    split = int(n * 0.8)
    train_ds = ChlorophyllSeqDataset(data[:split], seq_in, seq_out)
    val_ds = ChlorophyllSeqDataset(data[split - seq_in - seq_out :], seq_in, seq_out)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    class Args:
        pass

    Args.batch_size = batch_size
    Args.gpu_num = 1
    Args.img_size = data.shape[1]
    Args.num_layers = 1
    Args.frame_num = seq_in
    Args.input_dim = 1
    Args.hidden_dim = hidden_dim
    Args.patch_size = 4
    model = SA_ConvLSTM_Model(Args()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            opt.zero_grad()
            out = model(xb)
            out_last = out[:, -1:, ...]
            loss = loss_fn(out_last, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"Epoch {ep+1} train loss {total/len(train_ds):.4f}")

    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(val_loader))
        xb = xb.to(device).float()
        yb = yb.to(device).float()
        pred = model(xb)[:, -1:, ...]
    return model, xb.cpu(), yb.cpu(), pred.cpu()


def eval_test(data, model, seq_in=3, seq_out=1, num_samples=5, device="cpu"):
    ds = ChlorophyllSeqDataset(data, seq_in, seq_out)
    starts = list(range(max(0, len(ds) - num_samples), len(ds)))
    samples = []
    for idx in starts:
        xb, yb = ds[idx]
        xb_t = xb.unsqueeze(0).to(device).float()
        with torch.no_grad():
            pred = model(xb_t)[:, -1:, ...]
        samples.append((xb.squeeze(1), yb.squeeze(1), pred.cpu().squeeze(1)))
    return samples


# ---------------------------------------------------------------------------
# Visualization with DBSCAN (from dbscan_on_convlstm_preds.py)
# ---------------------------------------------------------------------------


def run_inference_with_dbscan(
    npz_path="chlorophyll_timeseries.npz",
    weights_path="convlstm_chlorophyll.pth",
    seq_in=3,
    seq_out=1,
    num_samples=5,
    threshold=0.5,
    threshold_percentile=99,
    eps_km=3,
    min_samples=5,
    out_path="convlstm_dbscan_analysis.png",
    device=None,
):
    npz = np.load(npz_path)
    data = npz["data"]  # normalized
    data_min = float(npz["data_min"])
    data_max = float(npz["data_max"])
    lat = npz.get("lat", None)
    lon = npz.get("lon", None)

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class Args:
        pass

    Args.batch_size = 1
    Args.gpu_num = 1
    Args.img_size = data.shape[1]
    Args.num_layers = 1
    Args.frame_num = seq_in
    Args.input_dim = 1
    Args.hidden_dim = 64
    Args.patch_size = 4
    model = SA_ConvLSTM_Model(Args()).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    samples = eval_test(data, model, seq_in=seq_in, seq_out=seq_out, num_samples=num_samples, device=device)

    def denorm(z):
        return z * (data_max - data_min) + data_min

    def to_log(z):
        return np.log10(np.clip(z, 1e-3, None))

    if lat is not None and lon is not None:
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    else:
        lat_grid = lon_grid = None

    cols = len(samples)
    fig, axes = plt.subplots(3, cols, figsize=(4 * cols, 9), sharex=True, sharey=True)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    cmap_main = plt.cm.viridis
    cmap_main.set_bad(color="#dcdcdc")
    cmap_err = plt.cm.Reds
    cmap_err.set_bad(color="#dcdcdc")

    prepared = []
    for _, y, p in samples:
        gt_lin = denorm(np.array(y).squeeze())
        pred_lin = denorm(np.array(p).squeeze())
        land_mask = gt_lin < 0.01
        gt_disp = np.ma.masked_where(land_mask, to_log(gt_lin))
        pred_disp = np.ma.masked_where(land_mask, to_log(pred_lin))
        err = np.abs(pred_lin - gt_lin)
        err_masked = np.ma.masked_where(land_mask, err)
        prepared.append(
            dict(
                gt_lin=gt_lin,
                pred_lin=pred_lin,
                gt_disp=gt_disp,
                pred_disp=pred_disp,
                err_masked=err_masked,
                land_mask=land_mask,
            )
        )

    main_vmin = min(np.ma.min(item["gt_disp"]) for item in prepared)
    main_vmax = max(np.ma.max(item["gt_disp"]) for item in prepared)
    main_vmin = min(main_vmin, min(np.ma.min(item["pred_disp"]) for item in prepared))
    main_vmax = max(main_vmax, max(np.ma.max(item["pred_disp"]) for item in prepared))
    err_vmax = max(max(np.ma.max(item["err_masked"]) for item in prepared), 1e-6)

    for c, item in enumerate(prepared):
        gt_lin = item["gt_lin"]
        pred_lin = item["pred_lin"]
        gt_disp = item["gt_disp"]
        pred_disp = item["pred_disp"]
        err_masked = item["err_masked"]
        land_mask = item["land_mask"]

        im_pred = axes[0, c].imshow(pred_disp, cmap=cmap_main, vmin=main_vmin, vmax=main_vmax)
        axes[0, c].set_title(f"Prediction #{c+1}")
        axes[0, c].axis("off")

        im_gt = axes[1, c].imshow(gt_disp, cmap=cmap_main, vmin=main_vmin, vmax=main_vmax)
        axes[1, c].set_title(f"Ground truth #{c+1}")
        axes[1, c].axis("off")

        im_err = axes[2, c].imshow(err_masked, cmap=cmap_err, vmin=0, vmax=err_vmax)
        axes[2, c].set_title("Abs error")
        axes[2, c].axis("off")

        for frame, ax, color, label in [
            (pred_lin, axes[0, c], "yellow", "Pred"),
            (gt_lin, axes[1, c], "yellow", "GT"),
        ]:
            if threshold_percentile is not None:
                tval = np.nanpercentile(frame, threshold_percentile)
            else:
                tval = threshold
            mask = np.isfinite(frame) & (frame >= tval) & (~land_mask)
            coords = np.argwhere(mask)
            if coords.shape[0] < min_samples:
                continue
            use_geo = (
                lat_grid is not None
                and lon_grid is not None
                and lat_grid.shape == frame.shape
                and lon_grid.shape == frame.shape
            )
            if use_geo:
                lat_pts = lat_grid[mask]
                lon_pts = lon_grid[mask]
                lat_mid = np.nanmean(lat_pts)
                scale_x = np.cos(np.deg2rad(lat_mid)) * 111.0
                scale_y = 111.0
                X_scaled = np.column_stack([lon_pts * scale_x, lat_pts * scale_y])
            else:
                scale = 1.0
                X_scaled = coords * scale

            labels = DBSCAN(eps=eps_km, min_samples=min_samples).fit_predict(X_scaled)
            cluster_num = 0
            for k in sorted(set(labels)):
                if k == -1:
                    continue
                pts = coords[labels == k]
                if pts.shape[0] < min_samples:
                    continue
                cluster_num += 1
                y_min, x_min = pts.min(axis=0)
                y_max, x_max = pts.max(axis=0)
                from matplotlib.patches import Rectangle

                rect = Rectangle(
                    (x_min - 0.5, y_min - 0.5),
                    x_max - x_min + 1,
                    y_max - y_min + 1,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                    alpha=0.9,
                )
                ax.add_patch(rect)
                cluster_vals = frame[pts[:, 0], pts[:, 1]]
                mean_val = cluster_vals.mean()
                total_pixels = pts.shape[0]
                label_x = x_min - 0.5
                label_y = y_min - 0.5
                label_text = f"C{cluster_num}\n{mean_val:.2f}\n({total_pixels} px)"
                ax.text(
                    label_x,
                    label_y,
                    label_text,
                    color=color,
                    fontsize=8,
                    weight="bold",
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6, edgecolor="none"),
                )

    fig.subplots_adjust(left=0.02, right=0.92, top=0.92, bottom=0.05, wspace=0.08, hspace=0.18)
    cax_main = fig.add_axes([0.94, 0.55, 0.015, 0.35])
    fig.colorbar(im_gt, cax=cax_main, label="log10 chlorophyll-a (mg m^-3)")
    cax_err = fig.add_axes([0.94, 0.12, 0.015, 0.25])
    fig.colorbar(im_err, cax=cax_err, label="Abs error (mg m^-3)")
    if threshold_percentile is not None:
        thresh_label = f"top {threshold_percentile}th pct"
    else:
        thresh_label = f">= {threshold} mg m^-3"
    fig.suptitle(f"Chlorophyll-a DBSCAN ({thresh_label})", fontsize=14, y=0.97)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Full ConvLSTM + DBSCAN pipeline (fetch, train, evaluate, cluster).")
    p.add_argument("--fetch", action="store_true", help="Download chlorophyll data to NPZ before training.")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=str(np.datetime64("today", "D")))
    p.add_argument("--lon-min", type=float, default=30)
    p.add_argument("--lon-max", type=float, default=80)
    p.add_argument("--lat-min", type=float, default=-10)
    p.add_argument("--lat-max", type=float, default=35)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--target", type=int, default=128)
    p.add_argument("--npz-path", default="chlorophyll_timeseries.npz")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--threshold-percentile", type=float, default=99)
    p.add_argument("--eps-km", type=float, default=3)
    p.add_argument("--min-samples", type=int, default=5)
    p.add_argument("--dbscan-fig", default="convlstm_dbscan_analysis.png")
    args = p.parse_args()

    if args.fetch:
        fetch_and_save_npz(
            start=args.start,
            end=args.end,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            stride=args.stride,
            target=args.target,
            out=args.npz_path,
        )

    npz = np.load(args.npz_path)
    data = npz["data"]
    data_min = float(npz["data_min"])
    data_max = float(npz["data_max"])
    if data.shape[0] > 365:
        data = data[-365:]
        print(f"Using last 365 frames (shape now {data.shape})")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, xb, yb, pred = train_model(
        data=data,
        data_min=data_min,
        data_max=data_max,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        device=device,
    )
    torch.save(model.state_dict(), "convlstm_chlorophyll.pth")
    print("Saved model to convlstm_chlorophyll.pth")

    test_samples = eval_test(data, model, seq_in=3, seq_out=1, num_samples=args.num_samples, device=device)
    run_inference_with_dbscan(
        npz_path=args.npz_path,
        weights_path="convlstm_chlorophyll.pth",
        seq_in=3,
        seq_out=1,
        num_samples=args.num_samples,
        threshold=args.threshold,
        threshold_percentile=args.threshold_percentile,
        eps_km=args.eps_km,
        min_samples=args.min_samples,
        out_path=args.dbscan_fig,
        device=device,
    )


if __name__ == "__main__":
    main()

