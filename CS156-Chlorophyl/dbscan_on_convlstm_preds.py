"""
Run inference with the trained ConvLSTM, then apply DBSCAN clustering on
high chlorophyll-a regions (>= threshold mg m^-3) for both predictions and
ground truth, and save a comparison figure (Prediction / Ground Truth / Error).

This does NOT train the ConvLSTM; it only loads the saved weights and runs
forward passes on recent samples.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import DBSCAN

from sa_convlstm import SA_ConvLSTM_Model
from train_convlstm_chlorophyll import ChlorophyllSeqDataset, eval_test


def build_model(hidden_dim=64, seq_in=3, img_size=96, device="cpu"):
    class Args:
        pass
    Args.batch_size = 1
    Args.gpu_num = 1
    Args.img_size = img_size
    Args.num_layers = 1
    Args.frame_num = seq_in
    Args.input_dim = 1
    Args.hidden_dim = hidden_dim
    Args.patch_size = 4
    model = SA_ConvLSTM_Model(Args()).to(device)
    return model


def run_inference(npz_path="chlorophyll_timeseries.npz",
                  weights_path="convlstm_chlorophyll.pth",
                  seq_in=3,
                  seq_out=1,
                  num_samples=5,
                  threshold=None,          # if None, use percentile
                  threshold_percentile=90, # top X% of values per frame
                  eps_km=75,
                  min_samples=5,
                  out_path="convlstm_dbscan_analysis.png",
                  device=None):
    npz = np.load(npz_path)
    data = npz["data"]  # normalized
    data_min = float(npz["data_min"])
    data_max = float(npz["data_max"])
    lat = npz.get("lat", None)
    lon = npz.get("lon", None)

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    print(f"Using device: {device}")

    model = build_model(hidden_dim=64, seq_in=seq_in, img_size=data.shape[1], device=device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    samples = eval_test(data, model, seq_in=seq_in, seq_out=seq_out, num_samples=num_samples, device=device)

    def denorm(z):
        return z * (data_max - data_min) + data_min
    def to_log(z):
        return np.log10(np.clip(z, 1e-3, None))

    # build lat/lon grids if available and shape-compatible with frames
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

    for c, (x, y, p) in enumerate(samples):
        gt_lin = denorm(np.array(y).squeeze())
        pred_lin = denorm(np.array(p).squeeze())
        land_mask = gt_lin <= 1e-9
        gt_disp = np.ma.masked_where(land_mask, to_log(gt_lin))
        pred_disp = np.ma.masked_where(land_mask, to_log(pred_lin))
        err = np.abs(pred_lin - gt_lin)
        err_masked = np.ma.masked_where(land_mask, err)

        vmin = min(gt_disp.min(), pred_disp.min())
        vmax = max(gt_disp.max(), pred_disp.max())
        err_max = max(err_masked.max(), 1e-6)

        im_pred = axes[0, c].imshow(pred_disp, cmap=cmap_main, vmin=vmin, vmax=vmax)
        axes[0, c].set_title(f"Prediction #{c+1}")
        axes[0, c].axis("off")

        im_gt = axes[1, c].imshow(gt_disp, cmap=cmap_main, vmin=vmin, vmax=vmax)
        axes[1, c].set_title(f"Ground truth #{c+1}")
        axes[1, c].axis("off")

        im_err = axes[2, c].imshow(err_masked, cmap=cmap_err, vmin=0, vmax=err_max)
        axes[2, c].set_title("Abs error")
        axes[2, c].axis("off")

        # DBSCAN on thresholded regions for both pred and gt
        for frame, ax, color, label in [
            (pred_lin, axes[0, c], "orange", "Pred"),
            (gt_lin, axes[1, c], "yellow", "GT"),
        ]:
            if threshold is None:
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
                # fall back to pixel units, scaled roughly to km
                lat_mid = 0.0
                scale = 1.0
                X_scaled = coords * scale

            labels = DBSCAN(eps=eps_km, min_samples=min_samples).fit_predict(X_scaled)
            for k in set(labels):
                if k == -1:
                    continue
                pts = coords[labels == k]
                if pts.shape[0] < min_samples:
                    continue
                cy, cx = pts.mean(axis=0)
                mean_val = frame[pts[:, 0], pts[:, 1]].mean()
                ax.scatter(cx, cy, s=45, facecolors="none", edgecolors=color, linewidths=1.3, alpha=0.9)
                ax.text(cx + 1, cy, f"{mean_val:.2f}", color=color, fontsize=7, weight="bold", ha="left", va="center")

    fig.subplots_adjust(left=0.02, right=0.92, top=0.92, bottom=0.05, wspace=0.08, hspace=0.18)
    cax_main = fig.add_axes([0.94, 0.55, 0.015, 0.35])
    fig.colorbar(im_gt, cax=cax_main, label="log10 chlorophyll-a (mg m^-3)")
    cax_err = fig.add_axes([0.94, 0.12, 0.015, 0.25])
    fig.colorbar(im_err, cax=cax_err, label="Abs error (mg m^-3)")
    if threshold is None:
        thresh_label = f"top {threshold_percentile}th pct"
    else:
        thresh_label = f">= {threshold} mg m^-3"
    fig.suptitle(f"Chlorophyll-a DBSCAN ({thresh_label})", fontsize=14, y=0.97)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)


if __name__ == "__main__":
    run_inference()
