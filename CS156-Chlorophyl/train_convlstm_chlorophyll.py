"""
Train/evaluate SA-ConvLSTM on chlorophyll NPZ and produce prediction grid.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

from sa_convlstm import SA_ConvLSTM_Model

torch.set_num_threads(2)  # keep CPU usage predictable on laptops

class ChlorophyllSeqDataset(Dataset):
    def __init__(self, data, seq_in=3, seq_out=1):
        self.data = data  # T,H,W normalized
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.max_start = data.shape[0] - (seq_in + seq_out) + 1
    def __len__(self):
        return max(0, self.max_start)
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_in]
        y = self.data[idx+self.seq_in:idx+self.seq_in+self.seq_out]
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
    # split train/val
    n = data.shape[0]
    split = int(n * 0.8)
    train_ds = ChlorophyllSeqDataset(data[:split], seq_in, seq_out)
    val_ds = ChlorophyllSeqDataset(data[split - seq_in - seq_out:], seq_in, seq_out)  # include context overlap
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
            out_last = out[:, -1:, ...]  # take last predicted frame
            loss = loss_fn(out_last, yb)
            loss.backward()
            opt.step()
            total += loss.item()*xb.size(0)
        print(f"Epoch {ep+1} train loss {total/len(train_ds):.4f}")
    # simple val forward
    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(val_loader))
        xb = xb.to(device).float()
        yb = yb.to(device).float()
        pred = model(xb)[:, -1:, ...]
    return model, xb.cpu(), yb.cpu(), pred.cpu()

def eval_test(data, model, seq_in=3, seq_out=1, num_samples=5, device="cpu"):
    ds = ChlorophyllSeqDataset(data, seq_in, seq_out)
    # take last num_samples sequences
    starts = list(range(max(0, len(ds)-num_samples), len(ds)))
    samples = []
    for idx in starts:
        xb, yb = ds[idx]
        xb_t = xb.unsqueeze(0).to(device).float()
        with torch.no_grad():
            pred = model(xb_t)[:, -1:, ...]
        samples.append((xb.squeeze(1), yb.squeeze(1), pred.cpu().squeeze(1)))
    return samples

def plot_prediction(xb, yb, pred, data_min, data_max, out_path="convlstm_pred_grid.png"):
    # use first sample; denorm and log10 for plotting
    x = xb[0].squeeze(1).numpy()  # seq, H, W
    y = yb[0, 0, 0].numpy()
    p = pred[0, 0, 0].numpy()
    def denorm(z):
        return z*(data_max - data_min) + data_min
    def to_log(z):
        return np.log10(np.clip(z, 1e-3, None))
    x_d = to_log(denorm(x))
    y_d = to_log(denorm(y))
    p_d = to_log(denorm(p))
    vmin = min(x_d.min(), y_d.min(), p_d.min())
    vmax = max(x_d.max(), y_d.max(), p_d.max())
    fig, axes = plt.subplots(1, x.shape[0]+2, figsize=(3*(x.shape[0]+2), 3))
    for i in range(x.shape[0]):
        axes[i].imshow(x_d[i], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i].set_title(f"Input t-{x.shape[0]-i}")
        axes[i].axis("off")
    axes[-2].imshow(y_d, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[-2].set_title("GT t+1")
    axes[-2].axis("off")
    im = axes[-1].imshow(p_d, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[-1].set_title("Pred t+1")
    axes[-1].axis("off")
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.01)
    cbar.set_label("log10 chlorophyll-a (mg m^-3)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)

def plot_test_samples(samples, data_min, data_max, out_path="convlstm_test_preds.png"):
    """
    Render two-row grid: top = ground truth t+1, bottom = predictions t+1.
    Adds a third row with absolute error heatmaps and masks land consistently.
    """
    def denorm(z):
        return z*(data_max - data_min) + data_min
    def to_log(z):
        return np.log10(np.clip(z, 1e-3, None))

    # collect all frames for shared color scaling
    gt_frames = []
    pred_frames = []
    diff_frames = []
    for _, y, p in samples:
        y_lin = denorm(np.array(y).squeeze())
        p_lin = denorm(np.array(p).squeeze())
        # use GT mask to mark land (zeros) across GT, pred, diff
        land_mask = y_lin <= 1e-9
        y_masked = np.ma.masked_where(land_mask, y_lin)
        p_masked = np.ma.masked_where(land_mask, p_lin)
        diff_masked = np.ma.masked_where(land_mask, np.abs(p_lin - y_lin))
        gt_frames.append(to_log(y_masked))
        pred_frames.append(to_log(p_masked))
        diff_frames.append(diff_masked)
    vmin = min(np.min(gt_frames), np.min(pred_frames))
    vmax = max(np.max(gt_frames), np.max(pred_frames))
    diff_vmax = max(np.max([np.max(d) for d in diff_frames]), 1e-6)

    cols = len(samples)
    fig, axes = plt.subplots(3, cols, figsize=(3*cols, 9), sharex=True, sharey=True)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    cmap_main = plt.cm.viridis
    cmap_main.set_bad(color="#dcdcdc")
    cmap_diff = plt.cm.Reds
    cmap_diff.set_bad(color="#dcdcdc")

    for c in range(cols):
        im_gt = axes[0, c].imshow(gt_frames[c], cmap=cmap_main, vmin=vmin, vmax=vmax)
        axes[0, c].set_title(f"Ground truth #{c+1}")
        axes[0, c].axis("off")

        im_pred = axes[1, c].imshow(pred_frames[c], cmap=cmap_main, vmin=vmin, vmax=vmax)
        axes[1, c].set_title(f"Prediction #{c+1}")
        axes[1, c].axis("off")

        im_diff = axes[2, c].imshow(diff_frames[c], cmap=cmap_diff, vmin=0, vmax=diff_vmax)
        axes[2, c].set_title("Abs error")
        axes[2, c].axis("off")

    # tighten layout but leave room on the right for colorbars
    fig.subplots_adjust(left=0.02, right=0.9, top=0.9, bottom=0.05, wspace=0.05, hspace=0.12)

    cax_main = fig.add_axes([0.92, 0.55, 0.015, 0.35])
    cbar = fig.colorbar(im_pred, cax=cax_main)
    cbar.set_label("log10 chlorophyll-a (mg m^-3)")

    cax_diff = fig.add_axes([0.92, 0.12, 0.015, 0.25])
    cbar_diff = fig.colorbar(im_diff, cax=cax_diff)
    cbar_diff.set_label("Abs error (mg m^-3)")

    fig.suptitle("Chlorophyll-a: ground truth vs predictions", fontsize=14, y=0.95)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_path)

def main():
    npz_path = Path("chlorophyll_timeseries.npz")
    if not npz_path.exists():
        raise SystemExit("Run fetch_chlorophyll_npz.py first to create chlorophyll_timeseries.npz")
    npz = np.load(npz_path)
    data = npz["data"]  # T,H,W normalized
    data_min = float(npz["data_min"])
    data_max = float(npz["data_max"])
    # keep a manageable recent window for more stable training quality
    if data.shape[0] > 365:
        data = data[-365:]
        print(f"Using last 365 frames (shape now {data.shape})")
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, xb, yb, pred = train_model(
        data,
        data_min,
        data_max,
        epochs=10,
        batch_size=1,
        hidden_dim=64,
        lr=1e-3,
        device=device,
    )
    plot_prediction(xb, yb, pred, data_min, data_max, out_path="convlstm_prediction.png")
    test_samples = eval_test(data, model, seq_in=3, seq_out=1, num_samples=5, device=device)
    plot_test_samples(test_samples, data_min, data_max, out_path="convlstm_test_preds.png")
    # save model
    torch.save(model.state_dict(), "convlstm_chlorophyll.pth")
    print("Saved model to convlstm_chlorophyll.pth")

if __name__ == "__main__":
    main()


