"""
DBSCAN clustering on chlorophyll (chlor_a) for a single day and region.
"""

import sys
print(sys.executable)

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from pathlib import Path
from scipy.spatial import ConvexHull
from urllib import parse, request
import tempfile
import os

# ERDDAP settings and region/time selection
ERDDAP_BASE = "https://coastwatch.noaa.gov/erddap"
DATASET_ID = "noaacwNPPN20VIIRSSCIDINEOFDaily"
VAR_NAME = "chlor_a"
DATE = "2025-12-05"
# Gulf/Arabian Sea box
LON_MIN, LON_MAX = 30, 80
LAT_MIN, LAT_MAX = -10, 35
# Stride to shrink download
STRIDE = 4

def fetch_subset():
    query = (
        f"{VAR_NAME}[({DATE}T00:00:00Z):1:({DATE}T00:00:00Z)]"
        f"[(0):1:(0)]"
        f"[({LAT_MAX}):{STRIDE}:({LAT_MIN})]"
        f"[({LON_MIN}):{STRIDE}:({LON_MAX})]"
    )
    encoded = parse.quote(query, safe="[]():,.-+TZ")
    url = f"{ERDDAP_BASE}/griddap/{DATASET_ID}.nc?{encoded}"
    print("Fetching:", url)
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name
    request.urlretrieve(url, tmp_path)
    ds = xr.open_dataset(tmp_path)
    da = ds[VAR_NAME].squeeze().transpose("latitude", "longitude")
    os.remove(tmp_path)
    return da

def run_dbscan(arr, eps_km=50, min_samples=10, threshold=None, out_png="chlorophyll_dbscan.png"):
    # mask NaNs
    data = np.array(arr)
    mask = np.isfinite(data)
    if threshold is not None:
        mask &= data > threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        print("No points to cluster after thresholding.")
        return
    X = np.column_stack([xs, ys])
    # crude lon/lat to km scaling: assume ~111km per deg; scale x by cos(lat_mid)
    lat_mid = 0.5 * (arr.latitude.min().item() + arr.latitude.max().item())
    scale_x = np.cos(np.deg2rad(lat_mid)) * 111.0
    scale_y = 111.0
    X_scaled = np.column_stack([X[:,0]*scale_x, X[:,1]*scale_y])
    db = DBSCAN(eps=eps_km, min_samples=min_samples).fit(X_scaled)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Clusters: {n_clusters_}, points: {len(X)}")

    fig, ax = plt.subplots(figsize=(8,6))
    img = ax.imshow(data, origin="lower", cmap="viridis")
    plt.colorbar(img, ax=ax, label="chlor_a (mg m^-3)")
    # plot clusters
    unique_labels = set(labels)
    colors = [plt.cm.gist_ncar(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0,0,0,1]
        class_mask = labels == k
        pts = X[class_mask]
        if k != -1 and len(pts) >= 3:
            hull = ConvexHull(pts)
            for simplex in hull.simplices:
                ax.plot(pts[simplex,0], pts[simplex,1], color=col, lw=2)
        else:
            ax.scatter(pts[:,0], pts[:,1], s=8, color=col, alpha=0.6)
    ax.set_title(f"DBSCAN {DATE} (eps={eps_km}km, min_samples={min_samples})")
    ax.set_xlabel("x (grid)")
    ax.set_ylabel("y (grid)")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out_png)

if __name__ == "__main__":
    da = fetch_subset()
    run_dbscan(da, eps_km=50, min_samples=8, threshold=None, out_png="chlorophyll_dbscan.png")








