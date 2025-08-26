import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import imageio.v2 as imageio
import argparse


def lat_lon_to_indices(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> Tuple[int, int]:
    """Convert lat/lon coordinates to nearest grid indices."""
    lat_idx = int(np.argmin(np.abs(lats - lat)))
    lon_idx = int(np.argmin(np.abs(lons - lon)))
    return lat_idx, lon_idx


def find_cygnss_files() -> List[Path]:
    """Try multiple possible paths for CYGNSS data and return sorted .nc files."""
    possible_paths = [
        Path("../../CYGNSS-data"),
        Path("../CYGNSS-data"),
        Path("../../Downloads/CYGNSS-data"),
        Path("../Downloads/CYGNSS-data"),
        Path("/Users/maradumitru/Downloads/CYGNSS-data"),
    ]

    for path in possible_paths:
        if path.exists():
            nc_files = sorted(path.glob("cyg.ddmi*.nc"))
            if nc_files:
                print(f"Found CYGNSS data at: {path}")
                return nc_files

    print("ERROR: No CYGNSS data files found! Checked paths:")
    for path in possible_paths:
        print(f"  - {path.absolute()} (exists: {path.exists()})")
    return []


def get_lat_lon(ds: xr.Dataset) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str], Optional[str]]:
    """Attempt to extract 1D latitude and longitude arrays and their variable names."""
    possible_lat_names = ["lat", "latitude", "y", "lat_1", "lat_2"]
    possible_lon_names = ["lon", "longitude", "x", "lon_1", "lon_2"]

    lat_name_found = None
    lon_name_found = None
    lats = None
    lons = None

    for lat_name in possible_lat_names:
        if lat_name in ds.variables:
            lats = ds[lat_name].values
            lat_name_found = lat_name
            break

    for lon_name in possible_lon_names:
        if lon_name in ds.variables:
            lons = ds[lon_name].values
            lon_name_found = lon_name
            break

    return lats, lons, lat_name_found, lon_name_found


def process_and_plot_dbscan(
    nc_file: Path,
    threshold: Optional[float] = None,
    quantile: float = 0.9,
    eps: Optional[float] = None,
    eps_cells: float = 1.5,
    min_samples: int = 3,
) -> plt.Figure:
    """
    Apply DBSCAN to high-concentration pixels within the Tsushima region
    and overlay clusters on the cropped East Asia display region.

    - threshold: concentration threshold for selecting candidate pixels
    - eps: DBSCAN epsilon in degrees (lon/lat space)
    - min_samples: DBSCAN min_samples
    """
    ds = xr.open_dataset(nc_file)
    try:
        data = ds["mp_concentration"].values.squeeze()
    except Exception:
        ds.close()
        raise

    lats, lons, lat_name, lon_name = get_lat_lon(ds)

    # Basic stats
    global_average_concentration = float(np.nanmean(data))
    print(f"File: {nc_file.name}, Global avg concentration: {global_average_concentration:.2f}")

    if lats is None or lons is None:
        print("WARNING: No lat/lon coordinates found. Falling back to index space visualization.")
        # Fallback: operate on a fixed index region similar to dbscan.py
        target_height, target_width = data.shape
        lon_start, lon_end = 380, 560
        lat_start, lat_end = 120, 170
        lon_start = max(0, min(lon_start, target_width - 1))
        lon_end = max(0, min(lon_end, target_width))
        lat_start = max(0, min(lat_start, target_height - 1))
        lat_end = max(0, min(lat_end, target_height))

        region = data[lat_start:lat_end, lon_start:lon_end]
        # Choose threshold: fixed or adaptive quantile within region
        if threshold is None:
            region_thresh = float(np.nanpercentile(region, quantile * 100))
        else:
            region_thresh = float(threshold)

        print(
            f"Fallback region stats — min:{np.nanmin(region):.2f} max:{np.nanmax(region):.2f} "
            f"mean:{np.nanmean(region):.2f} std:{np.nanstd(region):.2f} "
            f"threshold:{region_thresh:.2f} (quantile={quantile if threshold is None else 'fixed'})"
        )

        mask = (~np.isnan(region)) & (region > region_thresh)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            print("No points above threshold in fallback region.")
            fig, ax = plt.subplots(figsize=(12, 6))
            img = ax.imshow(data, aspect="equal", cmap="viridis", origin="lower")
            ax.add_patch(
                plt.Rectangle((lon_start, lat_start), lon_end - lon_start, lat_end - lat_start, linewidth=2, edgecolor="r", facecolor="none")
            )
            fig.colorbar(img, ax=ax, label="Microplastic Concentration").mappable.set_clim(10000, 21000)
            ax.set_title("DBSCAN (fallback - no clusters)")
            ax.set_xlabel("Longitude Index")
            ax.set_ylabel("Latitude Index")
            return fig

        # DBSCAN in pixel index space
        X = np.column_stack((xs, ys))
        db = DBSCAN(eps=10, min_samples=min_samples).fit(X)
        labels = db.labels_
        unique_labels = set(labels)

        fig, ax = plt.subplots(figsize=(12, 6))
        img = ax.imshow(data, aspect="equal", cmap="viridis", origin="lower")
        fig.colorbar(img, ax=ax, label="Microplastic Concentration").mappable.set_clim(10000, 21000)
        ax.add_patch(
            plt.Rectangle((lon_start, lat_start), lon_end - lon_start, lat_end - lat_start, linewidth=2, edgecolor="r", facecolor="none")
        )

        # Offset cluster points back to global index space
        colors = [plt.cm.gist_ncar(each) for each in np.linspace(0, 1, max(1, len(unique_labels)))]
        for k, col in zip(unique_labels, colors):
            class_mask = labels == k
            pts = X[class_mask]
            if k == -1:
                ax.scatter(pts[:, 0] + lon_start, pts[:, 1] + lat_start, s=4, c="k", alpha=0.6, label="noise")
            else:
                if len(pts) >= 3:
                    hull = ConvexHull(pts)
                    for simplex in hull.simplices:
                        ax.plot(
                            pts[simplex, 0] + lon_start,
                            pts[simplex, 1] + lat_start,
                            "-",
                            lw=2,
                            color=col,
                        )
                else:
                    ax.scatter(pts[:, 0] + lon_start, pts[:, 1] + lat_start, s=6, color=col)

        ax.set_xlabel("Longitude Index")
        ax.set_ylabel("Latitude Index")
        ax.set_title("DBSCAN clusters (fallback index space)")
        return fig

    # Coordinates available: define display (East Asia) and Tsushima regions
    display_sw_lat, display_sw_lon = 22.30649, 118.07623
    display_ne_lat, display_ne_lon = 36.96467, 143.9533

    japan_sw_lat, japan_sw_lon = 25.35753, 118.85766
    japan_ne_lat, japan_ne_lon = 36.98134, 145.47117

    tsushima_sw_lat, tsushima_sw_lon = 34.02837, 129.11613
    tsushima_ne_lat, tsushima_ne_lon = 34.76456, 129.55801

    # Convert to indices
    disp_sw_li, disp_sw_loi = lat_lon_to_indices(display_sw_lat, display_sw_lon, lats, lons)
    disp_ne_li, disp_ne_loi = lat_lon_to_indices(display_ne_lat, display_ne_lon, lats, lons)

    disp_lat_start = min(disp_sw_li, disp_ne_li)
    disp_lat_end = max(disp_sw_li, disp_ne_li)
    disp_lon_start = min(disp_sw_loi, disp_ne_loi)
    disp_lon_end = max(disp_sw_loi, disp_ne_loi)

    tsu_sw_li, tsu_sw_loi = lat_lon_to_indices(tsushima_sw_lat, tsushima_sw_lon, lats, lons)
    tsu_ne_li, tsu_ne_loi = lat_lon_to_indices(tsushima_ne_lat, tsushima_ne_lon, lats, lons)

    tsu_lat_start = min(tsu_sw_li, tsu_ne_li)
    tsu_lat_end = max(tsu_sw_li, tsu_ne_li)
    tsu_lon_start = min(tsu_sw_loi, tsu_ne_loi)
    tsu_lon_end = max(tsu_sw_loi, tsu_ne_loi)

    # Crop for display
    display_data = data[disp_lat_start:disp_lat_end, disp_lon_start:disp_lon_end]
    display_data_flipped = np.flipud(display_data)
    display_extent = [
        float(lons[disp_lon_start]),
        float(lons[disp_lon_end - 1]),
        float(lats[disp_lat_start]),
        float(lats[disp_lat_end - 1]),
    ]

    # Tsushima region mask and candidate points above threshold
    tsu_region = data[tsu_lat_start:tsu_lat_end, tsu_lon_start:tsu_lon_end]
    # Decide threshold for Tsushima region with adaptive fallback
    tried_desc = ""
    if threshold is None:
        quantiles_to_try = [quantile, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
        tsu_mask = np.zeros_like(tsu_region, dtype=bool)
        tsu_threshold = np.nan
        for q_try in quantiles_to_try:
            tsu_threshold = float(np.nanpercentile(tsu_region, q_try * 100))
            tsu_mask = (~np.isnan(tsu_region)) & (tsu_region >= tsu_threshold)
            if int(tsu_mask.sum()) >= max(min_samples, 5):
                tried_desc = f"adaptive q={q_try}"
                break
        # If still too few points, fallback to top-K selection
        if int(tsu_mask.sum()) < max(min_samples, 5):
            valid_vals = tsu_region[~np.isnan(tsu_region)].ravel()
            if valid_vals.size > 0:
                k = min(max(min_samples, 5), valid_vals.size)
                cutoff = float(np.partition(valid_vals, -k)[-k])
                tsu_threshold = cutoff
                tsu_mask = (~np.isnan(tsu_region)) & (tsu_region >= cutoff)
                tried_desc = f"topK k={k}"
            else:
                tsu_mask = np.zeros_like(tsu_region, dtype=bool)
                tried_desc = "no-valid"
    else:
        tsu_threshold = float(threshold)
        tsu_mask = (~np.isnan(tsu_region)) & (tsu_region >= tsu_threshold)
        tried_desc = "fixed"

    print(
        f"Tsushima stats — min:{np.nanmin(tsu_region):.2f} max:{np.nanmax(tsu_region):.2f} "
        f"mean:{np.nanmean(tsu_region):.2f} std:{np.nanstd(tsu_region):.2f} "
        f"threshold:{tsu_threshold:.2f} ({tried_desc}), candidates:{int(tsu_mask.sum())}"
    )

    tsu_ys, tsu_xs = np.where(tsu_mask)

    # Prepare lon/lat pairs for DBSCAN
    # Note: y -> latitude index, x -> longitude index
    tsu_lat_vals = lats[tsu_lat_start + tsu_ys]
    tsu_lon_vals = lons[tsu_lon_start + tsu_xs]
    coords_lonlat = np.column_stack((tsu_lon_vals, tsu_lat_vals))

    if len(coords_lonlat) == 0:
        print("No points above threshold in Tsushima region.")
        fig, ax = plt.subplots(figsize=(12, 8))
        img = ax.imshow(
            display_data_flipped,
            aspect="auto",
            cmap="viridis",
            extent=display_extent,
        )
        cbar = fig.colorbar(img, ax=ax, label="Microplastic Concentration")
        cbar.mappable.set_clim(10000, 21000)
        # Draw Tsushima box
        ax.add_patch(
            plt.Rectangle(
                (tsushima_sw_lon, tsushima_sw_lat),
                tsushima_ne_lon - tsushima_sw_lon,
                tsushima_ne_lat - tsushima_sw_lat,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                label="Tsushima Region",
            )
        )
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_title("DBSCAN (no clusters above threshold)")
        return fig

    # Determine grid resolution to set eps in degrees if not provided
    lat_step = float(np.nanmedian(np.abs(np.diff(lats)))) if lats.size > 1 else 0.1
    lon_step = float(np.nanmedian(np.abs(np.diff(lons)))) if lons.size > 1 else 0.1
    cell_deg = max(lat_step, lon_step)
    eps_used = float(eps) if eps is not None else float(eps_cells * cell_deg)
    print(f"Using DBSCAN eps={eps_used:.4f} deg (cell~{cell_deg:.4f} deg), min_samples={min_samples}")

    # DBSCAN in lon/lat degrees
    db = DBSCAN(eps=eps_used, min_samples=min_samples).fit(coords_lonlat)
    labels = db.labels_
    unique_labels = set(labels)
    n_clusters_ = len({lab for lab in unique_labels if lab != -1})
    n_noise_ = int(np.sum(labels == -1))
    print(f"Estimated clusters: {n_clusters_}, noise points: {n_noise_}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    img = ax.imshow(
        display_data_flipped,
        aspect="auto",
        cmap="viridis",
        extent=display_extent,
    )
    cbar = fig.colorbar(img, ax=ax, label="Microplastic Concentration")
    cbar.mappable.set_clim(10000, 21000)

    # Draw Tsushima region box
    ax.add_patch(
        plt.Rectangle(
            (tsushima_sw_lon, tsushima_sw_lat),
            tsushima_ne_lon - tsushima_sw_lon,
            tsushima_ne_lat - tsushima_sw_lat,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            label="Tsushima Region",
        )
    )

    colors = [plt.cm.gist_ncar(each) for each in np.linspace(0, 1, max(1, len(unique_labels)))]
    for k, col in zip(unique_labels, colors):
        class_mask = labels == k
        pts = coords_lonlat[class_mask]
        if k == -1:
            ax.scatter(pts[:, 0], pts[:, 1], s=6, c="k", alpha=0.4, label="noise")
        else:
            if len(pts) >= 3:
                hull = ConvexHull(pts)
                for simplex in hull.simplices:
                    ax.plot(pts[simplex, 0], pts[simplex, 1], "-", lw=2, color=col)
            else:
                ax.scatter(pts[:, 0], pts[:, 1], s=10, color=col)

    # Always overlay candidate points for transparency
    ax.scatter(coords_lonlat[:, 0], coords_lonlat[:, 1], s=8, c="white", alpha=0.6, edgecolors="black", linewidths=0.3, label="candidates")

    # File date/season for title
    file_name = nc_file.name
    date_str = file_name.split(".")[2][1:]
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    month_name = {
        "01": "January",
        "02": "February",
        "03": "March",
        "04": "April",
        "05": "May",
        "06": "June",
        "07": "July",
        "08": "August",
        "09": "September",
        "10": "October",
        "11": "November",
        "12": "December",
    }[month]

    if month in ["12", "01", "02"]:
        season = "Winter"
    elif month in ["03", "04", "05"]:
        season = "Spring"
    elif month in ["06", "07", "08"]:
        season = "Summer"
    else:
        season = "Autumn"

    date_title = f"{month_name} {day}, {year} - {season}"
    ax.set_title(f"DBSCAN Clusters over East Asia (Tsushima) — {date_title}\nClusters: {n_clusters_}, Global Avg: {global_average_concentration:.2f}")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(True, alpha=0.3)

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DBSCAN on Tsushima region with adaptive thresholding")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed threshold; if omitted, uses quantile within region")
    parser.add_argument("--quantile", type=float, default=0.9, help="Quantile (0-1) for adaptive threshold if --threshold not set")
    parser.add_argument("--eps", type=float, default=0.3, help="DBSCAN eps in degrees for lon/lat space")
    parser.add_argument("--min-samples", type=int, default=8, help="DBSCAN min_samples")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of frames processed")
    args = parser.parse_args()

    print(sys.executable)
    nc_files = find_cygnss_files()
    if not nc_files:
        sys.exit(1)

    limited_files = nc_files[: max(1, args.limit)]
    print(f"Processing {len(limited_files)} frame(s) (limit={args.limit})...")

    output_dir = Path("gif-images-tsushima-dbscan")
    output_dir.mkdir(exist_ok=True)
    image_files: List[Path] = []

    for idx, nc in enumerate(limited_files, start=1):
        try:
            print(f"[{idx}/{len(limited_files)}] Processing: {nc.name}")
            fig = process_and_plot_dbscan(
                nc,
                threshold=args.threshold,
                quantile=args.quantile,
                eps=args.eps,
                min_samples=args.min_samples,
            )
            out_path = output_dir / f"tsushima_dbscan_{nc.stem}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            image_files.append(out_path)
            print(f"  -> Saved: {out_path}")
        except Exception as e:
            print(f"  !! Error processing {nc.name}: {e}")
            continue

    if image_files:
        print("Creating GIF from limited frames...")
        images = [imageio.imread(p) for p in image_files]
        gif_path = Path("tsushima_dbscan_timeseries.gif")
        imageio.mimsave(gif_path, images, duration=0.5)
        print(f"GIF created: {gif_path.resolve()}")
    else:
        print("No images produced; GIF will not be created.")

