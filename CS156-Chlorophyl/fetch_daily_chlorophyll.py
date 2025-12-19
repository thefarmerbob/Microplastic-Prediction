#!/usr/bin/env python
"""
Fetch and plot daily chlorophyll-a (VIIRS, NOAA ERDDAP).

Defaults:
- Source: erdVHNchla1day (near-real-time VIIRS)
- Last 7 days
- Global bbox (-180..180 lon, -80..80 lat)
- Coarsen for speed and robust percentile scaling
- Optional GIF
"""
import argparse
import csv
import datetime as dt
import os
import pathlib
from typing import Tuple
from urllib import request, parse

import numpy as np
import xarray as xr
import matplotlib

# Headless rendering and local cache to avoid font permission warnings
matplotlib.use("Agg")
HERE = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

import matplotlib.pyplot as plt  # noqa: E402
import imageio.v2 as imageio  # noqa: E402


# NOAA CoastWatch ERDDAP host
ERDDAP_BASE = "https://coastwatch.noaa.gov/erddap"
# Default to gap-filled science-quality 9km VIIRS (solid coverage)
DEFAULT_DATASET_ID = "noaacwNPPN20VIIRSSCIDINEOFDaily"
DEFAULT_VAR_NAME = "chlor_a"


def fetch_time_bounds(dataset_id: str) -> Tuple[dt.datetime, dt.datetime]:
    """Return (tmin, tmax) from ERDDAP info CSV actual_range."""
    info_url = f"{ERDDAP_BASE}/info/{dataset_id}/index.csv"
    with request.urlopen(info_url) as resp:
        text = resp.read().decode("utf-8").splitlines()
    reader = csv.DictReader(text)
    for row in reader:
        if row["Variable Name"] == "time" and row["Attribute Name"] == "actual_range":
            start_sec, end_sec = map(float, row["Value"].split(","))
            return (
                dt.datetime.utcfromtimestamp(start_sec),
                dt.datetime.utcfromtimestamp(end_sec),
            )
    raise RuntimeError("Could not determine time bounds from info CSV.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and plot daily chlorophyll-a from NOAA ERDDAP (VIIRS)."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ID, help="ERDDAP dataset ID")
    parser.add_argument("--var", default=DEFAULT_VAR_NAME, help="Variable name in dataset")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD). Default: 30 days ago.")
    parser.add_argument("--end", help="End date (YYYY-MM-DD). Default: dataset max time.")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        default=(-180, 180, -60, 60),
        help="Bounding box in degrees. Default: lon -180..180, lat -60..60.",
    )
    parser.add_argument(
        "--coarsen",
        type=int,
        default=2,
        help="Coarsen factor for speed (1 = no downsample). Default: 2.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Server-side stride (skip factor) to shrink downloads. Default: 2.",
    )
    parser.add_argument(
        "--mask-min",
        type=float,
        default=0.01,
        help="Mask values below this (treated as fill). Default: 0.01.",
    )
    parser.add_argument(
        "--mask-max",
        type=float,
        default=50.0,
        help="Mask values above this (treated as fill). Default: 50.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Use log color scale (LogNorm).",
    )
    parser.add_argument(
        "--pctl",
        type=float,
        nargs=2,
        default=(2.0, 98.0),
        metavar=("LOW", "HIGH"),
        help="Percentile bounds for color scaling. Default: 2 98.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=HERE,
        help="Output directory. Default: script directory.",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Also create an animated GIF if multiple days.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=4,
        help="Frames per second for GIF. Default: 4.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help="Max frames to sample evenly across time range. Default: 20.",
    )
    return parser.parse_args()


def default_dates(tmax: dt.datetime) -> Tuple[str, str]:
    end = tmax.date()
    start = end - dt.timedelta(days=30)
    return start.isoformat(), end.isoformat()


def robust_limits(arr: xr.DataArray, low: float, high: float):
    vals = arr.values
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return None
    vmin, vmax = np.percentile(finite, [low, high])
    return float(vmin), float(vmax)


def plot_frame(
    da: xr.DataArray,
    output: pathlib.Path,
    title: str,
    pctl_low: float,
    pctl_high: float,
    use_log: bool,
    fixed_vmin: float | None = None,
    fixed_vmax: float | None = None,
):
    vlims = (fixed_vmin, fixed_vmax) if fixed_vmin is not None else robust_limits(da, pctl_low, pctl_high)
    if vlims is None or vlims[0] is None or vlims[1] is None:
        print(f"[WARN] {output.name}: all values NaN; skipped.")
        return False
    vmin, vmax = vlims

    if use_log:
        from matplotlib.colors import LogNorm

        vmin = max(vmin, 1e-3)
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))
        vmin_arg = vmax_arg = None
    else:
        norm = None
        vmin_arg, vmax_arg = vmin, vmax

    lon_min_f = float(da["longitude"].min())
    lon_max_f = float(da["longitude"].max())
    lat_min_f = float(da["latitude"].min())
    lat_max_f = float(da["latitude"].max())
    mid_lat = 0.5 * (lat_min_f + lat_max_f)
    # Rough geographic aspect correction
    aspect = 1.0 / max(np.cos(np.deg2rad(mid_lat)), 1e-6)
    lon_span = abs(lon_max_f - lon_min_f)
    lat_span = abs(lat_max_f - lat_min_f)
    # choose a base height and scale width by span and cos factor
    base_h = 6
    base_w = max(6, base_h * (lon_span * np.cos(np.deg2rad(mid_lat))) / max(lat_span, 1e-6))
    fig, ax = plt.subplots(figsize=(base_w, base_h))
    img = ax.pcolormesh(
        da["longitude"],
        da["latitude"],
        da,
        cmap="viridis",
        shading="auto",
        vmin=vmin_arg,
        vmax=vmax_arg,
        norm=norm,
    )
    ax.set_xlim(lon_min_f, lon_max_f)
    ax.set_ylim(lat_min_f, lat_max_f)
    ax.set_aspect(aspect)
    ax.set_facecolor("#f5f5f5")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cb = plt.colorbar(img, ax=ax, label="Chlorophyll-a (mg m^-3)")
    cb.ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"[OK] {output}")
    return True


def build_subset_url(
    start: str,
    end: str,
    bbox: Tuple[float, float, float, float],
    stride: int,
    dataset_id: str,
    var_name: str,
) -> str:
    lon_min, lon_max, lat_min, lat_max = bbox
    # Dataset latitude is descending, so request max->min
    lat_lo, lat_hi = min(lat_min, lat_max), max(lat_min, lat_max)
    lat_start, lat_stop = lat_hi, lat_lo
    step = max(1, stride)
    query = (
        f"{var_name}[({start}T00:00:00Z):1:({end}T00:00:00Z)]"
        f"[(0):1:(0)]"
        f"[({lat_start}):{step}:({lat_stop})]"
        f"[({lon_min}):{step}:({lon_max})]"
    )
    encoded = parse.quote(query, safe="[]():,.-+TZ")
    return f"{ERDDAP_BASE}/griddap/{dataset_id}.nc?{encoded}"


def main():
    args = parse_args()
    dataset_id = args.dataset
    var_name = args.var
    tmin, tmax = fetch_time_bounds(dataset_id)
    if args.start or args.end:
        start_str = args.start or (tmax - dt.timedelta(days=7)).date().isoformat()
        end_str = args.end or tmax.date().isoformat()
    else:
        start_str, end_str = default_dates(tmax)

    lon_min, lon_max, lat_min, lat_max = args.bbox
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    url = build_subset_url(
        start_str,
        end_str,
        (lon_min, lon_max, lat_min, lat_max),
        stride=args.stride,
        dataset_id=dataset_id,
        var_name=var_name,
    )
    print(f"[INFO] Fetching subset via {url}")
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        request.urlretrieve(url, tmp_path)
        ds = xr.open_dataset(tmp_path)
    except Exception as exc:
        raise SystemExit(f"[ERR] download/open failed: {exc}")
    chl = ds[var_name]

    if args.coarsen > 1:
        chl = chl.coarsen(latitude=args.coarsen, longitude=args.coarsen, boundary="trim").mean()

    times = chl["time"].values
    if times.size == 0:
        raise SystemExit("[ERR] No data returned for the requested range/bbox.")
    # downsample time dimension to max_frames evenly
    if times.size > args.max_frames:
        indices = np.linspace(0, times.size - 1, args.max_frames, dtype=int)
    else:
        indices = range(times.size)

    # Compute global vmin/vmax across selected frames for consistent colorbar
    sample_frames = chl.isel(time=list(indices)).squeeze(drop=True)
    fill_attr = sample_frames.attrs.get("_FillValue")
    sample_frames = sample_frames.where(np.isfinite(sample_frames))
    if fill_attr is not None:
        sample_frames = sample_frames.where(sample_frames != fill_attr)
    sample_frames = sample_frames.where(sample_frames >= args.mask_min)
    sample_frames = sample_frames.where(sample_frames <= args.mask_max)
    vlims = robust_limits(sample_frames, args.pctl[0], args.pctl[1])
    if vlims is None:
        raise SystemExit("[ERR] Unable to compute color limits (all NaN).")
    global_vmin, global_vmax = vlims

    frame_paths = []
    for idx in indices:
        t_val = times[idx]
        frame = chl.isel(time=idx).squeeze(drop=True)
        # mask fill and outliers
        fill = frame.attrs.get("_FillValue")
        frame = frame.where(np.isfinite(frame))
        if fill is not None:
            frame = frame.where(frame != fill)
        frame = frame.where(frame >= args.mask_min)
        frame = frame.where(frame <= args.mask_max)
        date_str = np.datetime_as_string(t_val, unit="D")
        fname = out_dir / f"chlorophyll_daily_{date_str}.png"
        title = f"Chlorophyll-a (VIIRS) {date_str}"
        ok = plot_frame(
            frame,
            fname,
            title,
            args.pctl[0],
            args.pctl[1],
            args.log,
            fixed_vmin=global_vmin,
            fixed_vmax=global_vmax,
        )
        if ok:
            frame_paths.append(fname)

    if args.gif and len(frame_paths) > 1:
        images = [imageio.imread(fp) for fp in frame_paths]
        gif_path = out_dir / "chlorophyll_daily.gif"
        imageio.mimsave(gif_path, images, fps=args.gif_fps)
        print(f"[OK] GIF saved: {gif_path}")


if __name__ == "__main__":
    main()
