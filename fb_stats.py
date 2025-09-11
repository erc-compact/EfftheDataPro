#!/usr/bin/env python3
"""
Filterbank channel statistics with flexible, separable plotting.
Author: Fazal Kareem
Version: V2.1
Date: 12-08-2025

NEW in this version
- Z-score heatmap (robust, per-channel z across time)
- Quantile ribbons over time (10/50/90%) for means or stds
- Mean–Std hexbin density summary
- Spectral Kurtosis (SK) per channel & gulp + SK heatmap
- Dynamic spectrum waterfall (mean-subtracted) over a requested time slice
- Frequency-labeled axes (MHz), correct for LSB/USB automatically; optional RFI markers
- Export an RFI mask based on robust z-thresholds
- Load previously computed stats from .npz (skip recomputation) with metadata fallbacks

Core features retained
- Streams a .fil file in gulps; computes per-channel mean & std per gulp
- Stores results in a compressed .npz (configurable name)
- Choose which plots to render via --plots / --exclude
- Save all plots in one figure or as separate files with --separate
- Figure size auto-adapts to number of plots; width/height, DPI configurable
- Percentile clipping for heatmaps (robust to outliers)
- Time axis in gulp index or seconds (from header tsamp)

Examples
  python fb_stats.py -f data.fil --plots medians mean_heatmap std_heatmap zscore_heatmap sk_heatmap \
      --x-as-time --pclip-mean 1 99 --pclip-std 1 99 --zmax 6

  python fb_stats.py --npz-in filterbank_stats.npz --plots quantile_ribbons sk_heatmap \
      --x-as-time --tsamp 6.4e-5 --freq-axis --fch1 1600 --foff -0.293

  python fb_stats.py -f data.fil --plots waterfall --waterfall-start-sec 120 --waterfall-dur-sec 30 \
      --freq-axis --rfi-freqs 1176.45 1575.42 2400 --separate --outdir figs

  python fb_stats.py -f data.fil --plots quantile_ribbons hexbin_mean_std \
      --quantile-metric std --hexbin-gridsize 60 --title "NGC 6401"

  python fb_stats.py --npz-in filterbank_stats.npz --export-mask rfi_mask.txt --z-thr 5 --mask-fraction 0.10
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sigpyproc.readers import FilReader

# -----------------------------
# Compute stats
# -----------------------------


def compute_stats(fil: FilReader, nsamples_total: int, gulp_size: int):
    """Compute per-gulp per-channel mean, std, and spectral kurtosis (SK).

    Returns
    -------
    mean_arr : (n_gulps, n_chans)
    std_arr  : (n_gulps, n_chans)
    sk_arr   : (n_gulps, n_chans)
    med_means: (n_gulps,)
    med_stds : (n_gulps,)
    """
    nchans = fil.header.nchans
    mean_list, std_list, sk_list = [], [], []
    med_means, med_stds = [], []

    for i in range(0, nsamples_total, gulp_size):
        nsamps = min(gulp_size, nsamples_total - i)
        block = fil.read_block(i, nsamps)
        # data shape: (nchans, nsamps) in Fortran order
        data = block.data.astype(np.float64).reshape((nchans, nsamps), order="F")

        mean_ch = data.mean(axis=1)
        std_ch = data.std(axis=1)

        # Spectral Kurtosis estimator per channel over time samples in this gulp
        # For power-like data x, SK ≈ ((M+1)/(M-1)) * ( (M * sum(x^2) / sum(x)^2) - 1 )
        M = nsamps
        S1 = data.sum(axis=1)
        S2 = (data**2).sum(axis=1)
        denom = np.where(S1 == 0, np.nan, S1**2)
        sk = ((M + 1) / max(M - 1, 1)) * (
            np.where(denom == 0, np.nan, (M * S2) / denom) - 1
        )

        med_means.append(np.median(mean_ch))
        med_stds.append(np.median(std_ch))

        mean_list.append(mean_ch)
        std_list.append(std_ch)
        sk_list.append(sk)

        print(
            f"Gulp {i}: MedMean={med_means[-1]:.3f}, MedStd={med_stds[-1]:.3f}, "
            f"Ch0 Mean={mean_ch[0]:.3f}, SK0={sk[0]:.3f}"
        )

    mean_arr = np.stack(mean_list)
    std_arr = np.stack(std_list)
    sk_arr = np.stack(sk_list)
    med_means = np.array(med_means)
    med_stds = np.array(med_stds)
    return mean_arr, std_arr, sk_arr, med_means, med_stds


# -----------------------------
# Helpers
# -----------------------------


def robust_z(arr: np.ndarray, axis: int) -> np.ndarray:
    """Robust z-score using median and MAD along `axis`.
    Returns an array of same shape as input (z-scored along axis).
    """
    med = np.nanmedian(arr, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(arr - med), axis=axis, keepdims=True)
    mad = np.where(mad == 0, 1e-9, mad)
    return (arr - med) / (1.4826 * mad)


def percentile_clip(arr: np.ndarray, lo_hi: Optional[Tuple[float, float]]):
    if lo_hi is None:
        return None, None
    plo, phi = np.nanpercentile(arr, [lo_hi[0], lo_hi[1]])
    return float(plo), float(phi)


def chan_frequencies(fch1: float, foff: float, nch: int) -> np.ndarray:
    return fch1 + foff * np.arange(nch)


# -----------------------------
# Plotters
# -----------------------------


def plot_medians(ax, t, med_means, med_stds, title=None):
    fig = ax.figure  # Get the parent figure from ax
    title_prefix=""
    # Create 2 vertically stacked subplots in the same space as ax
    # This assumes ax is part of a subplot layout; we replace it
    gs = ax.get_subplotspec().subgridspec(2, 1, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # --- Top: Median of channel means ---
    ax1.plot(t, med_means, color="tab:blue")
    ax1.set_ylabel("Median Mean")
    ax1.set_title(f"{title_prefix} Median of Channel Means")
    ax1.grid(True)

    # --- Bottom: Median of channel stds ---
    ax2.plot(t, med_stds, color="tab:orange")
    ax2.set_ylabel("Median Std")
    ax2.set_xlabel("Gulp Index")
    ax2.set_title(f"{title_prefix} Median of Channel Stds")
    ax2.grid(True)



def plot_quantile_ribbons(ax, t, arr, title, ylabel):
    q10 = np.nanquantile(arr, 0.10, axis=1)
    q50 = np.nanquantile(arr, 0.50, axis=1)
    q90 = np.nanquantile(arr, 0.90, axis=1)
    ax.fill_between(t, q10, q90, alpha=0.3, label="10–90%")
    ax.plot(t, q50, label="50% (median)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


def plot_heatmap(
    ax,
    data_T,
    *,
    xlabel,
    ylabel,
    title,
    cmap,
    vmin=None,
    vmax=None,
    cbar_label="",
    extent=None,
    rfi_freqs=None,
):
    im = ax.imshow(
        data_T,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)
    if rfi_freqs is not None and extent is not None:
        for fr in rfi_freqs:
            ax.axhline(fr, ls="--", lw=0.8, alpha=0.6)


def plot_hexbin(ax, xvals, yvals, gridsize):
    hb = ax.hexbin(xvals, yvals, gridsize=gridsize)
    ax.set_xlabel("Mean")
    ax.set_ylabel("Std")
    ax.set_title("Mean–Std density (hexbin)")
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("count")


PLOTTERS = {
    "medians": plot_medians,
    "quantile_ribbons": plot_quantile_ribbons,
    "mean_heatmap": plot_heatmap,
    "std_heatmap": plot_heatmap,
    "zscore_heatmap": plot_heatmap,
    "sk_heatmap": plot_heatmap,
    "hexbin_mean_std": plot_hexbin,
    "waterfall": plot_heatmap,
}

DEFAULT_PLOTS = [
    "medians",
    "quantile_ribbons",
    "mean_heatmap",
    "std_heatmap",
    "zscore_heatmap",
    "sk_heatmap",
    "hexbin_mean_std",
    "waterfall",
]


# -----------------------------
# CLI
# -----------------------------


def build_argparser():
    p = argparse.ArgumentParser(
        description="Filterbank channel stats with advanced diagnostics"
    )
    p.add_argument(
        "-f",
        type=str,
        help="Path to filterbank file (needed to compute or to plot waterfall)",
    )
    p.add_argument(
        "--npz-in",
        type=str,
        help="Path to a previously saved .npz stats file to load and plot",
    )

    p.add_argument(
        "-o",
        "--outfile",
        type=str,
        default="channel_stats_all.png",
        help="Output filename if NOT using --separate",
    )
    p.add_argument(
        "--npz-out",
        type=str,
        default="filterbank_stats.npz",
        help="Where to save computed stats (when computing)",
    )
    p.add_argument(
        "-g",
        "--gulp",
        type=int,
        default=100000,
        help="Gulp size (samples per read); ignored if loading .npz with gulp_size inside",
    )
    p.add_argument(
        "-n",
        type=int,
        default=None,
        help="Number of samples to process (default: whole file); ignored when loading .npz",
    )

    # Plot selection
    p.add_argument(
        "--plots",
        nargs="+",
        choices=sorted(PLOTTERS.keys()),
        default=DEFAULT_PLOTS,
        help="Which plots to include (order respected)",
    )
    p.add_argument(
        "--exclude",
        nargs="+",
        choices=sorted(PLOTTERS.keys()),
        default=[],
        help="Plots to exclude",
    )
    p.add_argument(
        "--separate",
        action="store_true",
        help="Save each plot to its own file in --outdir",
    )
    p.add_argument(
        "--outdir", type=str, default="figs", help="Directory for separate plot files"
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="fb",
        help="Filename prefix for separate plot files",
    )

    # Styling / layout
    p.add_argument(
        "--figwidth",
        type=float,
        default=14.0,
        help="Figure width (inches) for combined figure",
    )
    p.add_argument(
        "--per-plot-height",
        type=float,
        default=3.5,
        help="Height per subplot (inches) for combined figure",
    )
    p.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    p.add_argument(
        "--title", type=str, default=None, help="Overall figure title (combined only)"
    )

    # Axes & color
    p.add_argument(
        "--x-as-time",
        action="store_true",
        help="Use seconds on x-axis instead of gulp index",
    )
    p.add_argument(
        "--freq-axis",
        action="store_true",
        help="Label y-axis in MHz for heatmaps/waterfall",
    )
    p.add_argument(
        "--rfi-freqs",
        type=float,
        nargs="*",
        default=None,
        help="MHz lines to mark as possible RFI (e.g., 1176.45 1575.42)",
    )
    p.add_argument(
        "--cmap-mean", type=str, default="viridis", help="Colormap for mean heatmap"
    )
    p.add_argument(
        "--cmap-std", type=str, default="plasma", help="Colormap for std heatmap"
    )
    p.add_argument(
        "--cmap-z", type=str, default="coolwarm", help="Colormap for z-score heatmap"
    )
    p.add_argument(
        "--cmap-sk", type=str, default="magma", help="Colormap for SK heatmap"
    )

    # Robust clipping (percentiles)
    p.add_argument(
        "--pclip-mean",
        type=float,
        nargs=2,
        default=None,
        metavar=("PLO", "PHI"),
        help="Percentile clip for mean heatmap (e.g., 1 99)",
    )
    p.add_argument(
        "--pclip-std",
        type=float,
        nargs=2,
        default=None,
        metavar=("PLO", "PHI"),
        help="Percentile clip for std heatmap (e.g., 1 99)",
    )
    p.add_argument(
        "--zmax",
        type=float,
        default=6.0,
        help="Symmetric limit for z-score heatmap color scale (±zmax)",
    )

    # Quantile & hexbin
    p.add_argument(
        "--quantile-metric",
        choices=["mean", "std"],
        default="mean",
        help="Which metric to summarize in quantile ribbons",
    )
    p.add_argument(
        "--hexbin-gridsize",
        type=int,
        default=50,
        help="Gridsize for hexbin density plot",
    )

    # Waterfall options (require -f)
    p.add_argument(
        "--waterfall-start-sec",
        type=float,
        default=None,
        help="Start time (s) for waterfall",
    )
    p.add_argument(
        "--waterfall-dur-sec",
        type=float,
        default=10.0,
        help="Duration (s) for waterfall",
    )
    p.add_argument(
        "--pclip-waterfall",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("PLO", "PHI"),
        help="Percentile clip for waterfall intensity",
    )

    # Mask export
    p.add_argument(
        "--export-mask",
        type=str,
        default=None,
        help="Write bad-channel indices to this file",
    )
    p.add_argument(
        "--z-thr",
        type=float,
        default=5.0,
        help="|z| threshold for flagging in z/ SK z heatmaps",
    )
    p.add_argument(
        "--mask-fraction",
        type=float,
        default=0.1,
        help="Flag channel if fraction of gulps with |z|>thr exceeds this",
    )

    # Metadata fallbacks when loading legacy npz (without header info)
    p.add_argument(
        "--tsamp",
        type=float,
        default=None,
        help="Override/define tsamp (s) if missing in npz and plotting time axis",
    )
    p.add_argument(
        "--fch1",
        type=float,
        default=None,
        help="Override/define fch1 (MHz) if missing in npz and using --freq-axis",
    )
    p.add_argument(
        "--foff",
        type=float,
        default=None,
        help="Override/define foff (MHz) if missing in npz and using --freq-axis",
    )

    return p


# -----------------------------
# Main
# -----------------------------


def main():
    args = build_argparser().parse_args()

    # Load from npz or compute from fil
    meta = {}
    fil = None
    if args.npz_in:
        loaded = np.load(args.npz_in, allow_pickle=True)
        mean_arr = loaded["mean_arr"]
        std_arr = loaded["std_arr"]
        sk_arr = (
            loaded["sk_arr"] if "sk_arr" in loaded else np.nan * np.zeros_like(mean_arr)
        )
        med_means = (
            loaded["median_means"]
            if "median_means" in loaded
            else np.nanmedian(mean_arr, axis=1)
        )
        med_stds = (
            loaded["median_stds"]
            if "median_stds" in loaded
            else np.nanmedian(std_arr, axis=1)
        )
        gulp_size = int(loaded["gulp_size"]) if "gulp_size" in loaded else args.gulp
        for k in ["tsamp", "fch1", "foff", "nchans", "nsamples_total"]:
            if k in loaded:
                meta[k] = (
                    float(loaded[k])
                    if k not in ("nchans", "nsamples_total")
                    else int(loaded[k])
                )
        if args.f:
            fil = FilReader(args.f)
        print(f"Loaded stats from {args.npz_in}")
    else:
        if not args.f:
            raise SystemExit("Either provide -f to compute or --npz-in to load stats.")
        fil = FilReader(args.f)
        nsamples_total = (
            fil.header.nsamples if args.n is None else min(args.n, fil.header.nsamples)
        )
        gulp_size = int(args.gulp)
        mean_arr, std_arr, sk_arr, med_means, med_stds = compute_stats(
            fil, nsamples_total, gulp_size
        )
        np.savez(
            args.npz_out,
            mean_arr=mean_arr,
            std_arr=std_arr,
            sk_arr=sk_arr,
            median_means=med_means,
            median_stds=med_stds,
            gulp_size=gulp_size,
            tsamp=fil.header.tsamp,
            fch1=fil.header.fch1,
            foff=fil.header.foff,
            nchans=fil.header.nchans,
            nsamples_total=nsamples_total,
        )
        print(f"Saved stats to {args.npz_out}")

    # Axes domain (x)
    n_gulps, n_chans = mean_arr.shape
    tsamp = meta.get("tsamp")
    if args.x_as_time:
        if tsamp is None and fil is None:
            tsamp = args.tsamp
        if tsamp is None and fil is not None:
            tsamp = fil.header.tsamp
        if tsamp is None:
            raise SystemExit(
                "tsamp is required for time axis (provide --tsamp or a .fil)"
            )
        x = np.arange(n_gulps) * gulp_size * tsamp
        x_label = "Time (s)"
        extent_x = (0, n_gulps * gulp_size * tsamp)
    else:
        x = np.arange(n_gulps)
        x_label = "Gulp Index"
        extent_x = (0, n_gulps)

    # Frequency axis support (y)
    if args.freq_axis:
        fch1 = (
            meta.get("fch1")
            if "fch1" in meta
            else (fil.header.fch1 if fil else args.fch1)
        )
        foff = (
            meta.get("foff")
            if "foff" in meta
            else (fil.header.foff if fil else args.foff)
        )
        if fch1 is None or foff is None:
            raise SystemExit(
                "fch1/foff required for frequency axis (provide --fch1/--foff or a .fil)"
            )
        freqs = chan_frequencies(fch1, foff, n_chans)
        y_label = "Frequency (MHz)"
        y_min, y_max = (freqs.min(), freqs.max())
    else:
        y_label = "Channel"
        y_min, y_max = (0, n_chans - 1)

    # Determine plots
    plots = [p for p in args.plots if p in PLOTTERS]
    if args.exclude:
        plots = [p for p in plots if p not in set(args.exclude)]
    if not plots:
        raise SystemExit("No plots selected after applying --exclude.")

    if "waterfall" in plots and fil is None:
        raise SystemExit(
            "'waterfall' requires access to the raw .fil; provide -f along with --npz-in"
        )

    # Percentile-based color limits
    vmin_mean, vmax_mean = percentile_clip(mean_arr, args.pclip_mean)
    vmin_std, vmax_std = percentile_clip(std_arr, args.pclip_std)

    # Robust z maps (per channel across time)
    z_mean = robust_z(mean_arr, axis=0)
    z_sk = robust_z(sk_arr, axis=0)

    # Optional RFI mask export
    if args.export_mask:
        frac_bad_mean = np.mean(np.abs(z_mean) > args.z_thr, axis=0)
        frac_bad_sk = np.mean(np.abs(z_sk) > args.z_thr, axis=0)
        bad = np.where(
            (frac_bad_mean > args.mask_fraction) | (frac_bad_sk > args.mask_fraction)
        )[0]
        np.savetxt(args.export_mask, bad, fmt="%d")
        print(f"Exported bad-channel indices to {args.export_mask} (N={bad.size})")

    # Build extents for images
    extent = [extent_x[0], extent_x[1], y_min, y_max]

    # Separate files
    if args.separate:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        for key in plots:
            fig, ax = plt.subplots(
                1, 1, figsize=(args.figwidth, args.per_plot_height), dpi=args.dpi
            )
            if key == "medians":
                plot_medians(ax, x, med_means, med_stds, title=args.title)
                ax.set_xlabel(x_label)
            elif key == "quantile_ribbons":
                arr = mean_arr if args.quantile_metric == "mean" else std_arr
                plot_quantile_ribbons(
                    ax,
                    x,
                    arr,
                    f"Quantiles ({args.quantile_metric})",
                    args.quantile_metric,
                )
                ax.set_xlabel(x_label)
            elif key == "mean_heatmap":
                plot_heatmap(
                    ax,
                    mean_arr.T,
                    xlabel=x_label,
                    ylabel=y_label,
                    title="Channel-wise Mean per Gulp",
                    cmap=args.cmap_mean,
                    vmin=vmin_mean,
                    vmax=vmax_mean,
                    cbar_label="Mean",
                    extent=extent,
                    rfi_freqs=args.rfi_freqs if args.freq_axis else None,
                )
            elif key == "std_heatmap":
                plot_heatmap(
                    ax,
                    std_arr.T,
                    xlabel=x_label,
                    ylabel=y_label,
                    title="Channel-wise Std Dev per Gulp",
                    cmap=args.cmap_std,
                    vmin=vmin_std,
                    vmax=vmax_std,
                    cbar_label="Std Dev",
                    extent=extent,
                    rfi_freqs=args.rfi_freqs if args.freq_axis else None,
                )
            elif key == "zscore_heatmap":
                plot_heatmap(
                    ax,
                    np.clip(z_mean, -args.zmax, args.zmax).T,
                    xlabel=x_label,
                    ylabel=y_label,
                    title="Z-score of Means (per channel)",
                    cmap=args.cmap_z,
                    vmin=-args.zmax,
                    vmax=args.zmax,
                    cbar_label="z",
                    extent=extent,
                    rfi_freqs=args.rfi_freqs if args.freq_axis else None,
                )
            elif key == "sk_heatmap":
                plot_heatmap(
                    ax,
                    sk_arr.T,
                    xlabel=x_label,
                    ylabel=y_label,
                    title="Spectral Kurtosis",
                    cmap=args.cmap_sk,
                    vmin=None,
                    vmax=None,
                    cbar_label="SK",
                    extent=extent,
                    rfi_freqs=args.rfi_freqs if args.freq_axis else None,
                )
            elif key == "hexbin_mean_std":
                plot_hexbin(ax, mean_arr.ravel(), std_arr.ravel(), args.hexbin_gridsize)
            elif key == "waterfall":
                tsamp_eff = fil.header.tsamp
                start_sec = args.waterfall_start_sec or 0.0
                dur_sec = args.waterfall_dur_sec
                start_sample = int(start_sec / tsamp_eff)
                nsamp = int(dur_sec / tsamp_eff)
                block = fil.read_block(start_sample, nsamp)
                dyn = block.data.astype(np.float64).reshape(
                    (fil.header.nchans, nsamp), order="F"
                )
                # dyn = dyn - dyn.mean(axis=1, keepdims=True)
                vmin_w, vmax_w = percentile_clip(dyn, args.pclip_waterfall)
                extent_w = [start_sec, start_sec + dur_sec, y_min, y_max]
                plot_heatmap(
                    ax,
                    dyn,
                    xlabel="Time (s)",
                    ylabel=y_label,
                    title="Waterfall",
                    cmap="viridis",
                    vmin=vmin_w,
                    vmax=vmax_w,
                    cbar_label="arb",
                    extent=extent_w,
                    rfi_freqs=args.rfi_freqs if args.freq_axis else None,
                )
            fig.tight_layout()
            outfile = outdir / f"{args.prefix}_{key}.png"
            fig.savefig(outfile, dpi=args.dpi)
            print(f"Saved: {outfile}")
        return

    # Combined figure
    n = len(plots)
    fig_height = max(args.per_plot_height * n, 3)
    fig, axes = plt.subplots(n, 1, figsize=(args.figwidth, fig_height), dpi=args.dpi)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, plots):
        if key == "medians":
            plot_medians(ax, x, med_means, med_stds, title=None)
            ax.set_xlabel(x_label)
        elif key == "quantile_ribbons":
            arr = mean_arr if args.quantile_metric == "mean" else std_arr
            plot_quantile_ribbons(
                ax, x, arr, f"Quantiles ({args.quantile_metric})", args.quantile_metric
            )
            ax.set_xlabel(x_label)
        elif key == "mean_heatmap":
            plot_heatmap(
                ax,
                mean_arr.T,
                xlabel=x_label,
                ylabel=y_label,
                title="Channel-wise Mean per Gulp",
                cmap=args.cmap_mean,
                vmin=vmin_mean,
                vmax=vmax_mean,
                cbar_label="Mean",
                extent=extent,
                rfi_freqs=args.rfi_freqs if args.freq_axis else None,
            )
        elif key == "std_heatmap":
            plot_heatmap(
                ax,
                std_arr.T,
                xlabel=x_label,
                ylabel=y_label,
                title="Channel-wise Std Dev per Gulp",
                cmap=args.cmap_std,
                vmin=vmin_std,
                vmax=vmax_std,
                cbar_label="Std Dev",
                extent=extent,
                rfi_freqs=args.rfi_freqs if args.freq_axis else None,
            )
        elif key == "zscore_heatmap":
            plot_heatmap(
                ax,
                np.clip(z_mean, -args.zmax, args.zmax).T,
                xlabel=x_label,
                ylabel=y_label,
                title="Z-score of Means (per channel)",
                cmap=args.cmap_z,
                vmin=-args.zmax,
                vmax=args.zmax,
                cbar_label="z",
                extent=extent,
                rfi_freqs=args.rfi_freqs if args.freq_axis else None,
            )
        elif key == "sk_heatmap":
            plot_heatmap(
                ax,
                sk_arr.T,
                xlabel=x_label,
                ylabel=y_label,
                title="Spectral Kurtosis",
                cmap=args.cmap_sk,
                vmin=None,
                vmax=None,
                cbar_label="SK",
                extent=extent,
                rfi_freqs=args.rfi_freqs if args.freq_axis else None,
            )
        elif key == "hexbin_mean_std":
            plot_hexbin(ax, mean_arr.ravel(), std_arr.ravel(), args.hexbin_gridsize)
        elif key == "waterfall":
            if fil is None:
                raise SystemExit(
                    "'waterfall' requires -f (raw .fil) even when loading .npz"
                )
            tsamp_eff = fil.header.tsamp
            start_sec = args.waterfall_start_sec or 0.0
            dur_sec = args.waterfall_dur_sec
            start_sample = int(start_sec / tsamp_eff)
            nsamp = int(dur_sec / tsamp_eff)
            block = fil.read_block(start_sample, nsamp)
            dyn = block.data.astype(np.float64).reshape(
                (fil.header.nchans, nsamp), order="F"
            )
            dyn = dyn - dyn.mean(axis=1, keepdims=True)
            vmin_w, vmax_w = percentile_clip(dyn, args.pclip_waterfall)
            extent_w = [start_sec, start_sec + dur_sec, y_min, y_max]
            plot_heatmap(
                ax,
                dyn,
                xlabel="Time (s)",
                ylabel=y_label,
                title="Waterfall (mean-subtracted)",
                cmap="viridis",
                vmin=vmin_w,
                vmax=vmax_w,
                cbar_label="arb",
                extent=extent_w,
                rfi_freqs=args.rfi_freqs if args.freq_axis else None,
            )

    if args.title:
        fig.suptitle(args.title)
        fig.subplots_adjust(top=0.93)

    fig.tight_layout()
    fig.savefig(args.outfile, dpi=args.dpi)
    print(f"Saved combined figure to: {args.outfile}")


if __name__ == "__main__":
    main()
