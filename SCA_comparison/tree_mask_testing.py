#!/usr/bin/env python
"""
Tune PlanetScope modified Normalized Difference Snow Index (NDSI) threshold using lidar canopy height models.

Rainey Aberle (rainey.aberle@usace.army.mil)
Snow-Informed Reservoir Operations (SIRO)
USACE-ERDC-CRREL
June 2026
"""

import os
from glob import glob
import rioxarray as rxr
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def confusion_counts(pred, truth):
    """Calculate confusion matrix counts."""
    valid = ~np.isnan(pred) & ~np.isnan(truth)
    p = pred.where(valid)
    t = truth.where(valid)
    tp = int(((p == 1) & (t == 1)).sum().values)
    tn = int(((p == 0) & (t == 0)).sum().values)
    fp = int(((p == 1) & (t == 0)).sum().values)
    fn = int(((p == 0) & (t == 1)).sum().values)
    return tp, tn, fp, fn


def create_tree_masks(chm_files, tree_mask_dir, height_threshold=3):
    """Create binary tree masks from canopy height models."""
    print("\nCreating binary tree masks from CHMs...")
    os.makedirs(tree_mask_dir, exist_ok=True)
    
    for chm_file in tqdm(chm_files):
        tree_mask_file = os.path.join(
            tree_mask_dir, 
            os.path.splitext(os.path.basename(chm_file))[0] + "_tree_mask.tif"
        )
        if not os.path.exists(tree_mask_file):
            with rxr.open_rasterio(chm_file, masked=True).squeeze() as chm:
                tree_mask = xr.where(np.isnan(chm), 255, chm > height_threshold).astype(np.uint16)
                tree_mask = tree_mask.rio.write_crs(chm.rio.crs)
                tree_mask = tree_mask.rio.write_nodata(255)
                tree_mask.rio.to_raster(
                    tree_mask_file,
                    nodata=255,
                    dtype=np.uint16
                )


def tune_ndsi_thresholds(pss_mosaic_files, tree_mask_dir, thresholds, chunks):
    """Tune PSS modified NDSI thresholds to match CHM tree masks."""
    print("\nTuning PSS modified NDSI thresholds to match CHM tree masks...")
    
    pss_dates = [os.path.basename(f).split('_')[0] for f in pss_mosaic_files]
    
    tree_mask_files = sorted(glob(os.path.join(tree_mask_dir, '*_mask.tif')))
    tree_mask_dates = [os.path.basename(f).split('_')[3] for f in tree_mask_files]
    tree_mask_dates = [f"{d[0:4]}-{d[4:6]}-{d[6:]}" for d in tree_mask_dates]
    tree_mask_dts = [np.datetime64(d) for d in tree_mask_dates]
    
    # List to hold confusion-matrix counts
    records = []

    for d in tqdm(pss_dates):
        d_lidar = np.where(abs(np.datetime64(d) - tree_mask_dts) <= np.timedelta64(1, 'D'))[0]
        if len(d_lidar) == 0:
            continue

        pss_mosaic_file = [x for x in pss_mosaic_files if d in x][0]
        tree_mask_file = tree_mask_files[d_lidar[0]]

        with (
            rxr.open_rasterio(pss_mosaic_file, masked=True, chunks=chunks).squeeze() as pss_mosaic,
            rxr.open_rasterio(tree_mask_file, masked=True, chunks=chunks).squeeze() as tree_mask
        ):
            pss_mosaic = (pss_mosaic / 1e4).rio.write_crs("EPSG:32611")

            # Resample CHM mask to 5 m (average -> fractional canopy cover)
            tree_mask_reproj = tree_mask.rio.reproject(
                dst_crs=tree_mask.rio.crs, res=5, resampling='average'
            )
            pss_mosaic_reproj = pss_mosaic.rio.reproject_match(tree_mask_reproj)

            # Tree if >50% canopy cover within the 5 m pixel
            truth = xr.where(np.isnan(tree_mask_reproj), np.nan, tree_mask_reproj > 0.5)

            green = pss_mosaic_reproj.isel(band=1)
            nir = pss_mosaic_reproj.isel(band=3)
            NDSI = (green - nir) / (green + nir)

            for t in thresholds:
                pred = xr.where(np.isnan(NDSI), np.nan, NDSI < t)
                tp, tn, fp, fn = confusion_counts(pred, truth)
                n = tp + tn + fp + fn
                if n == 0:
                    continue
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall    = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) else 0.0
                    )
                tpr = recall
                tnr = tn / (tn + fp) if (tn + fp) else 0.0
                bal_acc = 0.5 * (tpr + tnr)
                iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
                accuracy = (tp + tn) / n
                records.append({
                    "date": d, "threshold": float(t),
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n,
                    "precision": precision, "recall": recall, "f1": f1,
                    "balanced_accuracy": bal_acc, "iou": iou, "accuracy": accuracy,
                })
    
    return pd.DataFrame(records)


def analyze_and_plot_results(df, tree_mask_dir, selection_metric="accuracy"):
    """Analyze results and create plots."""
    if df.empty:
        print("No coincident PSS/lidar dates found.")
        return
    
    # Save the full table 
    metrics_csv = os.path.join(tree_mask_dir, "ndsi_threshold_metrics.csv")
    df.to_csv(metrics_csv, index=False)
    print(f"\nSaved full metrics table to: {metrics_csv}")    

    # ---- Calculate and display class balance ----
    print("\n" + "="*60)
    print("CLASS BALANCE ANALYSIS")
    print("="*60)
    
    # Get the confusion matrix values for each date (using first threshold as representative)
    date_stats = []
    for date in df['date'].unique():
        date_data = df[df['date'] == date].iloc[0]  # Take first threshold
        total = date_data['tp'] + date_data['tn'] + date_data['fp'] + date_data['fn']
        tree_pixels = date_data['tp'] + date_data['fn']  # All actual trees
        non_tree_pixels = date_data['tn'] + date_data['fp']  # All actual non-trees
        tree_fraction = tree_pixels / total if total > 0 else 0
        date_stats.append({
            'date': date,
            'total_pixels': total,
            'tree_pixels': tree_pixels,
            'non_tree_pixels': non_tree_pixels,
            'tree_fraction': tree_fraction
        })
    
    balance_df = pd.DataFrame(date_stats)
    
    print("\nPer-date class distribution:")
    for _, row in balance_df.iterrows():
        print(f"  {row['date']}: {row['tree_fraction']:.1%} trees, "
              f"{1-row['tree_fraction']:.1%} non-trees "
              f"(n={row['total_pixels']:,} pixels)")
    
    mean_tree_frac = balance_df['tree_fraction'].mean()
    std_tree_frac = balance_df['tree_fraction'].std()
    
    print(f"\nOverall class balance:")
    print(f"  Tree pixels: {mean_tree_frac:.1%} ± {std_tree_frac:.1%}")
    print(f"  Non-tree pixels: {1-mean_tree_frac:.1%} ± {std_tree_frac:.1%}")
    
    # Determine if dataset is imbalanced
    if mean_tree_frac > 0.6 or mean_tree_frac < 0.4:
        imbalance_ratio = max(mean_tree_frac, 1-mean_tree_frac) / min(mean_tree_frac, 1-mean_tree_frac)
        print(f"\nDataset is imbalanced (ratio {imbalance_ratio:.1f}:1)")
        print(f"    -> Accuracy may plateau due to majority class dominance")
        print(f"    -> Consider using F1, balanced accuracy, or IoU instead")
    else:
        print(f"\n✓ Dataset is relatively balanced")
    
    print("="*60 + "\n")

    # ---- Per-date best ----
    per_date_best = (df.loc[df.groupby("date")[selection_metric].idxmax()]
                       [["date", "threshold", "accuracy", "f1", "iou",
                         "balanced_accuracy", "precision", "recall"]]
                       .sort_values("date"))
    print(f"\nPer-date best threshold (by {selection_metric}):")
    print(per_date_best.to_string(index=False))

    # ---- 1. Maximize mean accuracy across dates ----
    mean_by_t = df.groupby("threshold")[
        ["accuracy", "f1", "iou", "balanced_accuracy"]].mean()
    t_mean = mean_by_t[selection_metric].idxmax()

    # ---- 2. Maximize minimum (worst-case) accuracy across dates ----
    min_by_t = df.groupby("threshold")[selection_metric].min()
    t_robust = min_by_t.idxmax()

    # ---- 3. Pooled confusion matrix (pixel-weighted) ----
    pooled = df.groupby("threshold")[["tp", "tn", "fp", "fn"]].sum()
    pooled["accuracy"] = (pooled.tp + pooled.tn) / (
        pooled.tp + pooled.tn + pooled.fp + pooled.fn)
    pooled["precision"] = pooled.tp / (pooled.tp + pooled.fp).replace(0, np.nan)
    pooled["recall"]    = pooled.tp / (pooled.tp + pooled.fn).replace(0, np.nan)
    pooled["f1"] = (2 * pooled.precision * pooled.recall
                    / (pooled.precision + pooled.recall))
    t_pooled = pooled["accuracy"].idxmax()

    print(f"\nGlobal threshold candidates (selection metric = {selection_metric}):")
    print(f"  1. Max mean accuracy across dates : {t_mean:.2f} "
          f"(mean acc = {mean_by_t.loc[t_mean, 'accuracy']:.4f})")
    print(f"  2. Max worst-case accuracy (robust): {t_robust:.2f} "
          f"(min acc = {min_by_t.loc[t_robust]:.4f})")
    print(f"  3. Pooled-pixel max accuracy: {t_pooled:.2f} "
          f"(pooled acc = {pooled.loc[t_pooled, 'accuracy']:.4f})")

    # ---- Plot multiple metrics vs. threshold ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics_to_plot = ["accuracy", "f1", "balanced_accuracy", "iou"]
    titles = ["Accuracy", "F1 Score", "Balanced Accuracy", "IoU"]
    
    for ax, metric, title in zip(axes.flat, metrics_to_plot, titles):
        # Plot individual dates
        for date, g in df.groupby("date"):
            ax.plot(g["threshold"], g[metric], alpha=0.35, lw=1)
        
        # Plot mean
        mean_metric = df.groupby("threshold")[metric].mean()
        ax.plot(mean_metric.index, mean_metric.values, "k-", lw=2.5, label="mean")
        
        # Plot min (worst-case)
        min_metric = df.groupby("threshold")[metric].min()
        ax.plot(min_metric.index, min_metric.values, "r--", lw=2, label="min (worst-case)")
        
        # Mark optimal threshold
        t_opt = mean_metric.idxmax()
        ax.axvline(t_opt, color="k", ls=":", alpha=0.7, label=f"opt = {t_opt:.2f}")
        
        ax.set_xlabel("Modified NDSI threshold")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs Modified NDSI threshold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(tree_mask_dir, "ndsi_threshold_tuning_all_metrics.png"), dpi=150)
    plt.show()
    
    # Also save the original single-metric plot
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for date, g in df.groupby("date"):
        ax2.plot(g["threshold"], g["accuracy"], alpha=0.35, lw=1, label=f"_{date}")
    ax2.plot(mean_by_t.index, mean_by_t["accuracy"], "k-", lw=2.5, label="mean accuracy")
    ax2.plot(min_by_t.index, min_by_t.values, "r--", lw=2, label="min (worst-case) accuracy")
    ax2.axvline(t_mean, color="k", ls=":", label=f"chosen mean opt = {t_mean:.2f}")
    ax2.set_xlabel("Modified NDSI threshold")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Modified NDSI threshold vs accuracy across lidar-coincident dates")
    ax2.legend(loc="best", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(tree_mask_dir, "ndsi_threshold_tuning_accuracy.png"), dpi=150)
    plt.show()


def main():
    """Main function to run the tree mask testing workflow."""
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    
    # Lidar canopy height models
    lidar_dir = "/Users/rdcrlrka/Research/SkySat-Stereo/study-sites/MCS/SNEX_MCS_Lidar"
    chm_files = sorted(glob(os.path.join(lidar_dir, '*CHM*.tif')))
    print(f"Located {len(chm_files)} canopy height files")
    
    # PlanetScope image mosaics
    pss_mosaic_files = sorted(glob("/Users/rdcrlrka/Research/SIRO/MCS_SCA/PSS_image_mosaics_clipped/*.tif"))
    print(f"Located {len(pss_mosaic_files)} PSS image mosaics")
    
    tree_mask_dir = "/Users/rdcrlrka/Research/SIRO/MCS_SCA/lidar_tree_masks"
    os.makedirs(tree_mask_dir, exist_ok=True)
    
    # Processing parameters
    chunks = {"x": 2048, "y": 2048}
    thresholds = np.arange(-0.8, 0, 0.05)
    selection_metric = "accuracy"
    
    # ---------------------------------------------------------
    # RUN WORKFLOW
    # ---------------------------------------------------------
    
    # Create binary tree masks from CHMs
    create_tree_masks(chm_files, tree_mask_dir)
    
    # Tune modified NDSI thresholds
    df = tune_ndsi_thresholds(pss_mosaic_files, tree_mask_dir, thresholds, chunks)
    
    # Analyze and plot results
    analyze_and_plot_results(df, tree_mask_dir, selection_metric)


if __name__ == "__main__":
    main()