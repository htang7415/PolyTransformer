#!/usr/bin/env python
"""Aggregate Step1/Step2/Step3 metrics and create summary figures.

Designed for AR method folders (up to five methods by default).

Step1:
  - Extracts validation loss history and reports BPB (bits-per-byte) as:
      BPB = val_loss / ln(2)

Step2:
  - Aggregates generation metrics:
      validity, uniqueness, novelty, avg_diversity, mean_sa

Step3:
  - Aggregates split-level property metrics (train/val/test) with focus on R2.
  - Preferred source: {property}_split_metrics.csv
  - Fallback source: {property}_test_metrics.csv (test only)

This script is robust to partially-complete runs and writes empty CSVs when
metrics are missing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.metrics_schema import ensure_columns, infer_model_size, parse_method_representation


KNOWN_METHOD_ORDER = [
    "AR_Transformer_SMILES",
    "AR_Transformer_SMILES_BPE",
    "AR_Transformer_SELFIES",
    "AR_Transformer_Group_SELFIES",
    "AR_Transformer_Graph",
]
REPRESENTATION_ORDER = ["SMILES", "SMILES_BPE", "SELFIES", "Group_SELFIES", "Graph"]
MODEL_SIZE_ORDER = ["base", "small", "medium", "large", "xl"]
PROPERTY_ORDER = ["Tg", "Tm", "Eg", "Td"]

STEP1_COLUMNS = [
    "method_dir",
    "method",
    "representation",
    "model_size",
    "results_dir",
    "val_points",
    "final_val_loss",
    "best_val_loss",
    "final_val_bpb",
    "best_val_bpb",
    "source_file",
]

STEP2_COLUMNS = [
    "method_dir",
    "method",
    "representation",
    "model_size",
    "results_dir",
    "sample_id",
    "n_total",
    "n_valid",
    "validity",
    "validity_two_stars",
    "frac_star_eq_2",
    "uniqueness",
    "novelty",
    "avg_diversity",
    "mean_sa",
    "std_sa",
    "samples_per_sec",
    "valid_per_sec",
]

STEP3_COLUMNS = [
    "method_dir",
    "method",
    "representation",
    "model_size",
    "results_dir",
    "property",
    "split",
    "r2",
    "mae",
    "rmse",
    "source_file",
]


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json(path: Path) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_model_size(value: str) -> str:
    token = value.strip().lower()
    aliases = {
        "s": "small",
        "m": "medium",
        "l": "large",
        "x": "xl",
    }
    return aliases.get(token, token)


def _parse_model_sizes_arg(raw_value: str) -> Optional[Set[str]]:
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value or value.lower() == "all":
        return None

    allowed = {"small", "medium", "large", "xl", "base"}
    parsed = {_normalize_model_size(v) for v in value.split(",") if v.strip()}
    unknown = sorted(v for v in parsed if v not in allowed)
    if unknown:
        raise ValueError(
            f"Unsupported model size(s): {', '.join(unknown)}. "
            "Expected one or more of: small, medium, large, xl, base, all."
        )
    return parsed


def _include_model_size(model_size: str, allowed_model_sizes: Optional[Set[str]]) -> bool:
    return allowed_model_sizes is None or model_size in allowed_model_sizes


def _parse_methods_arg(root: Path, methods_arg: Optional[str], max_methods: int) -> List[str]:
    if methods_arg and methods_arg.strip():
        methods = [m.strip() for m in methods_arg.split(",") if m.strip()]
        return methods

    discovered = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and p.name.startswith("AR_Transformer_")
    )

    ordered = [m for m in KNOWN_METHOD_ORDER if m in discovered]
    ordered.extend([m for m in discovered if m not in ordered])

    if max_methods > 0:
        return ordered[:max_methods]
    return ordered


def _list_results_dirs(method_dir: Path) -> List[Path]:
    if not method_dir.exists():
        return []
    results = [p for p in method_dir.iterdir() if p.is_dir() and p.name.startswith("results")]
    return sorted(results, key=lambda p: (p.name != "results", p.name))


def _ordered_unique(values: List[str], preferred_order: List[str]) -> List[str]:
    seen = set(values)
    ordered = [x for x in preferred_order if x in seen]
    ordered.extend([x for x in values if x not in preferred_order and x not in ordered])
    return ordered


def _finite_vmin_vmax(values: np.ndarray) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmin == vmax:
        return vmin - 1e-12, vmax + 1e-12
    return vmin, vmax


def _extract_val_losses(metrics_dir: Path) -> Tuple[List[float], str]:
    # 1) Preferred: JSON history
    history_json = metrics_dir / "backbone_training_history.json"
    if history_json.exists():
        payload = _read_json(history_json)
        val_losses = payload.get("val_losses", []) if isinstance(payload, dict) else []
        vals = pd.to_numeric(pd.Series(val_losses), errors="coerce").dropna().tolist()
        if vals:
            return [float(v) for v in vals], history_json.name

    # 2) Fallback: explicit validation loss CSV
    val_csv = metrics_dir / "backbone_val_loss.csv"
    if val_csv.exists():
        df = _read_csv(val_csv)
        if not df.empty:
            for col in ["val_loss", "validation_loss", "loss"]:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
                    if vals:
                        return [float(v) for v in vals], val_csv.name

    # 3) Fallback: combined curve if val_loss exists
    curve_csv = metrics_dir / "backbone_loss_curve.csv"
    if curve_csv.exists():
        df = _read_csv(curve_csv)
        if not df.empty:
            for col in ["val_loss", "validation_loss"]:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
                    if vals:
                        return [float(v) for v in vals], curve_csv.name

    return [], ""


def _aggregate_step1(
    root: Path,
    methods: List[str],
    allowed_model_sizes: Optional[Set[str]] = None,
) -> pd.DataFrame:
    rows: List[Dict] = []

    for method_name in methods:
        method_dir = root / method_name
        if not method_dir.exists():
            print(f"WARNING: method directory missing: {method_dir}")
            continue

        info = parse_method_representation(method_name)
        for results_dir in _list_results_dirs(method_dir):
            model_size = infer_model_size(results_dir)
            if not _include_model_size(model_size, allowed_model_sizes):
                continue

            metrics_dir = results_dir / "step1_backbone" / "metrics"
            if not metrics_dir.exists():
                continue

            val_losses, source_file = _extract_val_losses(metrics_dir)
            if not val_losses:
                continue

            final_val_loss = float(val_losses[-1])
            best_val_loss = float(min(val_losses))
            ln2 = math.log(2.0)

            rows.append(
                {
                    "method_dir": method_name,
                    "method": info.method,
                    "representation": info.representation,
                    "model_size": model_size,
                    "results_dir": str(results_dir.relative_to(root)),
                    "val_points": len(val_losses),
                    "final_val_loss": final_val_loss,
                    "best_val_loss": best_val_loss,
                    "final_val_bpb": final_val_loss / ln2,
                    "best_val_bpb": best_val_loss / ln2,
                    "source_file": source_file,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=STEP1_COLUMNS)
    df = ensure_columns(df, STEP1_COLUMNS)
    return _coerce_numeric(
        df,
        ["val_points", "final_val_loss", "best_val_loss", "final_val_bpb", "best_val_bpb"],
    )


def _aggregate_step2(
    root: Path,
    methods: List[str],
    allowed_model_sizes: Optional[Set[str]] = None,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    for method_name in methods:
        method_dir = root / method_name
        if not method_dir.exists():
            continue

        info = parse_method_representation(method_name)
        for results_dir in _list_results_dirs(method_dir):
            model_size = infer_model_size(results_dir)
            if not _include_model_size(model_size, allowed_model_sizes):
                continue

            metrics_path = results_dir / "step2_sampling" / "metrics" / "sampling_generative_metrics.csv"
            if not metrics_path.exists():
                continue

            df = _read_csv(metrics_path)
            if df.empty:
                continue

            df["method_dir"] = method_name
            df["method"] = info.method
            df["representation"] = info.representation
            df["model_size"] = model_size
            df["results_dir"] = str(results_dir.relative_to(root))

            df = ensure_columns(df, STEP2_COLUMNS)
            rows.append(df[STEP2_COLUMNS])

    if not rows:
        return pd.DataFrame(columns=STEP2_COLUMNS)

    out = pd.concat(rows, ignore_index=True)
    return _coerce_numeric(
        out,
        [
            "n_total",
            "n_valid",
            "validity",
            "validity_two_stars",
            "frac_star_eq_2",
            "uniqueness",
            "novelty",
            "avg_diversity",
            "mean_sa",
            "std_sa",
            "samples_per_sec",
            "valid_per_sec",
        ],
    )


def _normalize_split(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"validation", "valid"}:
        return "val"
    if v in {"train", "val", "test"}:
        return v
    return v


def _aggregate_step3(
    root: Path,
    methods: List[str],
    allowed_model_sizes: Optional[Set[str]] = None,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    for method_name in methods:
        method_dir = root / method_name
        if not method_dir.exists():
            continue

        info = parse_method_representation(method_name)
        for results_dir in _list_results_dirs(method_dir):
            model_size = infer_model_size(results_dir)
            if not _include_model_size(model_size, allowed_model_sizes):
                continue

            for step3_dir in sorted(results_dir.glob("step3_*")):
                if not step3_dir.is_dir():
                    continue

                property_name = step3_dir.name.replace("step3_", "")
                metrics_dir = step3_dir / "metrics"
                if not metrics_dir.exists():
                    continue

                split_path = metrics_dir / f"{property_name}_split_metrics.csv"
                if split_path.exists():
                    split_df = _read_csv(split_path)
                    if not split_df.empty:
                        df = split_df.copy()
                        if "R2" in df.columns and "r2" not in df.columns:
                            df["r2"] = df["R2"]
                        if "R²" in df.columns and "r2" not in df.columns:
                            df["r2"] = df["R²"]
                        if "MAE" in df.columns and "mae" not in df.columns:
                            df["mae"] = df["MAE"]
                        if "RMSE" in df.columns and "rmse" not in df.columns:
                            df["rmse"] = df["RMSE"]
                        if "split" not in df.columns:
                            df["split"] = "test"

                        df["split"] = df["split"].apply(_normalize_split)
                        df["method_dir"] = method_name
                        df["method"] = info.method
                        df["representation"] = info.representation
                        df["model_size"] = model_size
                        df["results_dir"] = str(results_dir.relative_to(root))
                        if "property" not in df.columns:
                            df["property"] = property_name
                        df["source_file"] = split_path.name

                        df = ensure_columns(df, STEP3_COLUMNS)
                        rows.append(df[STEP3_COLUMNS])
                        continue

                test_path = metrics_dir / f"{property_name}_test_metrics.csv"
                if test_path.exists():
                    test_df = _read_csv(test_path)
                    if not test_df.empty:
                        df = test_df.copy()
                        if "R2" in df.columns and "r2" not in df.columns:
                            df["r2"] = df["R2"]
                        if "R²" in df.columns and "r2" not in df.columns:
                            df["r2"] = df["R²"]
                        if "MAE" in df.columns and "mae" not in df.columns:
                            df["mae"] = df["MAE"]
                        if "RMSE" in df.columns and "rmse" not in df.columns:
                            df["rmse"] = df["RMSE"]

                        df["split"] = "test"
                        df["method_dir"] = method_name
                        df["method"] = info.method
                        df["representation"] = info.representation
                        df["model_size"] = model_size
                        df["results_dir"] = str(results_dir.relative_to(root))
                        if "property" not in df.columns:
                            df["property"] = property_name
                        df["source_file"] = test_path.name

                        df = ensure_columns(df, STEP3_COLUMNS)
                        rows.append(df[STEP3_COLUMNS])

    if not rows:
        return pd.DataFrame(columns=STEP3_COLUMNS)

    out = pd.concat(rows, ignore_index=True)
    out = _coerce_numeric(out, ["r2", "mae", "rmse"])
    if "split" in out.columns:
        out["split"] = out["split"].apply(_normalize_split)
    return out


def _method_palette(representations: List[str]) -> Dict[str, str]:
    base = {
        "SMILES": "#1b9e77",
        "SMILES_BPE": "#d95f02",
        "SELFIES": "#7570b3",
        "Group_SELFIES": "#e7298a",
        "Graph": "#66a61e",
    }
    palette = {}
    fallback_colors = [
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc949",
    ]
    for idx, rep in enumerate(representations):
        palette[rep] = base.get(rep, fallback_colors[idx % len(fallback_colors)])
    return palette


def _plot_step1_bpb(step1_df: pd.DataFrame, output_path: Path) -> None:
    if step1_df.empty:
        print("No Step1 data found. Skipping Step1 figure.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    grouped = (
        step1_df.groupby(["representation", "model_size"], dropna=False)["best_val_bpb"]
        .mean()
        .reset_index()
    )
    grouped = grouped.dropna(subset=["representation", "model_size", "best_val_bpb"])
    if grouped.empty:
        print("No valid Step1 BPB data found. Skipping Step1 figure.")
        return

    reps = _ordered_unique(grouped["representation"].astype(str).tolist(), REPRESENTATION_ORDER)
    sizes = _ordered_unique(grouped["model_size"].astype(str).tolist(), MODEL_SIZE_ORDER)
    palette = _method_palette(reps)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel A: BPB by model size
    ax = axes[0]
    x = np.arange(len(sizes))
    for rep in reps:
        rep_df = grouped[grouped["representation"] == rep].copy()
        rep_df["model_size"] = pd.Categorical(rep_df["model_size"], categories=sizes, ordered=True)
        rep_df = rep_df.sort_values("model_size")
        y = rep_df["best_val_bpb"].to_numpy(dtype=float)
        x_rep = [sizes.index(str(s)) for s in rep_df["model_size"].astype(str).tolist()]
        ax.plot(x_rep, y, marker="o", linewidth=2.2, markersize=6, label=rep, color=palette[rep])

    ax.set_title("Step1 Backbone: Best Validation BPB")
    ax.set_xlabel("Model Size")
    ax.set_ylabel("BPB (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(frameon=True, fontsize=9)

    # Panel B: method-level BPB distribution (mean +/- std)
    ax = axes[1]
    method_stats = (
        step1_df.groupby("representation", dropna=False)["best_val_bpb"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    method_stats = method_stats.dropna(subset=["representation", "mean"])
    method_stats["representation"] = pd.Categorical(method_stats["representation"], categories=reps, ordered=True)
    method_stats = method_stats.sort_values("representation")

    x_bar = np.arange(len(method_stats))
    colors = [palette.get(str(r), "#4e79a7") for r in method_stats["representation"].astype(str)]
    yerr = method_stats["std"].fillna(0.0).to_numpy(dtype=float)
    bars = ax.bar(
        x_bar,
        method_stats["mean"].to_numpy(dtype=float),
        yerr=yerr,
        capsize=4,
        color=colors,
        alpha=0.9,
    )
    ax.set_title("Step1 BPB Summary Across Runs")
    ax.set_xlabel("Representation")
    ax.set_ylabel("Best Validation BPB")
    ax.set_xticks(x_bar)
    ax.set_xticklabels(method_stats["representation"].astype(str), rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.35)

    for bar, val in zip(bars, method_stats["mean"].to_numpy(dtype=float)):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Step1 Performance Overview", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote Step1 figure: {output_path}")


def _plot_step2_quality(step2_df: pd.DataFrame, output_path: Path) -> None:
    if step2_df.empty:
        print("No Step2 data found. Skipping Step2 figure.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    metric_cols = ["validity", "uniqueness", "novelty", "avg_diversity", "mean_sa"]
    grouped = (
        step2_df.groupby("representation", dropna=False)[metric_cols + ["valid_per_sec"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    grouped = grouped.dropna(subset=["representation"])
    if grouped.empty:
        print("No valid Step2 grouped data found. Skipping Step2 figure.")
        return

    reps = _ordered_unique(grouped["representation"].astype(str).tolist(), REPRESENTATION_ORDER)
    grouped["representation"] = pd.Categorical(grouped["representation"], categories=reps, ordered=True)
    grouped = grouped.sort_values("representation")
    palette = _method_palette(reps)
    colors = [palette.get(rep, "#4e79a7") for rep in grouped["representation"].astype(str)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Step2 Generation Quality Overview", fontsize=15, fontweight="bold", y=1.02)

    plot_specs = [
        ("validity", "Validity", (0.0, 1.0)),
        ("uniqueness", "Uniqueness", (0.0, 1.0)),
        ("novelty", "Novelty", (0.0, 1.0)),
        ("avg_diversity", "Diversity", (0.0, 1.0)),
        ("mean_sa", "SA Score (lower better)", None),
    ]

    x = np.arange(len(grouped))
    for idx, (col, title, ylim) in enumerate(plot_specs):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        vals = grouped[col].to_numpy(dtype=float)
        bars = ax.bar(x, vals, color=colors, alpha=0.92)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped["representation"].astype(str), rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # Panel 6: SA vs Validity trade-off with throughput bubble size
    ax = axes[1, 2]
    x_sa = grouped["mean_sa"].to_numpy(dtype=float)
    y_valid = grouped["validity"].to_numpy(dtype=float)
    speed = grouped["valid_per_sec"].to_numpy(dtype=float)
    finite_speed = speed[np.isfinite(speed)]
    if finite_speed.size > 0 and float(np.max(finite_speed)) > 0:
        size = 120 + 500 * (speed / float(np.max(finite_speed)))
    else:
        size = np.full_like(y_valid, 180.0)

    for i, rep in enumerate(grouped["representation"].astype(str).tolist()):
        ax.scatter(x_sa[i], y_valid[i], s=size[i], color=palette.get(rep, "#4e79a7"), alpha=0.85, edgecolor="white", linewidth=1.1)
        ax.text(x_sa[i], y_valid[i], f" {rep}", fontsize=8, va="center")

    ax.set_title("Trade-off: SA vs Validity")
    ax.set_xlabel("Mean SA (lower better)")
    ax.set_ylabel("Validity")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote Step2 figure: {output_path}")


def _plot_step3_r2(step3_df: pd.DataFrame, output_path: Path) -> None:
    if step3_df.empty:
        print("No Step3 data found. Skipping Step3 figure.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    df = step3_df.copy()
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.lower().map(_normalize_split)

    grouped = (
        df.groupby(["representation", "property", "split"], dropna=False)["r2"]
        .mean(numeric_only=True)
        .reset_index()
    )
    grouped = grouped.dropna(subset=["representation", "property", "split", "r2"])
    if grouped.empty:
        print("No valid Step3 R2 data found. Skipping Step3 figure.")
        return

    reps = _ordered_unique(grouped["representation"].astype(str).tolist(), REPRESENTATION_ORDER)
    props = _ordered_unique(grouped["property"].astype(str).tolist(), PROPERTY_ORDER)
    splits = [s for s in ["train", "val", "test"] if s in set(grouped["split"].astype(str))]
    if not splits:
        print("No Step3 splits available. Skipping Step3 figure.")
        return

    ncols = len(splits)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols + 1.0, 6.0))
    if ncols == 1:
        axes = [axes]

    all_values = grouped["r2"].to_numpy(dtype=float)
    vmin, vmax = _finite_vmin_vmax(all_values)

    for ax, split in zip(axes, splits):
        sub = grouped[grouped["split"] == split].copy()
        pivot = sub.pivot(index="property", columns="representation", values="r2")
        pivot = pivot.reindex(index=props, columns=reps)

        values = pivot.values.astype(float)
        im = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
        ax.set_title(f"Step3 R2 ({split})")
        ax.set_xticks(np.arange(len(reps)))
        ax.set_xticklabels(reps, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(props)))
        ax.set_yticklabels(props)

        for i in range(len(props)):
            for j in range(len(reps)):
                val = values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="black", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Step3 Property Prediction R2", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote Step3 figure: {output_path}")


def _write_csv(df: pd.DataFrame, columns: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return
    out = ensure_columns(df.copy(), columns)
    out.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Step1/Step2/Step3 metrics and generate summary figures."
    )
    parser.add_argument("--root", type=str, default=".", help="Repo root")
    parser.add_argument(
        "--output",
        type=str,
        default="results/aggregate",
        help="Output directory for aggregate CSVs and figures",
    )
    parser.add_argument(
        "--model_sizes",
        type=str,
        default="all",
        help="Comma-separated sizes: small,medium,large,xl,base (or all).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated method directories. Default: auto-discover AR_Transformer_*.",
    )
    parser.add_argument(
        "--max_methods",
        type=int,
        default=5,
        help="Maximum number of methods when auto-discovering (default: 5).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output).resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    allowed_model_sizes = _parse_model_sizes_arg(args.model_sizes)
    methods = _parse_methods_arg(root, args.methods, args.max_methods)
    print(f"Methods used ({len(methods)}): {', '.join(methods) if methods else '(none)'}")

    step1_df = _aggregate_step1(root, methods, allowed_model_sizes=allowed_model_sizes)
    step2_df = _aggregate_step2(root, methods, allowed_model_sizes=allowed_model_sizes)
    step3_df = _aggregate_step3(root, methods, allowed_model_sizes=allowed_model_sizes)

    # Raw exports
    _write_csv(step1_df, STEP1_COLUMNS, output_dir / "metrics_step1_bpb.csv")
    _write_csv(step2_df, STEP2_COLUMNS, output_dir / "metrics_step2_generation.csv")
    _write_csv(step3_df, STEP3_COLUMNS, output_dir / "metrics_step3_property_r2.csv")

    # Backward-compatible alias for older scripts that expect metrics_generation.csv
    _write_csv(step2_df, STEP2_COLUMNS, output_dir / "metrics_generation.csv")

    # Step1 summaries
    if step1_df.empty:
        step1_summary_size = pd.DataFrame(
            columns=["representation", "model_size", "best_val_bpb", "final_val_bpb", "runs"]
        )
        step1_summary_method = pd.DataFrame(
            columns=["representation", "best_val_bpb_mean", "best_val_bpb_std", "runs"]
        )
    else:
        step1_summary_size = (
            step1_df.groupby(["representation", "model_size"], dropna=False)[["best_val_bpb", "final_val_bpb"]]
            .mean(numeric_only=True)
            .reset_index()
        )
        runs_size = step1_df.groupby(["representation", "model_size"], dropna=False).size().reset_index(name="runs")
        step1_summary_size = step1_summary_size.merge(runs_size, on=["representation", "model_size"], how="left")

        step1_summary_method = (
            step1_df.groupby("representation", dropna=False)["best_val_bpb"]
            .agg(best_val_bpb_mean="mean", best_val_bpb_std="std", runs="count")
            .reset_index()
        )

    step1_summary_size.to_csv(output_dir / "summary_step1_bpb_by_method_size.csv", index=False)
    step1_summary_method.to_csv(output_dir / "summary_step1_bpb_by_method.csv", index=False)

    # Step2 summaries
    step2_metrics = ["validity", "uniqueness", "novelty", "avg_diversity", "mean_sa"]
    if step2_df.empty:
        step2_summary_size = pd.DataFrame(columns=["representation", "model_size", *step2_metrics, "runs"])
        step2_summary_method = pd.DataFrame(columns=["representation", *step2_metrics, "runs"])
    else:
        step2_summary_size = (
            step2_df.groupby(["representation", "model_size"], dropna=False)[step2_metrics]
            .mean(numeric_only=True)
            .reset_index()
        )
        runs_size = step2_df.groupby(["representation", "model_size"], dropna=False).size().reset_index(name="runs")
        step2_summary_size = step2_summary_size.merge(runs_size, on=["representation", "model_size"], how="left")

        step2_summary_method = (
            step2_df.groupby("representation", dropna=False)[step2_metrics]
            .mean(numeric_only=True)
            .reset_index()
        )
        runs_method = step2_df.groupby("representation", dropna=False).size().reset_index(name="runs")
        step2_summary_method = step2_summary_method.merge(runs_method, on="representation", how="left")

    step2_summary_size.to_csv(output_dir / "summary_step2_by_method_size.csv", index=False)
    step2_summary_method.to_csv(output_dir / "summary_step2_by_method.csv", index=False)

    # Step3 summaries (R2-focused)
    if step3_df.empty:
        step3_summary_long = pd.DataFrame(columns=["representation", "property", "split", "r2", "runs"])
        step3_summary_wide = pd.DataFrame(columns=["representation", "property", "r2_train", "r2_val", "r2_test"])
    else:
        step3_summary_long = (
            step3_df.groupby(["representation", "property", "split"], dropna=False)["r2"]
            .mean(numeric_only=True)
            .reset_index()
        )
        runs = step3_df.groupby(["representation", "property", "split"], dropna=False).size().reset_index(name="runs")
        step3_summary_long = step3_summary_long.merge(runs, on=["representation", "property", "split"], how="left")

        step3_pivot = step3_summary_long.pivot_table(
            index=["representation", "property"],
            columns="split",
            values="r2",
            aggfunc="mean",
        ).reset_index()
        step3_pivot.columns.name = None
        rename_map = {}
        for col in step3_pivot.columns:
            if col in {"train", "val", "test"}:
                rename_map[col] = f"r2_{col}"
        step3_summary_wide = step3_pivot.rename(columns=rename_map)

    step3_summary_long.to_csv(output_dir / "summary_step3_r2_by_method_property_split.csv", index=False)
    step3_summary_wide.to_csv(output_dir / "summary_step3_r2_by_method_property.csv", index=False)

    # Figures
    _plot_step1_bpb(step1_df, figures_dir / "step1_bpb_overview.png")
    _plot_step2_quality(step2_df, figures_dir / "step2_quality_overview.png")
    _plot_step3_r2(step3_df, figures_dir / "step3_r2_overview.png")

    print(f"Wrote aggregate metrics to: {output_dir}")
    if allowed_model_sizes is not None:
        print(f"Filtered model sizes: {','.join(sorted(allowed_model_sizes))}")


if __name__ == "__main__":
    main()
