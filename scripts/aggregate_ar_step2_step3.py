#!/usr/bin/env python
"""Aggregate AR Step2/Step3 metrics and generate key figures.

This script is intentionally focused on the 4 AR methods:
  - AR_Transformer_SMILES
  - AR_Transformer_SMILES_BPE
  - AR_Transformer_SELFIES
  - AR_Transformer_Group_SELFIES

Outputs:
  - metrics_step2_generation.csv
  - metrics_step3_property_test.csv
  - summary_step2_by_representation.csv
  - summary_step3_by_rep_property.csv
  - figures/step2_generation_overview.{png,pdf}
  - figures/step3_property_overview.{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.metrics_schema import (
    GENERATION_COLUMNS,
    PROPERTY_COLUMNS,
    ensure_columns,
    infer_model_size,
    parse_method_representation,
)


AR_METHOD_DIRS = [
    "AR_Transformer_SMILES",
    "AR_Transformer_SMILES_BPE",
    "AR_Transformer_SELFIES",
    "AR_Transformer_Group_SELFIES",
]

REPRESENTATION_ORDER = ["SMILES", "SMILES_BPE", "SELFIES", "Group_SELFIES"]
PROPERTY_ORDER = ["Tg", "Tm", "Eg", "Td"]


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _normalize_model_size(value: str) -> str:
    token = value.strip().lower()
    aliases = {"s": "small", "m": "medium", "l": "large", "x": "xl"}
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


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _finite_vmin_vmax(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmin == vmax:
        return vmin - 1e-12, vmax + 1e-12
    return vmin, vmax


def _aggregate_step2(
    root: Path,
    allowed_model_sizes: Optional[Set[str]] = None,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    for method_name in AR_METHOD_DIRS:
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

            df["method"] = info.method
            df["representation"] = info.representation
            df["model_size"] = model_size
            df = ensure_columns(df, GENERATION_COLUMNS)
            rows.append(df[GENERATION_COLUMNS])

    if not rows:
        return pd.DataFrame(columns=GENERATION_COLUMNS)

    out = pd.concat(rows, ignore_index=True)
    numeric_cols = [
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
        "min_sa",
        "max_sa",
        "mean_length",
        "std_length",
        "min_length",
        "max_length",
        "samples_per_sec",
        "valid_per_sec",
    ]
    return _coerce_numeric(out, numeric_cols)


def _aggregate_step3(
    root: Path,
    allowed_model_sizes: Optional[Set[str]] = None,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    for method_name in AR_METHOD_DIRS:
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

                test_metrics_path = metrics_dir / f"{property_name}_test_metrics.csv"
                if not test_metrics_path.exists():
                    continue

                df = _read_csv(test_metrics_path)
                if df.empty:
                    continue

                # Normalize common trainer output schema.
                if "MAE" in df.columns and "mae" not in df.columns:
                    df["mae"] = df["MAE"]
                if "RMSE" in df.columns and "rmse" not in df.columns:
                    df["rmse"] = df["RMSE"]
                if "R2" in df.columns and "r2" not in df.columns:
                    df["r2"] = df["R2"]
                if "R²" in df.columns and "r2" not in df.columns:
                    df["r2"] = df["R²"]

                df["method"] = info.method
                df["representation"] = info.representation
                df["model_size"] = model_size
                if "property" not in df.columns:
                    df["property"] = property_name
                if "split" not in df.columns:
                    df["split"] = "test"

                df = ensure_columns(df, PROPERTY_COLUMNS)
                rows.append(df[PROPERTY_COLUMNS])

    if not rows:
        return pd.DataFrame(columns=PROPERTY_COLUMNS)

    out = pd.concat(rows, ignore_index=True)
    return _coerce_numeric(out, ["mae", "rmse", "r2"])


def _plot_step2_overview(step2_df: pd.DataFrame, output_path: Path) -> None:
    if step2_df.empty:
        print("No Step2 data found. Skipping Step2 figure.")
        return

    metrics = ["validity_two_stars", "novelty", "uniqueness", "avg_diversity", "valid_per_sec"]
    grouped = (
        step2_df.groupby("representation", dropna=False)[metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    grouped = grouped.dropna(subset=["representation"])
    if grouped.empty:
        print("No valid Step2 grouped data found. Skipping Step2 figure.")
        return

    reps = _ordered_unique(grouped["representation"].tolist(), REPRESENTATION_ORDER)
    grouped["representation"] = pd.Categorical(grouped["representation"], categories=reps, ordered=True)
    grouped = grouped.sort_values("representation")

    rep_colors = {
        "SMILES": "#1f77b4",
        "SMILES_BPE": "#aec7e8",
        "SELFIES": "#ff7f0e",
        "Group_SELFIES": "#2ca02c",
    }
    colors = [rep_colors.get(r, "gray") for r in grouped["representation"].astype(str).tolist()]
    x = np.arange(len(grouped))
    width = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Validity (2-star)
    ax = axes[0, 0]
    ax.bar(x, grouped["validity_two_stars"], color=colors)
    ax.set_title("Step2: Validity (Two Attachments)")
    ax.set_ylabel("Validity (2-star)")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["representation"], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)

    # Throughput
    ax = axes[0, 1]
    ax.bar(x, grouped["valid_per_sec"], color=colors)
    ax.set_title("Step2: Throughput")
    ax.set_ylabel("Valid samples / sec")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["representation"], rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    # Novelty + uniqueness
    ax = axes[1, 0]
    ax.bar(x - width / 2, grouped["novelty"], width, label="Novelty", color="#1f77b4")
    ax.bar(x + width / 2, grouped["uniqueness"], width, label="Uniqueness", color="#ff7f0e")
    ax.set_title("Step2: Novelty and Uniqueness")
    ax.set_ylabel("Fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["representation"], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    # Diversity
    ax = axes[1, 1]
    ax.bar(x, grouped["avg_diversity"], color=colors)
    ax.set_title("Step2: Diversity")
    ax.set_ylabel("Average diversity")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["representation"], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote Step2 figure: {output_path}")
    print(f"Wrote Step2 figure: {pdf_path}")


def _plot_step3_overview(step3_df: pd.DataFrame, output_path: Path) -> None:
    if step3_df.empty:
        print("No Step3 data found. Skipping Step3 figure.")
        return

    df = step3_df.copy()
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == "test"].copy()
    if df.empty:
        print("No Step3 test-split data found. Skipping Step3 figure.")
        return

    grouped = (
        df.groupby(["representation", "property"], dropna=False)[["r2", "mae"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    grouped = grouped.dropna(subset=["representation", "property"])
    if grouped.empty:
        print("No valid Step3 grouped data found. Skipping Step3 figure.")
        return

    reps = _ordered_unique(grouped["representation"].tolist(), REPRESENTATION_ORDER)
    props = _ordered_unique(grouped["property"].tolist(), PROPERTY_ORDER)

    r2_pivot = grouped.pivot(index="property", columns="representation", values="r2").reindex(index=props, columns=reps)
    mae_pivot = grouped.pivot(index="property", columns="representation", values="mae").reindex(index=props, columns=reps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # R2 heatmap
    ax = axes[0]
    r2_values = r2_pivot.values.astype(float)
    r2_vmin, r2_vmax = _finite_vmin_vmax(r2_values)
    im = ax.imshow(r2_values, aspect="auto", cmap="viridis", vmin=r2_vmin, vmax=r2_vmax)
    ax.set_title("Step3: Test R2 (higher is better)")
    ax.set_xticks(np.arange(len(reps)))
    ax.set_xticklabels(reps, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(props)))
    ax.set_yticklabels(props)
    for i in range(len(props)):
        for j in range(len(reps)):
            val = r2_values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # MAE heatmap
    ax = axes[1]
    mae_values = mae_pivot.values.astype(float)
    mae_vmin, mae_vmax = _finite_vmin_vmax(mae_values)
    im = ax.imshow(mae_values, aspect="auto", cmap="magma_r", vmin=mae_vmin, vmax=mae_vmax)
    ax.set_title("Step3: Test MAE (lower is better)")
    ax.set_xticks(np.arange(len(reps)))
    ax.set_xticklabels(reps, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(props)))
    ax.set_yticklabels(props)
    for i in range(len(props)):
        for j in range(len(reps)):
            val = mae_values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote Step3 figure: {output_path}")
    print(f"Wrote Step3 figure: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate AR Step2/Step3 metrics for 4 methods and generate figures."
    )
    parser.add_argument("--root", type=str, default=".", help="Repo root")
    parser.add_argument(
        "--output",
        type=str,
        default="results/aggregate_step2_step3",
        help="Output directory for aggregated CSVs and figures",
    )
    parser.add_argument(
        "--model_sizes",
        type=str,
        default="all",
        help="Comma-separated model sizes to include: small,medium,large,xl,base (or all).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output).resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    allowed_model_sizes = _parse_model_sizes_arg(args.model_sizes)

    step2_df = _aggregate_step2(root, allowed_model_sizes=allowed_model_sizes)
    step3_df = _aggregate_step3(root, allowed_model_sizes=allowed_model_sizes)

    # Raw aggregated exports.
    step2_df.to_csv(output_dir / "metrics_step2_generation.csv", index=False)
    step3_df.to_csv(output_dir / "metrics_step3_property_test.csv", index=False)

    # Summaries.
    if step2_df.empty:
        step2_summary = pd.DataFrame(columns=["representation", "validity_two_stars", "novelty", "uniqueness", "avg_diversity", "valid_per_sec"])
    else:
        step2_summary = (
            step2_df.groupby("representation", dropna=False)[
                ["validity_two_stars", "novelty", "uniqueness", "avg_diversity", "valid_per_sec"]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
    step2_summary.to_csv(output_dir / "summary_step2_by_representation.csv", index=False)

    if step3_df.empty:
        step3_summary = pd.DataFrame(columns=["representation", "property", "r2", "mae", "rmse"])
    else:
        step3_filtered = step3_df.copy()
        if "split" in step3_filtered.columns:
            step3_filtered = step3_filtered[step3_filtered["split"].astype(str).str.lower() == "test"].copy()
        step3_summary = (
            step3_filtered.groupby(["representation", "property"], dropna=False)[["r2", "mae", "rmse"]]
            .mean(numeric_only=True)
            .reset_index()
        )
    step3_summary.to_csv(output_dir / "summary_step3_by_rep_property.csv", index=False)

    # Figures.
    _plot_step2_overview(step2_df, figures_dir / "step2_generation_overview.png")
    _plot_step3_overview(step3_df, figures_dir / "step3_property_overview.png")

    print(f"Wrote Step2 aggregate CSV: {output_dir / 'metrics_step2_generation.csv'}")
    print(f"Wrote Step3 aggregate CSV: {output_dir / 'metrics_step3_property_test.csv'}")
    print(f"Wrote Step2 summary: {output_dir / 'summary_step2_by_representation.csv'}")
    print(f"Wrote Step3 summary: {output_dir / 'summary_step3_by_rep_property.csv'}")


if __name__ == "__main__":
    main()
