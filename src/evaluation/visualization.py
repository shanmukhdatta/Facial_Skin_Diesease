from pathlib import Path
from typing import Any, Dict, List, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
import tensorflow as tf


def plot_class_distribution(
    train_df_raw: pd.DataFrame, df_balanced: pd.DataFrame, save_dir: Union[str, Path]
) -> None:
    """Plot class distributions before and after balancing (Cell 7)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    train_df_raw.groupby("class")["path"].count().plot(
        kind="bar", ax=axes[0], color="steelblue", edgecolor="k"
    )
    axes[0].set_title("Train Before Balancing", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Image Count")
    axes[0].tick_params(axis="x", rotation=45)

    df_balanced.groupby("class")["path"].count().plot(
        kind="bar", ax=axes[1], color="seagreen", edgecolor="k"
    )
    axes[1].set_title("Train After Balancing", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Image Count")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_dir / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_sample_images(
    dataset: tf.data.Dataset, class_names: List[str], save_dir: Union[str, Path]
) -> None:
    """Plot grid of sample augmented training images (Cell 8 / Cell 16)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sample_images, sample_labels = next(iter(dataset))
    n = min(12, len(sample_images))
    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    for i, ax in enumerate(axes.flat):
        if i < n:
            img = sample_images[i].numpy()
            img = np.clip(img / 255.0, 0.0, 1.0)  # display-only rescale
            ax.imshow(img)
            ax.set_title(class_names[sample_labels[i].numpy()], fontsize=8)
        ax.axis("off")

    plt.suptitle("Sample Augmented Training Images", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "sample_images.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_dir: Union[str, Path],
) -> None:
    """Plot count and normalized confusion matrices (Cell 12)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title(f"{model_name} - Confusion Matrix (Counts)", fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title(f"{model_name} - Confusion Matrix (Normalized)", fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    class_names: List[str],
    model_name: str,
    save_dir: Union[str, Path],
) -> None:
    """Plot multi-class One-vs-Rest ROC curves (Cell 12)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, (cls, col) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{cls} (AUC={roc_auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} - ROC Curves (One-vs-Rest)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_history(
    history_list: List[Dict[str, List[float]]],
    model_name: str,
    save_dir: Union[str, Path],
) -> None:
    """Concatenate phase histories and plot training & validation accuracy and loss curves (Cell 12)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    merged = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    for h in history_list:
        for k in merged:
            if k in h:
                merged[k].extend(h[k])

    epochs = range(1, len(merged["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, merged["accuracy"], label="Train Acc", lw=2)
    axes[0].plot(epochs, merged["val_accuracy"], label="Val Acc", lw=2, linestyle="--")
    axes[0].set_title(f"{model_name} - Accuracy", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, merged["loss"], label="Train Loss", lw=2)
    axes[1].plot(epochs, merged["val_loss"], label="Val Loss", lw=2, linestyle="--")
    axes[1].set_title(f"{model_name} - Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"{model_name} Training Curves", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_metrics(
    report_dict: Dict[str, Any],
    class_names: List[str],
    model_name: str,
    save_dir: Union[str, Path],
) -> None:
    """Plot per-class precision, recall, and F1-score bar chart (Cell 12)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_list = ["precision", "recall", "f1-score"]
    data = {m: [report_dict[c][m] for c in class_names] for m in metrics_list}
    df_m = pd.DataFrame(data, index=class_names)

    ax = df_m.plot(
        kind="bar",
        figsize=(12, 5),
        width=0.7,
        edgecolor="k",
        color=["#4C72B0", "#55A868", "#C44E52"],
    )
    ax.set_title(f"{model_name} - Per-Class Metrics", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    for bar in ax.patches:
        ax.annotate(
            f"{bar.get_height():.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_per_class_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison_dashboard(df_compare: pd.DataFrame, save_dir: Union[str, Path]) -> None:
    """Plot side-by-side horizontal bar charts comparing all trained models (Cell 20)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = ["Accuracy", "F1 (Macro)", "AUC-ROC", "Cohen's Kappa", "MCC"]
    metrics_present = [m for m in metrics_to_plot if m in df_compare.columns]
    palette = sns.color_palette("tab10", n_colors=len(df_compare))

    fig, axes = plt.subplots(1, len(metrics_present), figsize=(5 * len(metrics_present), 6))
    if len(metrics_present) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_present):
        bars = ax.barh(df_compare["Model"], df_compare[metric], color=palette, edgecolor="k", linewidth=0.6)
        ax.set_title(metric, fontweight="bold", fontsize=12)
        val_min = df_compare[metric].dropna().min() if not df_compare[metric].dropna().empty else 0.0
        ax.set_xlim(max(0, val_min - 0.05), 1.0)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, df_compare[metric]):
            if not np.isnan(val):
                ax.text(
                    val + 0.003,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.suptitle("Model Performance Comparison - Face Skin Disease Classification", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_dir / "model_comparison_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_radar_chart(df_compare: pd.DataFrame, save_dir: Union[str, Path]) -> None:
    """Generate multi-axis radar chart for comparing model profiles (Cell 20)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    radar_metrics = ["Accuracy", "F1 (Macro)", "F1 (Weighted)", "AUC-ROC", "Cohen's Kappa"]
    radar_metrics = [m for m in radar_metrics if m in df_compare.columns]
    if len(radar_metrics) < 3:
        return

    n = len(radar_metrics)
    angles = [i / float(n) * 2 * np.pi for i in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for _, row in df_compare.iterrows():
        values = [row[m] if not np.isnan(row[m]) else 0.0 for m in radar_metrics]
        values += [values[0]]
        ax.plot(angles, values, "o-", linewidth=2, label=row["Model"])
        ax.fill(angles, values, alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Radar Chart - Model Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "model_radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
