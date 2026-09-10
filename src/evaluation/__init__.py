"""Evaluation, visualization, metrics, and ensemble voting."""
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualization import (
    plot_class_distribution,
    plot_sample_images,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
    plot_per_class_metrics,
    plot_comparison_dashboard,
    plot_radar_chart,
)
from src.evaluation.ensemble import WeightedEnsemble

__all__ = [
    "evaluate_model",
    "plot_class_distribution",
    "plot_sample_images",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_training_history",
    "plot_per_class_metrics",
    "plot_comparison_dashboard",
    "plot_radar_chart",
    "WeightedEnsemble",
]
