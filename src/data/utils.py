import json
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
def compute_class_weights(y_train: Union[np.ndarray, List[int]], num_classes: int = None) -> Dict[int, float]:
    """Compute balanced class weights for handling class imbalance during training."""
    y_arr = np.asarray(y_train)
    if num_classes is None:
        num_classes = len(np.unique(y_arr))
    n_samples = len(y_arr)

    # Standard balanced weighting formula: n_samples / (n_classes * count_c)
    weights = {}
    for c in range(num_classes):
        count_c = np.sum(y_arr == c)
        if count_c > 0:
            weights[c] = float(n_samples / (num_classes * count_c))
        else:
            weights[c] = 1.0
    return weights


def save_label_map(class_names: List[str], output_path: Union[str, Path]) -> None:
    """Save label mapping dictionary to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_map = {i: cls for i, cls in enumerate(class_names)}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"[OK] Label map saved to: {output_path}")


def load_label_map(input_path: Union[str, Path]) -> Dict[int, str]:
    """Load label mapping dictionary from JSON."""
    input_path = Path(input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}
