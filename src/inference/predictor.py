from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    tf = None
    keras = None

from src.data.utils import load_label_map
from src.models.custom_layers import CUSTOM_OBJECTS
from src.models.losses import FocalLoss


class DiseasePredictor:
    """Inference engine for Face Skin Disease Classification models."""

    def __init__(
        self,
        model_path: Union[str, Path],
        label_map: Union[Dict[int, str], str, Path],
        img_size: int = 224,
    ):
        self.model_path = Path(model_path)
        self.img_size = img_size

        if isinstance(label_map, (str, Path)):
            self.label_map = load_label_map(label_map)
        else:
            self.label_map = label_map

        if keras is None:
            raise ImportError(
                "TensorFlow is required to load models and run inference. "
                "Install dependencies: pip install -r requirements.txt"
            )

        print(f"Loading model checkpoint: {self.model_path} ...")
        self.model = keras.models.load_model(
            str(self.model_path),
            custom_objects={**CUSTOM_OBJECTS, "FocalLoss": FocalLoss},
        )
        print("[OK] Model loaded successfully.")

    def predict_image(self, image_path: Union[str, Path]) -> Tuple[str, float, np.ndarray]:
        """
        Run inference on a single image file.
        Passes raw [0, 255] float32 image to the model (matching internal model normalization).
        """
        image_path = Path(image_path)
        assert image_path.exists(), f"Image not found: {image_path}"

        with PILImage.open(image_path) as pil_img:
            pil_img = pil_img.convert("RGB").resize((self.img_size, self.img_size), PILImage.BILINEAR)
            img_np = np.array(pil_img, dtype=np.float32)

        img = tf.constant(img_np)
        img = tf.clip_by_value(img, 0.0, 255.0)
        img = tf.expand_dims(img, 0)  # batch dimension (1, H, W, 3)

        probs = self.model.predict(img, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = self.label_map[pred_idx]
        confidence = float(probs[pred_idx])

        return pred_class, confidence, probs

    def plot_prediction(
        self,
        image_path: Union[str, Path],
        pred_class: str,
        confidence: float,
        probs: np.ndarray,
        model_name: str = "Classifier",
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot the input image alongside predicted probability distribution."""
        with PILImage.open(image_path) as pil_img:
            img_disp = np.array(pil_img.convert("RGB").resize((self.img_size, self.img_size), PILImage.BILINEAR))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img_disp)
        axes[0].set_title(f"Input: {Path(image_path).name}\nPred: {pred_class} ({confidence:.2%})", fontsize=11)
        axes[0].axis("off")

        y_pos = np.arange(len(self.label_map))
        classes = [self.label_map[i] for i in range(len(self.label_map))]
        colors = ["#2ecc71" if c == pred_class else "#95a5a6" for c in classes]

        axes[1].barh(y_pos, probs, color=colors, edgecolor="none", height=0.55)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(classes, fontsize=9)
        axes[1].set_xlim(0, 1.05)
        axes[1].set_xlabel("Confidence Score", fontsize=9)
        axes[1].set_title(f"{model_name} Prediction Distribution", fontsize=11)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        for i, p in enumerate(probs):
            axes[1].text(p + 0.02, i, f"{p:.1%}", va="center", fontsize=8, color="#2c3e50")

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[INFO] Prediction plot saved to {save_path}")
        plt.close()
