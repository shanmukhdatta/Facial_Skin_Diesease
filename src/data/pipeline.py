from __future__ import annotations
import io
import numpy as np
from PIL import Image as PILImage, ImageOps
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    tf = None
    keras = None
    layers = None


def build_augmentation_layer() -> keras.Sequential:
    """Build mild data augmentation layer preserving medical skin lesion textures."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.10),
        layers.RandomBrightness(0.08),
        layers.RandomTranslation(0.05, 0.05),
    ], name="augmentation")


def _pil_load(path_bytes: bytes, raw_bytes: bytes, img_size: int = 224) -> np.ndarray:
    """
    Load any image format (JPEG/PNG/BMP/TIFF/WebP) with PIL and return a float32 RGB array in [0, 255].
    Backbone-specific normalizations happen inside the model itself.
    Raises RuntimeError with the offending file path if corrupted or unreadable.
    """
    try:
        with PILImage.open(io.BytesIO(raw_bytes)) as img:
            img = img.convert("RGB")
            # Resize without stretching facial geometry; reflect-pad the shorter axis.
            contained = ImageOps.contain(img, (img_size, img_size), method=PILImage.Resampling.BILINEAR)
            arr = np.asarray(contained, dtype=np.uint8)
            h, w = arr.shape[:2]
            top, left = (img_size - h) // 2, (img_size - w) // 2
            pad = ((top, img_size - h - top), (left, img_size - w - left), (0, 0))
            arr = np.pad(arr, pad, mode="reflect" if min(h, w) > 1 else "edge")
            return arr.astype(np.float32)
    except Exception as e:
        file_path = path_bytes.decode("utf-8", errors="replace") if isinstance(path_bytes, bytes) else str(path_bytes)
        raise RuntimeError(
            f"[ERROR] Corrupted or unreadable image encountered during preprocessing: '{file_path}' ({e})"
        ) from e


def preprocess_image(path, label, augment: bool, img_size: int, augmentation_layer=None):
    """Bridge PIL loader into the TensorFlow execution graph."""
    raw = tf.io.read_file(path)
    image = tf.py_function(
        func=lambda p, r: _pil_load(p.numpy(), r.numpy(), img_size),
        inp=[path, raw],
        Tout=tf.float32,
    )
    image.set_shape([img_size, img_size, 3])

    if augment and augmentation_layer is not None:
        image = augmentation_layer(image, training=True)

    image = tf.clip_by_value(image, 0.0, 255.0)
    label = tf.cast(label, tf.int32)
    return image, label



def make_dataset(
    paths,
    labels,
    img_size: int = 224,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build high-performance tf.data pipeline with batching and prefetching."""
    ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(paths, dtype=tf.string),
        tf.constant(labels, dtype=tf.int32),
    ))

    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    aug_layer = build_augmentation_layer() if augment else None

    ds = ds.map(
        lambda p, l: preprocess_image(p, l, augment, img_size, aug_layer),
        num_parallel_calls=4,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
