try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    LayerBase = layers.Layer
    register_serializable = keras.saving.register_keras_serializable(package="custom_preprocessing")
except (ImportError, AttributeError):
    tf = None
    keras = None
    layers = None
    LayerBase = object
    def register_serializable(cls):
        return cls

from src.models.losses import FocalLoss


@register_serializable
class ResNet50Preprocess(LayerBase):
    """Caffe-style ResNet50 preprocessing. Expects raw [0, 255] RGB input."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return keras.applications.resnet50.preprocess_input(x)

    def get_config(self):
        return super().get_config()


@register_serializable
class DenseNet121Preprocess(LayerBase):
    """Torch-style DenseNet121 preprocessing. Expects raw [0, 255] RGB input."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return keras.applications.densenet.preprocess_input(x)

    def get_config(self):
        return super().get_config()


@register_serializable
class ViTPreprocess(LayerBase):
    """Rescales [0, 255] image pixels to [-1, 1] for Vision Transformer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return (tf.cast(x, tf.float32) / 127.5) - 1.0

    def get_config(self):
        return super().get_config()


@register_serializable
class ExtractCLSToken(LayerBase):
    """Extracts the first CLS token vector [:, 0, :] from sequence representations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x[:, 0, :]

    def get_config(self):
        return super().get_config()


PREPROCESS_LAYERS = {
    "ResNet50": ResNet50Preprocess,
    "DenseNet121": DenseNet121Preprocess,
    "MobileNetV3Large": None,
    "EfficientNetV2B0": None,
    "ViT_B16": ViTPreprocess,
}

CUSTOM_OBJECTS = {
    "ResNet50Preprocess": ResNet50Preprocess,
    "DenseNet121Preprocess": DenseNet121Preprocess,
    "ViTPreprocess": ViTPreprocess,
    "ExtractCLSToken": ExtractCLSToken,
    "FocalLoss": FocalLoss,
}


def get_preprocess_layer(model_name: str):
    """Return a fresh preprocessing layer instance for model_name, or None."""
    layer_cls = PREPROCESS_LAYERS.get(model_name)
    return layer_cls(name="backbone_preprocess") if layer_cls is not None else None
