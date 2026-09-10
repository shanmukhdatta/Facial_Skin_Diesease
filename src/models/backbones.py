try:
    from tensorflow.keras import Model
    from tensorflow.keras.applications import (
        EfficientNetV2B0,
        ResNet50,
        DenseNet121,
        MobileNetV3Large,
    )
except ImportError:
    Model = None
    EfficientNetV2B0 = None
    ResNet50 = None
    DenseNet121 = None
    MobileNetV3Large = None


def get_backbone(model_name: str, img_size: int = 224, weights: str = "imagenet") -> Model:
    """
    Factory function to instantiate pretrained vision backbones.

    Supported models:
      - EfficientNetV2B0
      - ResNet50
      - DenseNet121
      - MobileNetV3Large
      - ViT_B16 (requires vit-keras)
    """
    input_shape = (img_size, img_size, 3)

    if model_name == "EfficientNetV2B0":
        return EfficientNetV2B0(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            include_preprocessing=True,
        )

    elif model_name == "ResNet50":
        return ResNet50(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )

    elif model_name == "DenseNet121":
        return DenseNet121(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )

    elif model_name == "MobileNetV3Large":
        return MobileNetV3Large(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            include_preprocessing=True,
        )

    elif model_name == "ViT_B16":
        try:
            from vit_keras import vit
            return vit.vit_b16(
                image_size=img_size,
                activation="softmax",
                pretrained=True,
                include_top=False,
                pretrained_top=False,
            )
        except ImportError as e:
            raise ImportError(
                "vit-keras is required to use ViT_B16. Install it with: pip install vit-keras"
            ) from e

    else:
        raise ValueError(
            f"Unsupported model_name: {model_name}. Supported: "
            "['EfficientNetV2B0', 'ResNet50', 'DenseNet121', 'MobileNetV3Large', 'ViT_B16']"
        )
