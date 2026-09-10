try:
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.regularizers import l2
except ImportError:
    keras = None
    layers = None
    Model = None
    l2 = None

from src.models.custom_layers import ExtractCLSToken, ViTPreprocess


def build_classification_head(
    base_model: Model,
    num_classes: int,
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4,
    img_size: int = 224,
    name: str = "model",
    preprocess_layer=None,
) -> Model:
    """
    Attach classification head:
    GAP -> BN -> Dense(512) -> BN -> Dropout -> Dense(256) -> BN -> Dropout -> Softmax.

    preprocess_layer, if specified, is applied to the raw [0, 255] input before base_model.
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input_image")
    x = inputs
    if preprocess_layer is not None:
        x = preprocess_layer(x)

    x = base_model(x)

    if len(x.shape) == 4:
        x = layers.GlobalAveragePooling2D(name="gap")(x)
    elif len(x.shape) == 3:
        x = ExtractCLSToken(name="cls_token")(x)

    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=l2(l2_reg), name="dense_512")(x)
    x = layers.BatchNormalization(name="bn_512")(x)
    x = layers.Dropout(dropout_rate, name="drop_512")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=l2(l2_reg), name="dense_256")(x)
    x = layers.BatchNormalization(name="bn_256")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop_256")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs, outputs, name=name)


def build_vit_model(
    vit_base_model: Model,
    num_classes: int,
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4,
    img_size: int = 224,
    name: str = "ViT_B16",
) -> Model:
    """Build classification model around a Vision Transformer (ViT-B16) base."""
    vit_input = keras.Input(shape=(img_size, img_size, 3), name="vit_input")
    # ViT expects pixels scaled to [-1, 1]
    vit_preprocess = ViTPreprocess(name="vit_preprocess")(vit_input)
    vit_out = vit_base_model(vit_preprocess)

    cls_token = ExtractCLSToken(name="cls_token")(vit_out)
    x = layers.BatchNormalization(name="bn_vit")(cls_token)
    x = layers.Dense(512, activation="relu", kernel_regularizer=l2(l2_reg), name="dense_512")(x)
    x = layers.BatchNormalization(name="bn_512")(x)
    x = layers.Dropout(dropout_rate, name="drop_512")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=l2(l2_reg), name="dense_256")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop_256")(x)
    vit_output = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(vit_input, vit_output, name=name)


def unfreeze_layers(base_model: Model, n_layers: int) -> int:
    """Unfreeze the last n_layers of the base model."""
    base_model.trainable = True
    total_layers = len(base_model.layers)
    if n_layers <= 0 or n_layers >= total_layers:
        for layer in base_model.layers:
            layer.trainable = True
    else:
        for layer in base_model.layers[:-n_layers]:
            layer.trainable = False
        for layer in base_model.layers[-n_layers:]:
            layer.trainable = True
    n_trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"  -> {n_trainable}/{total_layers} backbone layers trainable")
    return n_trainable


def freeze_all(base_model: Model) -> None:
    """Freeze all layers in the base model."""
    base_model.trainable = False
    print("  -> Backbone fully frozen")
