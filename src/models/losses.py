try:
    import tensorflow as tf
    from tensorflow import keras
    LossBase = keras.losses.Loss
    register_loss = keras.saving.register_keras_serializable(package="custom_losses")
except (ImportError, AttributeError):
    tf = None
    keras = None
    LossBase = object
    def register_loss(cls):
        return cls


@register_loss
class FocalLoss(LossBase):
    """
    Multi-class Focal Loss for addressing class imbalance and focusing on hard examples.
    Reference: Lin et al. 2017 (https://arxiv.org/abs/1708.02002).
    Expects y_pred to be class probabilities (e.g. from Softmax).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, name: str = "focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Clip values to prevent NaN when calculating log
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true_oh = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1], dtype=tf.float32)

        pt = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        ce = -tf.math.log(pt)
        focal = self.alpha * tf.pow(1.0 - pt, self.gamma) * ce
        return tf.reduce_mean(focal)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg
