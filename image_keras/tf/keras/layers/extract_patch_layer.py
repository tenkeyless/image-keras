from image_keras.tf.utils.images import tf_extract_patches
from tensorflow.keras.layers import Layer


class ExtractPatchLayer(Layer):
    def __init__(self, k_size: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.k_size = k_size

    def get_config(self):
        config = super(ExtractPatchLayer, self).get_config()
        config.update({"k_size": self.k_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        return tf_extract_patches(inputs, self.k_size)