import tensorflow as tf
from keras import layers

class InstanceNorm(layers.Layer):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def build(self, shape):
        channels = shape[-1]
        self.gamma = self.add_weight(
            shape=[channels], initializer="ones", name="in_gamma"
        )
        self.beta = self.add_weight(
            shape=[channels], initializer="zeros", name="in_beta"
        )

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return self.gamma * (x - mean) / tf.sqrt(var + self.epsilon) + self.beta
    