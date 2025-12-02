import tensorflow as tf
from keras import layers
from .generator import InstanceNorm

class ResidualBlock(layers.Layer):
    def __init__(self, channels: int, kernel_size: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size

        self.conv1 = layers.Conv2D(
            self.channels, self.kernel_size, padding="same", use_bias=True
        )
        self.norm1 = InstanceNorm()
        self.relu = layers.ReLU()

        self.conv2 = layers.Conv2D(
            self.channels, self.kernel_size, padding="same", use_bias=True
        )
        self.norm2 = InstanceNorm()

    def call(self, inputs):
        y = self.conv1(inputs)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)

        return layers.add([inputs, y])