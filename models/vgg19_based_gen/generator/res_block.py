import tensorflow as tf
from keras import layers
from .generator import InstanceNorm
from .reflection_padding import ReflectionPadding2D


class ResidualBlock(layers.Layer):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        use_reflection_padding = False,
        **kwargs,
    ):
        super(ResidualBlock, self).__init__(**kwargs)
        
        self.use_reflection_padding = use_reflection_padding

        self.channels = channels
        self.kernel_size = kernel_size

        self.pad1 = ReflectionPadding2D(padding=(1, 1))
        self.conv1 = layers.Conv2D(
            self.channels, self.kernel_size, padding="valid", use_bias=True
        )
        self.norm1 = InstanceNorm()
        self.relu = layers.ReLU()

        self.pad2 = ReflectionPadding2D(padding=(1, 1))
        self.conv2 = layers.Conv2D(
            self.channels, self.kernel_size, padding="valid", use_bias=True
        )
        self.norm2 = InstanceNorm()

    def call(self, inputs):
        if self.use_reflection_padding:
            x = self.pad1(inputs)
            x = self.conv1(x)
        else:
            x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.relu(x)

        if self.use_reflection_padding:
            x = self.pad2(x)
        x = self.conv2(x)
        x = self.norm2(x)

        return layers.add([inputs, x])
