import keras
from keras import layers
from .instance_norm import InstanceNorm
from .res_block import ResidualBlock


def GeneratorNetwork(input_shape):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 9, strides=1, padding="same")(inputs)
    x = InstanceNorm()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = InstanceNorm()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = InstanceNorm()(x)
    x = layers.ReLU()(x)

    for _ in range(5):
        x = ResidualBlock(128)(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = InstanceNorm()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = InstanceNorm()(x)
    x = layers.ReLU()(x)

    outputs = layers.Conv2D(3, 9, activation="tanh", padding="same")(x)

    outputs = (outputs + 1) / 2.0

    return keras.Model(inputs=inputs, outputs=outputs, name="generator")
