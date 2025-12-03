import keras
from keras import layers
from .instance_norm import InstanceNorm
from .res_block import ResidualBlock
from .reflection_padding import ReflectionPadding2D


def GeneratorNetwork(input_shape, use_upsampling=False, use_reflection_padding=False):
    inputs = keras.Input(shape=input_shape)

    if use_reflection_padding:
        x = ReflectionPadding2D(padding=(4, 4))(inputs)
        x = layers.Conv2D(32, 9, strides=1, padding="valid")(x)
    else:
        x = layers.Conv2D(32, 9, strides=1, padding="same")(inputs)
    x = InstanceNorm()(x)
    x = layers.ReLU()(x)

    if use_reflection_padding:
        x = ReflectionPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(64, 3, strides=2, padding="valid")(x)
    else:
        x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = InstanceNorm()(x)
    x = layers.ReLU()(x)

    if use_reflection_padding:
        x = ReflectionPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(128, 3, strides=2, padding="valid")(x)
    else:
        x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = InstanceNorm()(x)
    x = layers.ReLU()(x)

    for _ in range(5):
        x = ResidualBlock(128, use_reflection_padding=use_reflection_padding)(x)

    if use_upsampling:
        x = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x)
        x = ReflectionPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(64, 3, strides=1, padding="valid")(x)
    else:
        x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = InstanceNorm()(x)
    x = layers.ReLU()(x)

    if use_upsampling:
        x = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x)
        x = ReflectionPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(32, 3, strides=1, padding="valid")(x)
    else:
        x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = InstanceNorm()(x)
    x = layers.ReLU()(x)

    if use_reflection_padding:
        x = ReflectionPadding2D(padding=(4, 4))(x)
        outputs = layers.Conv2D(3, 9, activation="tanh", padding="valid")(x)
    else:
        outputs = layers.Conv2D(3, 9, activation="tanh", padding="same")(x)

    outputs = (outputs + 1) / 2.0

    return keras.Model(inputs=inputs, outputs=outputs, name="generator")
