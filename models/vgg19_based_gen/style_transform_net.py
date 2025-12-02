from typing import Literal, Tuple
import keras
import tensorflow as tf
from .generator import GeneratorNetwork
from .feature_extractor import VGGFeatureExtractor
from .utils import preprocess,  gram_matrix


class StyleTransferNet(keras.Model):
    def __init__(
        self,
        input_shape: Tuple,
        style_image: tf.Tensor,
        alpha: float,
        beta: float,
        use_reflection_padding = False
    ):
        super().__init__()

        self.alpha = tf.constant(alpha, dtype=tf.float32, name="alpha")
        self.beta = tf.constant(beta, dtype=tf.float32, name="beta")

        self.generator = GeneratorNetwork(input_shape)
        self.vgg_feature_extractor = VGGFeatureExtractor()

        self.num_style_layers = 4
        preprocessed_style = preprocess(style_image)

        style_features = self.vgg_feature_extractor(preprocessed_style)
        self.style_targets = [
            gram_matrix(f) for f in style_features[: self.num_style_layers]
        ]

        self.content_loss_fn = lambda g, t: tf.reduce_mean(tf.square(g - t))

    def compile(self, optimizer, loss=None, metrics=None, loss_weights=None, **kwargs):
        super().compile(optimizer=optimizer, metrics=metrics, **kwargs)

    @tf.function
    def compute_loss(self, content_image: tf.Tensor) -> tf.Tensor:
        generated_image = self.generator(content_image)

        p_content = preprocess(content_image)
        p_generated = preprocess(generated_image)

        content_features = self.vgg_feature_extractor(p_content)
        generated_features = self.vgg_feature_extractor(p_generated)

        content_target = content_features[self.num_style_layers]
        generated_content = generated_features[self.num_style_layers]
        c_loss = self.content_loss_fn(generated_content, content_target)

        style_outputs = [
            gram_matrix(f) for f in generated_features[: self.num_style_layers]
        ]

        s_loss = tf.constant(0.0, dtype=tf.float32)
        style_layer_channels = [64.0, 128.0, 256.0, 512.0]

        for output, target, chans in zip(
            style_outputs, self.style_targets, style_layer_channels
        ):
            layer_loss = tf.reduce_mean(tf.square(output - target))
            s_loss += layer_loss / (chans * chans)

        total_loss = (self.alpha * c_loss) + (self.beta * s_loss)

        return total_loss, generated_image, c_loss, s_loss

    @tf.function
    def train_step(self, data) -> dict[Literal["loss", "content_loss", "style_loss"], float]:
        content_image = data

        with tf.GradientTape() as tape:
            total_loss, generated_image, c_loss, s_loss = self.compute_loss(content_image)

        trainable_vars = self.generator.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": total_loss, 'content_loss': c_loss, 'style_loss': s_loss}

    @tf.function
    def call(self, inputs):
        return self.generator(inputs)
