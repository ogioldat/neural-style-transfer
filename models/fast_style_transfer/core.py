from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = 256


def preprocess(image: tf.Tensor) -> tf.Tensor:
    """Normalize tensors for VGG19 feature extraction."""
    image = tf.image.convert_image_dtype(image, tf.float32)
    return keras.applications.vgg19.preprocess_input(image * 255.0)


def gram_matrix(tensor: tf.Tensor) -> tf.Tensor:
    """Compute Gram matrix used in the style loss."""
    batch, h, w, c = tf.unstack(tf.shape(tensor))
    features = tf.reshape(tensor, [batch, h * w, c])
    gram = tf.matmul(features, features, transpose_a=True)
    return gram / tf.cast(h * w * c, tf.float32)


def load_and_preprocess(path: str, img_size: int = IMG_SIZE) -> tf.Tensor:
    """Read an image file and convert it to a float32 tensor in [0, 1]."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (img_size, img_size))
    return img


class StyleTransfer(keras.Model):
    """Single-style optimization baseline."""

    def __init__(
        self,
        *,
        content_image: tf.Tensor,
        style_image: tf.Tensor,
        content_weight: float = 1e-1,
        style_weight: float = 1e4,
        tvl_weight: float = 1e1,
    ) -> None:
        super().__init__()

        self.alpha = tf.constant(content_weight, dtype=tf.float32)
        self.beta = tf.constant(style_weight, dtype=tf.float32)
        self.gamma = tf.constant(tvl_weight, dtype=tf.float32)

        self.style_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
        ]
        self.content_layers = ["block5_conv2"]

        vgg = keras.applications.VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False
        outputs = [
            vgg.get_layer(name).output
            for name in self.style_layers + self.content_layers
        ]
        self.feature_extractor = keras.Model(vgg.input, outputs)

        initial_image = content_image[tf.newaxis, ...]
        self.generated_image = tf.Variable(
            initial_image, dtype=tf.float32, name="generated_image"
        )
        self.clip = lambda x: tf.clip_by_value(
            x, clip_value_min=0.0, clip_value_max=1.0
        )

        preprocessed_style = preprocess(style_image)[tf.newaxis, ...]
        preprocessed_content = preprocess(content_image)[tf.newaxis, ...]

        target_style_content_features = self.feature_extractor(preprocessed_style)
        target_content_content_features = self.feature_extractor(preprocessed_content)

        num_style_layers = len(self.style_layers)

        self.style_targets = [
            gram_matrix(f) for f in target_style_content_features[:num_style_layers]
        ]

        self.content_targets = target_content_content_features[num_style_layers:]

        self.loss_fn = tf.losses.MeanSquaredError()

    def compile(self, optimizer, loss=None, metrics=None, loss_weights=None, **kwargs):
        # Skip keras.Model.compile defaults we don't use
        super().compile(optimizer=optimizer, metrics=metrics, **kwargs)

    def content_loss(self, generated_features: Sequence[tf.Tensor]) -> tf.Tensor:
        """L2 distance between generated and target content features."""
        total_content_loss = tf.constant(0.0)
        for generated, target in zip(generated_features, self.content_targets):
            total_content_loss += self.loss_fn(generated, target)
        return total_content_loss

    def style_loss(self, generated_style_features: Sequence[tf.Tensor]) -> tf.Tensor:
        """L2 distance between Gram matrices of generated/style targets."""
        total_style_loss = tf.constant(0.0)
        for generated_feature, style_target in zip(
            generated_style_features, self.style_targets
        ):
            generated_gram = gram_matrix(generated_feature)
            total_style_loss += self.loss_fn(style_target, generated_gram)
        return total_style_loss

    def total_variation_loss(self, img: tf.Tensor) -> tf.Tensor:
        horizontal_diff = tf.abs(img[:, 1:, :-1, :] - img[:, :-1, :-1, :])
        vertical_diff = tf.abs(img[:, :-1, 1:, :] - img[:, :-1, :-1, :])
        return tf.reduce_sum(horizontal_diff + vertical_diff)

    @tf.function
    def compute_loss(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        processed_generated = preprocess(self.generated_image)
        all_features = self.feature_extractor(processed_generated)

        num_style_layers = len(self.style_layers)
        generated_style_features = all_features[:num_style_layers]
        generated_content_features = all_features[num_style_layers:]

        c_loss = self.content_loss(generated_content_features)
        s_loss = self.style_loss(generated_style_features)
        tv_loss = self.total_variation_loss(self.generated_image)

        total_loss = (self.alpha * c_loss) + (self.beta * s_loss) + (self.gamma * tv_loss)
        return total_loss, self.alpha * c_loss, self.beta * s_loss, self.gamma * tv_loss

    @tf.function
    def train_step(self, data):
        del data
        with tf.GradientTape() as tape:
            total_loss, c_loss, s_loss, _ = self.compute_loss()

        gradients = tape.gradient(total_loss, [self.generated_image])
        self.optimizer.apply_gradients(zip(gradients, [self.generated_image]))
        self.generated_image.assign(self.clip(self.generated_image))

        return total_loss, c_loss, s_loss

    @tf.function
    def call(self, inputs=None):
        del inputs
        return self.generated_image


def train_style_transfer(
    model: StyleTransfer,
    *,
    epochs: int = 1000,
    lr: float = 1e-2,
    verbose_every: int = 100,
) -> np.ndarray:
    """Run the optimization loop and return loss tuples as a NumPy array."""
    model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=lr))

    history: List[Tuple[float, float, float]] = []
    for step in range(epochs):
        total_loss, content_loss, style_loss = model.train_step(None)
        history.append(
            (
                float(total_loss.numpy()),
                float(content_loss.numpy()),
                float(style_loss.numpy()),
            )
        )

        if verbose_every and (step % verbose_every == 0 or step == epochs - 1):
            print(f"Step {step}: Total Loss {total_loss.numpy():.2f}")
    return np.array(history)


def plot_training_curves(history: np.ndarray) -> None:
    """Plot total/content/style losses for a single run."""
    total_loss = history[:, 0]
    content_loss = history[:, 1]
    style_loss = history[:, 2]
    steps = np.arange(len(total_loss))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, total_loss, label="Total Loss", linewidth=2)
    plt.plot(steps, content_loss, label="Content Loss")
    plt.plot(steps, style_loss, label="Style Loss")
    plt.yscale("log")
    plt.xlabel("Training Step")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Losses")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()


def show_style_transfer_triplet(
    style_img: np.ndarray,
    content_img: np.ndarray,
    generated_img: np.ndarray,
) -> None:
    """Display style/content inputs next to the stylized output."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(style_img)
    axes[0].set_title("Style Image")
    axes[0].axis("off")

    axes[1].imshow(content_img)
    axes[1].set_title("Content Image")
    axes[1].axis("off")

    axes[2].imshow(generated_img)
    axes[2].set_title("Stylized Result")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    content_weight: float
    style_weight: float
    tvl_weight: float = 10.0
    epochs: int = 2000
    lr: float = 1e-2


@dataclass
class ExperimentResult:
    name: str
    alpha: float
    beta: float
    gamma: float
    loss_data: np.ndarray
    styled_image: np.ndarray


def run_experiment(
    style_image: tf.Tensor,
    content_image: tf.Tensor,
    config: ExperimentConfig,
    *,
    verbose_every: int = 200,
) -> ExperimentResult:
    """Train a StyleTransfer model for a single hyperparameter config."""
    print(
        f"\n--- Starting Config {config.name}: "
        f"alpha={config.content_weight}, beta={config.style_weight}, "
        f"gamma={config.tvl_weight} ---"
    )

    model = StyleTransfer(
        style_image=style_image,
        content_image=content_image,
        content_weight=config.content_weight,
        style_weight=config.style_weight,
        tvl_weight=config.tvl_weight,
    )

    history = train_style_transfer(
        model,
        epochs=config.epochs,
        lr=config.lr,
        verbose_every=verbose_every,
    )

    styled_image_tensor = model.call()
    styled_image = tf.squeeze(styled_image_tensor, axis=0).numpy()

    return ExperimentResult(
        name=config.name,
        alpha=config.content_weight,
        beta=config.style_weight,
        gamma=config.tvl_weight,
        loss_data=history,
        styled_image=styled_image,
    )


def run_experiments(
    style_image: tf.Tensor,
    content_image: tf.Tensor,
    configs: Iterable[ExperimentConfig],
    *,
    verbose_every: int = 200,
) -> List[ExperimentResult]:
    """Run multiple experiments sequentially."""
    return [
        run_experiment(
            style_image,
            content_image,
            config,
            verbose_every=verbose_every,
        )
        for config in configs
    ]


def plot_loss_mosaic(experiments: Sequence[ExperimentResult]) -> None:
    """Plot the loss curve for every experiment on one chart."""
    plt.figure(figsize=(10, 6))

    for exp in experiments:
        total_loss = exp.loss_data[:, 0]
        steps = np.arange(len(total_loss))
        label = (
            f"{exp.name}: "
            f"$\\alpha={exp.alpha:.0e}, \\beta={exp.beta:.0e}, "
            f"\\gamma={exp.gamma:.0e}$"
        )
        plt.plot(steps, total_loss, label=label, linewidth=2)

    plt.title("Total Loss Comparison Across Hyperparameter Configurations")
    plt.xlabel("Training Step (Iteration)")
    plt.ylabel("Total Weighted Loss (Log Scale)")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()


def plot_image_mosaic(
    experiments: Sequence[ExperimentResult],
    style_image: np.ndarray,
    content_image: np.ndarray,
) -> None:
    """Display original inputs plus stylized outputs for every experiment."""
    num_experiments = len(experiments)
    num_rows = 1 + num_experiments

    fig, axes = plt.subplots(num_rows, 2, figsize=(8, 4 * num_rows))

    axes[0, 0].imshow(style_image)
    axes[0, 0].set_title("Original Style", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(content_image)
    axes[0, 1].set_title("Original Content", fontsize=12)
    axes[0, 1].axis("off")

    for i, exp in enumerate(experiments):
        row_index = i + 1
        title = (
            f"Config {exp.name}\n"
            f"$\\alpha={exp.alpha:.0e}$\n"
            f"$\\beta={exp.beta:.0e}$\n"
            f"$\\gamma={exp.gamma:.0e}$"
        )

        axes[row_index, 0].text(
            0.5,
            0.5,
            title,
            ha="center",
            va="center",
            fontsize=12,
            transform=axes[row_index, 0].transAxes,
        )
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(exp.styled_image)
        axes[row_index, 1].set_title(f"Stylized Result {exp.name}", fontsize=12)
        axes[row_index, 1].axis("off")

    plt.tight_layout()
    plt.show()
