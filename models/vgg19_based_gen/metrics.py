"""Quantitative metrics helpers for the VGG19-based style transfer model."""

from __future__ import annotations

import random
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .style_transform_net import StyleTransferNet
from .utils import restore_model

AUTOTUNE = tf.data.AUTOTUNE

INCEPTION = tf.keras.applications.InceptionV3(
    include_top=False, weights="imagenet", pooling="avg"
)

LPIPS_LAYERS = [
    "block1_conv2",
    "block2_conv2",
    "block3_conv3",
    "block4_conv3",
    "block5_conv3",
]

_LPIPS_MODEL: tf.keras.Model | None = None


def _get_lpips_model() -> tf.keras.Model:
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
        vgg.trainable = False
        _LPIPS_MODEL = tf.keras.Model(
            inputs=vgg.input,
            outputs=[vgg.get_layer(name).output for name in LPIPS_LAYERS],
        )
        _LPIPS_MODEL.trainable = False
    return _LPIPS_MODEL


def load_and_resize_image(path: str, image_size: int) -> tf.Tensor:
    path = tf.convert_to_tensor(path, dtype=tf.string)
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.resize(image, (image_size, image_size))


def make_content_dataset(
    glob_pattern: str, image_size: int, batch_size: int, max_images: int
) -> tf.data.Dataset:
    files = tf.data.Dataset.list_files(glob_pattern, shuffle=True)
    dataset = files.map(
        lambda p: load_and_resize_image(p, image_size),
        num_parallel_calls=AUTOTUNE,
    )
    dataset = dataset.take(max_images)
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def sample_style_batch(
    style_image: tf.Tensor, batch_size: int, image_size: int, jitter: int = 32
) -> tf.Tensor:
    enlarged = tf.image.resize(style_image, (image_size + jitter, image_size + jitter))
    crops = []
    for _ in range(batch_size):
        crop = tf.image.random_crop(enlarged, size=(image_size, image_size, 3))
        crops.append(crop)
    return tf.stack(crops)


def preprocess_for_vgg(images: tf.Tensor) -> tf.Tensor:
    return tf.keras.applications.vgg16.preprocess_input(images * 255.0)


def lpips_distance_tf(batch_a: tf.Tensor, batch_b: tf.Tensor) -> tf.Tensor:
    model = _get_lpips_model()
    feats_a = model(preprocess_for_vgg(batch_a))
    feats_b = model(preprocess_for_vgg(batch_b))
    per_layer = []
    for fa, fb in zip(feats_a, feats_b):
        fa = tf.nn.l2_normalize(fa, axis=-1)
        fb = tf.nn.l2_normalize(fb, axis=-1)
        per_layer.append(tf.reduce_mean(tf.square(fa - fb), axis=[1, 2, 3]))
    stacked = tf.stack(per_layer, axis=1)
    return tf.reduce_mean(stacked, axis=1)


def get_inception_activations(images: tf.Tensor) -> tf.Tensor:
    resized = tf.image.resize(images, (299, 299))
    processed = tf.keras.applications.inception_v3.preprocess_input(resized * 255.0)
    return INCEPTION(processed)


def polynomial_mmd(samples_a: tf.Tensor, samples_b: tf.Tensor) -> tf.Tensor:
    n_a = tf.shape(samples_a)[0]
    n_b = tf.shape(samples_b)[0]
    tf.debugging.assert_greater(
        n_a, 1, message="Need at least two stylized samples for KID."
    )
    tf.debugging.assert_greater(
        n_b, 1, message="Need at least two style samples for KID."
    )

    dim = tf.cast(tf.shape(samples_a)[1], tf.float32)
    k_aa = tf.pow(tf.matmul(samples_a, samples_a, transpose_b=True) / dim + 1.0, 3)
    k_bb = tf.pow(tf.matmul(samples_b, samples_b, transpose_b=True) / dim + 1.0, 3)
    k_ab = tf.pow(tf.matmul(samples_a, samples_b, transpose_b=True) / dim + 1.0, 3)

    def _off_diagonal_mean(mat: tf.Tensor) -> tf.Tensor:
        n = tf.cast(tf.shape(mat)[0], tf.float32)
        trace = tf.linalg.trace(mat)
        return (tf.reduce_sum(mat) - trace) / (n * (n - 1.0))

    cross = tf.reduce_sum(k_ab) / tf.cast(tf.size(k_ab), tf.float32)
    return _off_diagonal_mean(k_aa) + _off_diagonal_mean(k_bb) - 2.0 * cross


def evaluate_checkpoint(
    *,
    checkpoint_dir,
    checkpoint_id,
    style_image_path,
    dataset_glob,
    image_size,
    batch_size,
    max_images,
    alpha,
    beta,
    gamma=1e-3,
    use_upsampling=True,
    use_reflection_padding=True,
    enable_lpips=True,
) -> Dict[str, float | int | None]:
    style_tensor = tf.expand_dims(
        load_and_resize_image(style_image_path, image_size), axis=0
    )
    model = StyleTransferNet(
        input_shape=(image_size, image_size, 3),
        style_image=style_tensor,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        use_upsampling=use_upsampling,
        use_reflection_padding=use_reflection_padding,
    )
    restore_model(model, dir=str(checkpoint_dir), id=checkpoint_id)

    content_ds = make_content_dataset(dataset_glob, image_size, batch_size, max_images)
    ssim_metric = tf.keras.metrics.Mean(name="ssim")
    lpips_metric = tf.keras.metrics.Mean(name="lpips") if enable_lpips else None
    stylized_feats = []
    style_feats = []

    processed = 0
    for content_batch in content_ds:
        stylized_batch = model(content_batch, training=False)
        ssim_vals = tf.image.ssim(content_batch, stylized_batch, max_val=1.0)
        ssim_metric.update_state(ssim_vals)

        bs = int(stylized_batch.shape[0])
        style_batch = sample_style_batch(style_tensor[0], bs, image_size)
        if enable_lpips and lpips_metric is not None:
            lpips_vals = lpips_distance_tf(stylized_batch, style_batch)
            lpips_metric.update_state(lpips_vals)

        stylized_feats.append(get_inception_activations(stylized_batch))
        style_feats.append(get_inception_activations(style_batch))

        processed += bs
        if processed >= max_images:
            break

    stylized_acts = tf.concat(stylized_feats, axis=0)
    style_acts = tf.concat(style_feats, axis=0)
    kid_value = polynomial_mmd(stylized_acts, style_acts)

    return {
        "num_images": int(stylized_acts.shape[0]),
        "ssim": float(ssim_metric.result().numpy()),
        "lpips": float(lpips_metric.result().numpy())
        if enable_lpips and lpips_metric is not None
        else None,
        "kid": float(kid_value.numpy()),
    }


def _pick_content_paths(
    config: Mapping[str, Sequence[str]],
    dataset_glob: str,
    max_images: int,
    seed: int = 0,
) -> List[str]:
    paths = list(dict.fromkeys(config["content_images"]))
    if len(paths) >= max_images:
        return paths[:max_images]

    all_paths = tf.io.gfile.glob(dataset_glob)
    all_paths = [p for p in all_paths if p not in set(paths)]
    rng = random.Random(seed)
    rng.shuffle(all_paths)

    need = max_images - len(paths)
    return paths + all_paths[:need]


def plot_style_grid(
    style_name: str,
    content_np: np.ndarray,
    stylized_list: Sequence[np.ndarray],
    metrics_list: Sequence[Mapping[str, object]],
    dpi: int,
) -> None:
    rows = int(content_np.shape[0])
    model_cols = len(stylized_list)
    cols = 1 + model_cols

    w = max(8.0, cols * 2.05)
    h = max(3.5, rows * 2.05)
    fig, axes = plt.subplots(rows, cols, figsize=(w, h), dpi=dpi, squeeze=False)
    fig.subplots_adjust(
        left=0.01, right=0.995, bottom=0.01, top=0.90, wspace=0.02, hspace=0.02
    )

    for r in range(rows):
        ax = axes[r, 0]
        ax.imshow(content_np[r])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    for c, (stylized_np, info) in enumerate(zip(stylized_list, metrics_list), start=1):
        for r in range(rows):
            ax = axes[r, c]
            ax.imshow(stylized_np[r])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            text = f"SSIM {info['example_ssim'][r]:.3f}"
            example_lpips = info.get("example_lpips")
            if example_lpips is not None:
                text += f"  LPIPS {example_lpips[r]:.3f}"
            ax.text(
                0.02,
                0.02,
                text,
                transform=ax.transAxes,
                fontsize=6,
                va="bottom",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    alpha=0.8,
                    facecolor="white",
                    edgecolor="white",
                ),
            )

    fig.suptitle(
        f"{style_name.replace('_', ' ').title()} stylizations", fontsize=12, y=0.975
    )

    header_y = 0.915
    for c in range(cols):
        if c == 0:
            pos = axes[0, c].get_position()
            x = (pos.x0 + pos.x1) / 2
            fig.text(x, header_y, "Content", ha="center", va="bottom", fontsize=9)
        else:
            info = metrics_list[c - 1]
            ax = axes[0, c]
            lpips_text = (
                f"  LPIPS {info['batch_lpips']:.3f}"
                if info.get("batch_lpips") is not None
                else ""
            )
            ax.text(
                0.02,
                0.98,
                f"{info['label']}\n"
                f"SSIM {info['batch_ssim']:.3f}{lpips_text}  "
                f"KID {info['batch_kid']:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                zorder=10,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                ),
            )

    for r in range(rows):
        pos = axes[r, 0].get_position()
        y = (pos.y0 + pos.y1) / 2
        fig.text(0.006, y, f"{r + 1}", ha="left", va="center", fontsize=8)

    plt.show()


def evaluate_style(
    style_name: str,
    config: Mapping[str, object],
    *,
    dataset_glob: str,
    eval_image_size: int,
    eval_batch_size: int,
    max_eval_images: int,
    max_grid_images: int,
    checkpoint_specs: Sequence[Mapping[str, object]],
    fig_dpi: int,
    enable_lpips: bool,
    plot: bool,
) -> List[Dict[str, object]]:
    content_paths = _pick_content_paths(
        config, dataset_glob, max_grid_images, seed=0
    )
    content_examples = tf.stack(
        [load_and_resize_image(path, eval_image_size) for path in content_paths],
        axis=0,
    )
    content_np = content_examples.numpy()

    style_tensor = tf.expand_dims(
        load_and_resize_image(config["style_path"], eval_image_size), axis=0
    )
    style_batch_for_lpips = (
        tf.repeat(style_tensor, repeats=content_examples.shape[0], axis=0)
        if enable_lpips
        else None
    )

    stylized_outputs = []
    metrics_per_model = []
    style_results: List[Dict[str, object]] = []

    for checkpoint in checkpoint_specs:
        eval_cfg = dict(
            checkpoint_dir=config["checkpoint_dir"],
            checkpoint_id=checkpoint["checkpoint_id"],
            style_image_path=config["style_path"],
            dataset_glob=dataset_glob,
            image_size=eval_image_size,
            batch_size=eval_batch_size,
            max_images=max_eval_images,
            alpha=checkpoint["alpha"],
            beta=checkpoint["beta"],
            gamma=checkpoint.get("gamma", 1e-3),
            use_upsampling=checkpoint["use_upsampling"],
            use_reflection_padding=checkpoint["use_reflection_padding"],
            enable_lpips=enable_lpips,
        )
        batch_metrics = evaluate_checkpoint(**eval_cfg)
        style_results.append(
            {"style": style_name, "label": checkpoint["label"], **batch_metrics}
        )

        sample_model = StyleTransferNet(
            input_shape=(eval_image_size, eval_image_size, 3),
            style_image=style_tensor,
            alpha=checkpoint["alpha"],
            beta=checkpoint["beta"],
            gamma=checkpoint.get("gamma", 1e-3),
            use_upsampling=checkpoint["use_upsampling"],
            use_reflection_padding=checkpoint["use_reflection_padding"],
        )
        restore_model(
            sample_model,
            dir=str(config["checkpoint_dir"]),
            id=checkpoint["checkpoint_id"],
        )

        stylized_batch = sample_model(content_examples, training=False)
        stylized_np = np.clip(stylized_batch.numpy(), 0.0, 1.0)

        ssim_examples = tf.image.ssim(
            content_examples, stylized_batch, max_val=1.0
        ).numpy()
        lpips_examples = (
            lpips_distance_tf(stylized_batch, style_batch_for_lpips).numpy()
            if enable_lpips and style_batch_for_lpips is not None
            else None
        )

        stylized_outputs.append(stylized_np)
        metrics_per_model.append(
            {
                "label": checkpoint["label"],
                "example_ssim": ssim_examples,
                "example_lpips": lpips_examples,
                "batch_ssim": batch_metrics["ssim"],
                "batch_lpips": batch_metrics["lpips"],
                "batch_kid": batch_metrics["kid"],
            }
        )

    if plot:
        plot_style_grid(style_name, content_np, stylized_outputs, metrics_per_model, dpi=fig_dpi)
    return style_results


def run_quantitative_evaluation(
    *,
    style_eval_configs: Mapping[str, Mapping[str, object]],
    checkpoint_specs: Sequence[Mapping[str, object]],
    dataset_glob: str,
    eval_image_size: int,
    eval_batch_size: int,
    max_eval_images: int,
    max_grid_images: int,
    fig_dpi: int = 160,
    enable_lpips: bool = True,
    plot: bool = True,
) -> List[Dict[str, object]]:
    """Evaluate all style checkpoints and return aggregated metrics."""
    all_results: List[Dict[str, object]] = []
    for style_name, cfg in style_eval_configs.items():
        all_results.extend(
            evaluate_style(
                style_name,
                cfg,
                dataset_glob=dataset_glob,
                eval_image_size=eval_image_size,
                eval_batch_size=eval_batch_size,
                max_eval_images=max_eval_images,
                max_grid_images=max_grid_images,
                checkpoint_specs=checkpoint_specs,
                fig_dpi=fig_dpi,
                enable_lpips=enable_lpips,
                plot=plot,
            )
        )
    return all_results
