import dataclasses
import logging
import os
from typing import List
import keras
from keras.callbacks import CSVLogger
from matplotlib import pyplot as plt
import tensorflow as tf
from models import StyleTransferNet
from models.vgg19_based_gen import SaveOnEpochEnd, restore_model, get_model_name


IMG_SIZE = 256

@dataclasses.dataclass
class ExperimentConfig:
    alpha: float
    beta: float
    use_upsampling: bool
    use_reflection_padding: bool
    lr: float
    num_epochs: int


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("experiment_run_summary.log", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_and_preprocess(path, IMG_SIZE):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def make_dataset(dir_path):
    files = tf.data.Dataset.list_files(os.path.join(dir_path, "*"), shuffle=True)
    return files.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)


def get_log_filename(config: ExperimentConfig, style_variant: str, exp_num: int):
    """Creates a unique and descriptive filename for the CSV log."""
    base = f"exp_{exp_num}_a{config.alpha:.0e}_b{config.beta:.0e}_lr{config.lr:.0e}"
    flags = ""
    if config.use_upsampling:
        flags += "_UP"
    if config.use_reflection_padding:
        flags += "_RP"

    log_dir = f"logs/experiments/{style_variant}"
    os.makedirs(log_dir, exist_ok=True)

    return os.path.join(log_dir, f"{base}{flags}.csv")


def show_and_save_experiment_result(
    model: StyleTransferNet,
    config: ExperimentConfig,
    fig_suffix,
    content_img,
    style_img,
    style_variant,
    IMG_SIZE,
):
    out_img = model(tf.reshape(content_img, (1, IMG_SIZE, IMG_SIZE, 3)))

    plt.figure(figsize=(12, 4.5))

    plt.subplot(1, 3, 1)
    plt.imshow(tf.reshape(style_img, (IMG_SIZE, IMG_SIZE, 3)).numpy())
    plt.title("Style")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(tf.reshape(content_img, (IMG_SIZE, IMG_SIZE, 3)).numpy())
    plt.title("Content")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(tf.reshape(out_img, (IMG_SIZE, IMG_SIZE, 3)).numpy())
    plt.title("Stylized")
    plt.axis("off")

    suptitle = f"alpha={config.alpha:.0e}, beta={config.beta:.0e}, epochs={config.num_epochs} \n total variation loss"

    if config.use_upsampling:
        suptitle += ", upsampling"

    if config.use_reflection_padding:
        suptitle += ", reflection padding"

    plt.suptitle(suptitle)

    fig_path = f"img/experiments/vgg-with-generator/bulk-experiment/{style_variant}/{get_model_name(model, suffix=fig_suffix)}"

    plt.savefig(fig_path, dpi=300, bbox_inches="tight")


STYLE_PATHS = {
    "shoes": "data/shoes_style.jpg",
    "starry_night": "data/starry_night_style.jpg",
}


def run_experiments(
    configs: List[ExperimentConfig],
    IMG_SIZE,
    BATCH_SIZE,
    dataset=None,
    style=None,
):
    if style not in STYLE_PATHS:
        raise ValueError(
            f"Invalid style '{style}'. Must be one of {list(STYLE_PATHS.keys())}."
        )

    style_image = load_and_preprocess(STYLE_PATHS[style])
    style_image = tf.expand_dims(style_image, axis=0)

    if dataset is None or style is None:
        raise Exception("Specify dataset and style")

    content_image = load_and_preprocess("data/sea_content.jpg")
    content_image = tf.expand_dims(content_image, axis=0)

    train_content_ds = None

    if dataset == "coco":
        train_content_ds = make_dataset("data/coco/train_subset")

        # test_content_ds = make_dataset("data/coco/train_subset")

    if dataset is None:
        raise Exception("Specify dataset")

    train_ds = (
        train_content_ds.shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )

    logger.info(f"Starting batch of {len(configs)} experiments for style: {style}")

    for exp_num, config in enumerate(configs):
        logger.info("--------------------------------------------------")
        logger.info(f"STARTING EXPERIMENT {exp_num + 1}/{len(configs)}:")
        logger.info(f"  > Content Weight (alpha): {config.alpha:.0e}")
        logger.info(f"  > Style Weight (beta): {config.beta:.0e}")
        logger.info(f"  > Learning Rate (LR): {config.lr:.1e}")
        logger.info(f"  > Upsampling Enabled: {config.use_upsampling}")

        model = StyleTransferNet(
            input_shape=(IMG_SIZE, IMG_SIZE) + (3,),
            style_image=style_image,
            alpha=config.alpha,
            beta=config.beta,
            use_upsampling=config.use_upsampling,
            use_reflection_padding=config.use_reflection_padding,
        )
        model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=config.lr))

        exp_suffix = "tv_loss"
        if config.use_upsampling:
            exp_suffix += "_upsample"
        if config.use_reflection_padding:
            exp_suffix += "_refl_pad"

        log_filepath = get_log_filename(config, style, exp_num)
        csv_logger = CSVLogger(log_filepath, separator=",", append=False)

        callbacks_list = [
            SaveOnEpochEnd(suffix=exp_suffix, subdir="bulk-experiment/" + style),
            csv_logger,
        ]

        logger.info(f"  > Training model for {config.num_epochs} epochs...")

        model.fit(
            train_ds,
            epochs=config.num_epochs,
            callbacks=callbacks_list,
        )

        logger.info("  > Training complete. Saving final image and logs.")

        show_and_save_experiment_result(
            model=model,
            config=config,
            fig_suffix=exp_suffix,
            content_img=content_image,
            style_img=style_image,
            style_variant=style,
        )

        logger.info(f"  > Results saved to: img/.../{style} and CSV to: {log_filepath}")
        logger.info("--------------------------------------------------\n")
