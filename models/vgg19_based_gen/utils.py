import keras
import tensorflow as tf
from os import path

BASE_CHECKPOINTS_DIR = "checkpoints/weights/"


def preprocess(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = keras.applications.vgg19.preprocess_input(image * 255.0)
    return image


def gram_matrix(tensor):
    batch, h, w, c = tf.unstack(tf.shape(tensor))
    features = tf.reshape(tensor, [batch, h * w, c])
    gram = tf.matmul(features, features, transpose_a=True)
    return gram / tf.cast(h * w * c, tf.float32)


def get_model_name(model, path_save_id=None, suffix=None):
    if path_save_id is None:
        path_save_id = f"alpha_{model.alpha:.0e}_beta_{model.beta:.0e}"

    if suffix is not None:
        path_save_id = "_".join([path_save_id, suffix])

    return path_save_id


class SaveOnEpochEnd(tf.keras.callbacks.Callback):
    def __init__(self, path_save_id=None, subdir=None, suffix=None):
        super().__init__()
        self.path_save_id = path_save_id
        self.subdir = subdir
        self.suffix = suffix

    def on_epoch_end(self, epoch, logs=None):
        dir = BASE_CHECKPOINTS_DIR

        if self.subdir is not None:
            dir = path.join(dir, self.subdir)

        save_model(
            self.model,
            dir=dir,
            id=get_model_name(self.model, self.path_save_id, self.suffix),
        )


def save_model(model, dir=BASE_CHECKPOINTS_DIR, id="default"):
    model.generator.save_weights(path.join(dir, id, id))


def restore_model(model, dir=BASE_CHECKPOINTS_DIR, id="default"):
    model.generator.load_weights(path.join(dir, id, id))
    return model
