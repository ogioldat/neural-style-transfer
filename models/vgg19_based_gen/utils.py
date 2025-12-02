import keras
import tensorflow as tf
from os import path


def preprocess(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = keras.applications.vgg19.preprocess_input(image * 255.0)
    return image


def gram_matrix(tensor):
    batch, h, w, c = tf.unstack(tf.shape(tensor))
    features = tf.reshape(tensor, [batch, h * w, c])
    gram = tf.matmul(features, features, transpose_a=True)
    return gram / tf.cast(h * w * c, tf.float32)


class SaveOnEpochEnd(tf.keras.callbacks.Callback):
    def __init__(self, paths_save_id="default"):
        super().__init__()
        self.paths_save_id = paths_save_id

    def on_epoch_end(self, epoch, logs=None):
        save_model(self.model, id=f"{self.paths_save_id}-ep{epoch}")


def save_model(model, id="default"):
    model.generator.save_weights(path.join("checkpoints/weights/", id, id))


def restore_model(model, id="default"):
    model.generator.load_weights(path.join("checkpoints/weights/", id, id))
    return model
