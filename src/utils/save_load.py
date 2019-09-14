import os

import tensorflow as tf


def save_h5_model(model, saved_model_path="saved_models", model_name="model.h5"):
    if not os.path.exists(saved_model_path):
        os.mkdir(saved_model_path)
    tf.keras.models.save_model(model=model, filepath=os.path.join(saved_model_path, model_name))
    print("OK!")
