import numpy as np
import tensorflow as tf


def run(data: dict, model_dir: str) -> dict:
    model = tf.keras.models.load_model(model_dir)
    prediction_values = np.array(list(data.values()))
    transposed = np.transpose(prediction_values, (1, 2, 0))
    predictions = model.predict(np.expand_dims(transposed, axis=0)).tolist()

    return {"predictions": predictions}