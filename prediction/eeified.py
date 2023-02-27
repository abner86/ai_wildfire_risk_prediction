import tensorflow as tf

def ee_decode(batch: tf.Tensor, dtype: tf.DType = tf.float32) -> tf.Tensor:
  parse_tensor = lambda x: tf.io.parse_tensor(x, dtype)
  return tf.map_fn(parse_tensor, tf.io.decode_base64(batch), dtype)

def ee_encode(batch: tf.Tensor) -> tf.Tensor:
  encode_tensor = lambda x: tf.io.encode_base64(tf.io.serialize_tensor(x))
  return tf.map_fn(encode_tensor, batch, tf.string)

def eeify_model(model: tf.keras.Model) -> tf.keras.Model:
  # Assume we called `ee.Image.toArray` on the input image.
  input_layers = {
      "array": tf.keras.Input([], dtype=tf.string, name="array")
  }
  eeification = [
      tf.keras.layers.Lambda(ee_decode, name="ee_decode"),
      model,
      tf.keras.layers.Lambda(ee_encode, name="ee_encode"),
  ]
  output = input_layers["array"]
  for layer in eeification:
    output = layer(output)
  return tf.keras.Model(input_layers, output)

def save_to_cloud(trained_model: tf.keras.Model, bucket: str) -> None:
  # EEify the model and save it to Cloud Storage.
  eeified_model = eeify_model(trained_model)
  eeified_model.save(f"gs://{bucket}/fire-risk/eeified-model")
  
  tf.keras.utils.plot_model(eeified_model, show_shapes=True)