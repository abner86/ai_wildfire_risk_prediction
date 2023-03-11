from __future__ import annotations

import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Default values.
EPOCHS = 100
BATCH_SIZE = 512
KERNEL_SIZE = 5

# Constants.
NUM_INPUTS = 21
NUM_CLASSES = 8
TRAIN_TEST_RATIO = 80  # percent for training, the rest for testing/validation
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 8


def read_example(serialized: bytes) -> tuple[tf.Tensor, tf.Tensor]:
    """Parses and reads a training example from TFRecords.
    Args:
        serialized: Serialized example bytes from TFRecord files.
    Returns: An (inputs, labels) pair of tensors.
    """
    features_dict = {
        "inputs": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(serialized, features_dict)
    inputs = tf.io.parse_tensor(example["inputs"], tf.float32)
    labels = tf.io.parse_tensor(example["labels"], tf.uint8)

    # TensorFlow cannot infer the shape's rank, so we set the shapes explicitly.
    inputs.set_shape([None, None, NUM_INPUTS])
    labels.set_shape([None, None, 1])

    # Classifications are measured against one-hot encoded vectors.
    one_hot_labels = tf.one_hot(labels[:, :, 0], NUM_CLASSES)
    return (inputs, one_hot_labels)


def read_dataset(data_path: str) -> tf.data.Dataset:
    """Reads compressed TFRecord files from a directory into a tf.data.Dataset.
    Args:
        data_path: Local or Cloud Storage directory path where the TFRecord files are.
    Returns: A tf.data.Dataset with the contents of the TFRecord files.
    """
    file_pattern = tf.io.gfile.join(data_path, "*.tfrecord.gz")
    file_names = tf.data.Dataset.list_files(file_pattern).cache()
    dataset = tf.data.TFRecordDataset(file_names, compression_type="GZIP")
    return dataset.map(read_example, num_parallel_calls=tf.data.AUTOTUNE)


def split_dataset(
    dataset: tf.data.Dataset,
    batch_size: int = BATCH_SIZE,
    train_test_ratio: int = TRAIN_TEST_RATIO,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Splits a dataset into training and validation subsets.
    Args:
        dataset: Full dataset with all the training examples.
        batch_size: Number of examples per training batch.
        train_test_ratio: Percent of the data to use for training.
    Returns: A (training, validation) dataset pair.
    """
    # For more information on how to optimize your tf.data.Dataset, see:
    #   https://www.tensorflow.org/guide/data_performance
    indexed_dataset = dataset.enumerate()  # add an index to each example
    train_dataset = (
        indexed_dataset.filter(lambda i, _: i % 100 <= train_test_ratio)
        .map(lambda _, data: data, num_parallel_calls=tf.data.AUTOTUNE)  # remove index
        .cache()  # cache the individual parsed examples
        .shuffle(SHUFFLE_BUFFER_SIZE)  # randomize the examples for the batches
        .batch(batch_size)  # batch randomized examples
        .prefetch(tf.data.AUTOTUNE)  # prefetch the next batch
    )
    validation_dataset = (
        indexed_dataset.filter(lambda i, _: i % 100 > train_test_ratio)
        .map(lambda _, data: data, num_parallel_calls=tf.data.AUTOTUNE)  # remove index
        .batch(batch_size)  # batch the parsed examples, no need to shuffle
        .cache()  # cache the batches of examples
        .prefetch(tf.data.AUTOTUNE)  # prefetch the next batch
    )
    return (train_dataset, validation_dataset)


def create_model(
    kernel_size: int,
    num_filters_1: int,
    num_filters_2: int,
    num_filters_3: int,
) -> tf.keras.Model:
    """Creates a Fully Convolutional Network Keras model.
    Args:
        kernel_size: Size of the square of neighboring pixels for the model to look at.
        num_filters_1: Number of filters for the first convolutional layer.
        num_filters_2: Number of filters for the second convolutional layer.
        num_filters_3: Number of filters for the first deconvolutional layer.
    Returns: A compiled fresh new model (not trained).
    """
    # Adapt the preprocessing layers.
    normalization = tf.keras.layers.Normalization()

    # Define the Fully Convolutional Network.
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(None, None, NUM_INPUTS), name="inputs"),
            normalization,
            tf.keras.layers.Conv2D(num_filters_1, kernel_size, activation="relu", name="conv2D_1"),
            tf.keras.layers.Conv2D(num_filters_2, kernel_size, activation="relu", name="conv2D_2"),
            tf.keras.layers.Conv2DTranspose(
                num_filters_3, kernel_size, activation="relu", name="deconv2D_1"
            ),
            tf.keras.layers.Conv2DTranspose(
                8, kernel_size, activation="relu", name="deconv2D_2"
            ),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="firerisk"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.OneHotIoU(
                num_classes=NUM_CLASSES,
                target_class_ids=list(range(NUM_CLASSES)),
            )
        ],
    )
    return model


def run(
    data_path: str,
    model_path: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    kernel_size: int = KERNEL_SIZE,
    train_test_ratio: int = TRAIN_TEST_RATIO,
    param_grid: dict = {},
) -> tf.keras.Model:
    """Creates and trains the model.
    Args:
        data_path: Local or Cloud Storage directory path where the TFRecord files are.
        model_path: Local or Cloud Storage directory path to store the trained model.
        epochs: Number of times the model goes through the training dataset during training.
        batch_size: Number of examples per training batch.
        kernel_size: Size of the square of neighboring pixels for the model to look at.
        train_test_ratio: Percent of the data to use for training.
        param_grid: Grid of hyperparameters to search over.
    Returns: The trained model.
    """
    print(f"data_path: {data_path}")
    print(f"model_path: {model_path}")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print(f"kernel_size: {kernel_size}")
    print(f"train_test_ratio: {train_test_ratio}")
    print("-" * 40)

    dataset = read_dataset(data_path)
    (train_dataset, test_dataset) = split_dataset(dataset, batch_size, train_test_ratio)

    model = KerasClassifier(
            build_fn=create_model,
            kernel_size=kernel_size,
            verbose=0
        )
    
    # Use GridSearchCV to find the best hyperparameters.
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(train_dataset)
    
    # Print the best parameters.
    print(f"Best parameters: {grid_result.best_params_}")
    
    # Re-create the model with the best hyperparameters.
    model = create_model(
        kernel_size=grid_result.best_params_["kernel_size"],
        num_filters_1=grid_result.best_params_["num_filters_1"],
        num_filters_2=grid_result.best_params_["num_filters_2"],
        num_filters_3=grid_result.best_params_["num_filters_3"],
    )
    
    print(model.summary())

    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
    )
    model.save(model_path)
    print(f"Model saved to path: {model_path}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
        help="Local or Cloud Storage directory path where the TFRecord files are.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local or Cloud Storage directory path to store the trained model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of times the model goes through the training dataset during training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of examples per training batch.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=KERNEL_SIZE,
        help="Size of the square of neighboring pixels for the model to look at.",
    )
    parser.add_argument(
        "--train-test-ratio",
        type=int,
        default=TRAIN_TEST_RATIO,
        help="Percent of the data to use for training.",
    )
    args = parser.parse_args()

    run(
        data_path=args.data_path,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        kernel_size=args.kernel_size,
        train_test_ratio=args.train_test_ratio,
    )
