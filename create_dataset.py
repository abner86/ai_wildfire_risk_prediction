"""This creates training and validation datasets for the model.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import List, Optional

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import ee
import numpy as np
import requests

from serving import data

# Default values.
POINTS_PER_CLASS = 100
PATCH_SIZE = 128
MAX_REQUESTS = 20  # default EE request quota

# Simplified polygons covering most land areas in the world.
US_POLYGON = [[-127.18, 19.39], [-64.03, 19.39], [-64.03, 51.29], [-127.18, 51.29], [-127.18, 19.39]]


def sample_points(
    seed: int,
    polygons: list[tuple[float, float]],
    points_per_class: int,
    scale: int = 500,
) -> Iterable[tuple[float, float]]:
    """Selects around the same number of points for every classification.
    This expects the input image to be an integer, for balanced regression points
    you could do `image.int()` to truncate the values into an integer.
    If the values are too large, it might be good to bucketize, for example
    the range is between 0 and ~1000 `image.divide(100).int()` would give ~10 buckets.
    Args:
        seed: Random seed to make sure to get different results on different workers.
        polygons: List of polygons to pick points from.
        points_per_class: Number of points per classification to pick.
        scale: Number of meters per pixel, a smaller scale will take longer.
    Returns: An iterable of the picked points.
    """
    data.ee_init()
    image = data.get_label_image()
    region = ee.Geometry.Polygon(polygons)
    points = image.stratifiedSample(
        points_per_class,
        region=region,
        scale=scale,
        seed=seed,
        geometries=True,
    )
    for point in points.toList(points.size()).getInfo():
        yield point["geometry"]["coordinates"]


def get_training_example(
    lonlat: tuple[float, float], patch_size: int = PATCH_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Gets an (inputs, labels) training example for year 2022.
    Args:
        lonlat: A (longitude, latitude) pair for the point of interest.
        patch_size: Size in pixels of the surrounding square patch.
    Returns: An (inputs, labels) pair of NumPy arrays.
    """
    data.ee_init()
    return (
        data.get_input_patch(2022, lonlat, patch_size),
        data.get_label_patch(lonlat, patch_size),
    )


def try_get_example(
    lonlat: tuple[float, float], patch_size: int = PATCH_SIZE
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Wrapper over `get_training_examples` that allows it to simply log errors instead of crashing."""
    try:
        yield get_training_example(lonlat, patch_size)
    except requests.exceptions.HTTPError as e:
        logging.exception(e)


def serialize_tensorflow(inputs: np.ndarray, labels: np.ndarray) -> bytes:
    """Serializes inputs and labels NumPy arrays as a tf.Example.
    Both inputs and outputs are expected to be dense tensors, not dictionaries.
    We serialize both the inputs and labels to save their shapes.
    Args:
        inputs: The example inputs as dense tensors.
        labels: The example labels as dense tensors.
    Returns: The serialized tf.Example as bytes.
    """
    import tensorflow as tf

    features = {
        name: tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(data).numpy()])
        )
        for name, data in {"inputs": inputs, "labels": labels}.items()
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


def run_tensorflow(
    data_path: str,
    points_per_class: int = POINTS_PER_CLASS,
    patch_size: int = PATCH_SIZE,
    max_requests: int = MAX_REQUESTS,
    polygons: list[tuple[float, float]] = US_POLYGON,
    beam_args: Optional[List[str]] = None,
) -> None:
    """Runs an Apache Beam pipeline to create a dataset.
    This fetches data from Earth Engine and creates a TFRecords dataset.
    We use `max_requests` to limit the number of concurrent requests to Earth Engine
    to avoid quota issues. You can request for an increas of quota if you need it.
    Args:
        data_path: Directory path to save the TFRecord files.
        points_per_class: Number of points to get for each classification.
        patch_size: Size in pixels of the surrounding square patch.
        max_requests: Limit the number of concurrent requests to Earth Engine.
        polygons: List of polygons to pick points from.
        beam_args: Apache Beam command line arguments to parse as pipeline options.
    """
    # Equally divide the number of points by the number of concurrent requests.
    num_points = max(int(points_per_class / max_requests), 1)

    beam_options = PipelineOptions(
        beam_args,
        save_main_session=True,
        setup_file="./setup.py",
        max_num_workers=max_requests,  # distributed runners
        direct_num_workers=max(max_requests, 20),  # direct runner
        disk_size_gb=50,
    )
    with beam.Pipeline(options=beam_options) as pipeline:
        (
            pipeline
            | "🌱 Make seeds" >> beam.Create(range(max_requests))
            | "📌 Sample points" >> beam.FlatMap(sample_points, polygons, num_points)
            | "🃏 Reshuffle" >> beam.Reshuffle()
            | "📑 Get examples" >> beam.FlatMap(try_get_example, patch_size)
            | "✍🏽 Serialize" >> beam.MapTuple(serialize_tensorflow)
            | "📚 Write TFRecords"
            >> beam.io.WriteToTFRecord(
                f"{data_path}/part", file_name_suffix=".tfrecord.gz"
            )
        )


if __name__ == "__main__":
    import argparse
    import logging

    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("framework", choices=["tensorflow"])
    parser.add_argument(
        "--data-path",
        required=True,
        help="Directory path to save the TFRecord files.",
    )
    parser.add_argument(
        "--points-per-class",
        default=POINTS_PER_CLASS,
        type=int,
        help="Number of points to get for each classification.",
    )
    parser.add_argument(
        "--patch-size",
        default=PATCH_SIZE,
        type=int,
        help="Size in pixels of the surrounding square patch.",
    )
    parser.add_argument(
        "--max-requests",
        default=MAX_REQUESTS,
        type=int,
        help="Limit the number of concurrent requests to Earth Engine.",
    )
    args, beam_args = parser.parse_known_args()

    if args.framework == "tensorflow":
        run_tensorflow(
            data_path=args.data_path,
            points_per_class=args.points_per_class,
            patch_size=args.patch_size,
            max_requests=args.max_requests,
            beam_args=beam_args,
        )
    else:
        raise ValueError(f"framework not supported: {args.framework}")
