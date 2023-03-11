from __future__ import annotations

import numpy as np
from plotly.graph_objects import Image
from plotly.subplots import make_subplots

CLASSIFICATIONS = {
    "🪨 No Data": "FFFFFF",
    "🟢 Very Low": "38A800",
    "🍏 Low": "D1FF73",
    "🟡 Moderate": "FFFF00",
    "🟠 High": "FFAA00",
    "🔴 Very High": "FF0000",
    "⚫ Non-burnable": "B2B2B2",
    "💧 Water": "0070FF",
}


def render_classifications(values: np.ndarray, palette):
    """Renders a classifications NumPy array with shape (width, height, 1) as an image.
    Args:
        values: An uint8 array with shape (width, height, 1).
        palette: List of hex encoded colors.
    Returns: An uint8 array with shape (width, height, rgb) with colors from the palette.
    """
    # Create a color map from a hex color palette.
    xs = np.linspace(0, len(palette), 256)
    indices = np.arange(len(palette))

    red = np.interp(xs, indices, [int(c[0:2], 16) for c in palette])
    green = np.interp(xs, indices, [int(c[2:4], 16) for c in palette])
    blue = np.interp(xs, indices, [int(c[4:6], 16) for c in palette])

    color_map = np.array([red, green, blue]).astype(np.uint8).transpose()
    color_indices = (values / len(palette) * 255).astype(np.uint8)
    return np.take(color_map, color_indices, axis=0)


def render_rgb_images(
    values: np.ndarray, min: float = 0.0, max: float = 1.0
) -> np.ndarray:
    """Renders a numeric NumPy array with shape (width, height, rgb) as an image.
    Args:
        values: A float array with shape (width, height, rgb).
        min: Minimum value in the values.
        max: Maximum value in the values.
    Returns: An uint8 array with shape (width, height, rgb).
    """
    scaled_values = (values - min) / (max - min)
    rgb_values = np.clip(scaled_values, 0, 1) * 255
    return rgb_values.astype(np.uint8)


def render_label_image(patch: np.ndarray):
    """Renders a land cover image."""
    palette = list(CLASSIFICATIONS.values())
    return render_classifications(patch[:, :, 0], palette)


def render_input(patch: np.ndarray, max: float = 3000) -> np.ndarray:
    red = patch[:, :, 3]
    green = patch[:, :, 2]
    blue = patch[:, :, 1]
    rgb_patch = np.stack([red, green, blue], axis=-1)
    return render_rgb_images(rgb_patch, 0, max)

def show_inputs(inputs: np.ndarray, max: float = 3000) -> None:
    """Shows the input data as an image."""
    fig = make_subplots(rows=1, cols=1, subplot_titles=("Input"))
    fig.add_trace(Image(z=render_input(inputs, max)), row=1, col=1)
    fig.show()


def show_outputs(outputs: np.ndarray) -> None:
    """Shows the outputs/labels data as an image."""
    fig = make_subplots(rows=1, cols=1, subplot_titles=("Fire Risk",))
    fig.add_trace(Image(z=render_label_image(outputs)), row=1, col=1)
    fig.show()

def show_example(inputs: np.ndarray, labels: np.ndarray, max: float = 3000):
    """Shows an example of inputs and labels an image."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Input", "WHP"))
    fig.add_trace(Image(z=render_input(inputs, max)), row=1, col=1)
    fig.add_trace(Image(z=render_label_image(labels)), row=1, col=2)
    fig.show()

def show_prediction(outputs: np.ndarray, labels: np.ndarray, max: float = 3000):
    """Shows an example of inputs and labels an image."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Prediction", "WHP"))
    fig.add_trace(Image(z=render_label_image(outputs)), row=1, col=1)
    fig.add_trace(Image(z=render_label_image(labels)), row=1, col=2)
    fig.show()

def show_legend() -> None:
    """Shows the legend of the land cover classifications."""

    def color_box(red: int, green: int, blue: int) -> str:
        return f"\033[48;2;{red};{green};{blue}m"

    reset_color = "\u001b[0m"
    for name, color in CLASSIFICATIONS.items():
        red = int(color[0:2], 16)
        green = int(color[2:4], 16)
        blue = int(color[4:6], 16)
        print(f"{color_box(red, green, blue)}   {reset_color} {name}")
