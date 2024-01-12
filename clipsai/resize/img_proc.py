"""
Utilities for image processing.
"""
import numpy as np


def rgb_to_gray(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale.

    Parameters
    ----------
    rgb_image: np.ndarray
        The RGB image to convert to grayscale.

    Returns
    -------
    np.ndarray
        The grayscale image.
    """
    rgb_to_gray = np.array([0.299, 0.587, 0.114])
    return (rgb_image @ rgb_to_gray).astype(np.uint8)


def calc_img_bytes(width: int, height: int, channels: int) -> int:
    """
    Calculate the memory required to store a set of images.

    Parameters
    ----------
    width: int
        The width of the image.
    height: int
        The height of the image.
    channels: int
        The number of channels in the image.

    Returns
    -------
    int
        The number of bytes required to store the images.
    """
    return width * height * channels
