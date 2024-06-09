#!/usr/bin/env python3
import numpy as np
from scipy import signal, ndimage

__author__ = "Paola Ruiz Puentes, Regine Bueter"

def mean_filter_signal(pupil_diam, filter_type, kernel_size):
    """
    Filters the pupil signal. First a butterworth filter, and then a mean or meadian filter.

    Parameters
    ----------
    pupil_diam: list
        Pupil diameter or area signal
    filter_type: {'mean', 'median'}
        Specifies the filter type
    kernel_size: int
        The windows size for the filter.

    Returns
    -------
    np.array
        The filtered pupil diameter
    """
    # Frequency filter
    sampling_rate = 120
    butterworth_filter = signal.butter(2, (1 / 10) * sampling_rate, "low", output="sos", fs=sampling_rate)

    d = np.array(pupil_diam)
    d = signal.sosfilt(butterworth_filter, d)
    d[0:5] = [d[6]]
    # Median filter
    if filter_type == "median":
        pupil_diam = ndimage.median_filter(d, kernel_size)  # 1443

    # Mean filter
    if filter_type == "mean":
        pupil_diam = ndimage.uniform_filter(d, kernel_size)

    return pupil_diam


def mean_filter_fb(brightness_signal, filter_type, kernel_size):
    """
    Filters the brightness signal. First a butterworth filter, and then a mean or meadian filter.

    Parameters
    ----------
    brightness_signal: list
        Pupil diameter or area signal
    filter_type: {'mean', 'median'}
        Specifies the filter type
    kernel_size: int
        The windows size for the filter.

    Returns
    -------
    np.array
        The filtered brightness signal
    """
    # Median filter
    brightness = None
    if filter_type == "median":
        brightness = ndimage.median_filter(brightness_signal, kernel_size)  # 1443

    # Mean filter
    if filter_type == "mean":
        brightness = ndimage.uniform_filter(brightness_signal, kernel_size)
    return brightness
