import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# Function to calculate dark and glow

def get_raw_data(dark_data, lim_lo, lim_hi):
    """
    Get the raw data from the dark data files.
    
    :param dark_data: numpy.ndarray
        Dark data
    :param lim_lo: int
        Lower limit for the data
    :param lim_hi: int
        Upper limit for the data
    :return: numpy.ndarray
        counts
    :return: int
        Number of frames
    """
    # Get the raw data
    raw_data = dark_data[lim_lo:lim_hi]
    # Get the number of frames
    nframes = raw_data.shape[0]
    
    return raw_data, nframes

def calculate_dark_and_glow(cts1, cts2, nframes_1, nframes_2, fps1, fps2):
    """
    Calculate the dark and glow values from two sets of counts.

    :param cts1: numpy.ndarray
        First set of counts.
    :param cts2: numpy.ndarray
        Second set of counts.

    :return: numpy.ndarray
        Dark map.
    :return: numpy.ndarray
        Glow map.
    :return: float
        Median dark value.
    :return: float
        Median glow value.
    """'

    t_1 = 1 / fps1 * nframes_1
    t_2 = 1 / fps2 * nframes_2

    shape = cts_1.shape  # Assuming cts_1 and cts_2 have the same shape

    # Reshape into 1D arrays for vectorized computation
    cts_1_flat = cts_1.ravel()
    cts_2_flat = cts_2.ravel()

    # Define the coefficient matrix
    A = np.array([[t_1, nframes_1], 
                [t_2, nframes_2]])

    # Solve for each pixel independently
    b = np.stack([cts_1_flat, cts_2_flat], axis=1)  # Shape (num_pixels, 2)
    solution = np.linalg.solve(A, b.T)  # Shape (2, num_pixels)

    # Reshape back to original image dimensions
    DC = solution[0].reshape(shape)
    Gl = solution[1].reshape(shape)

    # convert to e/px/frame
    gain = 1.87 # e/ADU
    DC *= gain
    Gl *= gain

    # Calculate mean values
    median_DC = np.median(DC)
    median_G = np.median(Gl)

    return DC, Gl, median_DC, median_G

