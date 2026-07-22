import os
import sys
import numpy as np
# Compatibility patch for old Astropy with NumPy >= 1.23
if not hasattr(np, "asscalar"):
    np.asscalar = lambda array: array.item()
from scipy.ndimage import affine_transform, median_filter
from skimage.registration import phase_cross_correlation
from skimage.util import view_as_windows
from scipy.ndimage import shift
from scipy.optimize import curve_fit
import json
import io
from configobj import flatten_errors, ConfigObj
from validate import Validator, ValidateError
import warnings
import hcipy
import matplotlib.pyplot as plt
import astropy.io.fits as fits


def locate_badpix(data, sigmaclip = 5, plot=True):
    ''' -----------------------------------------------------------------------
    Locates bad pixels by fitting a gaussian distribution to the image
    intensity and then cutting outliers at the level of 'sigmaclip'

    XYZ -- this function needs to be rewritten to be cleaner
    ----------------------------------------------------------------------- '''
    # Create a vector of values borned by the min and max of the provided data
    xvals = np.arange(data.min(), data.max())
    # Short the value of the provided data to create an histogram
    yvals = np.histogram(data.ravel(), bins=xvals, density=True)[0]
    # Find the position associated to the faintest and brightest points.
    m1 = np.abs(np.cumsum(yvals)-0.0005).argmin()
    m2 = np.abs(np.cumsum(yvals)-0.9995).argmin()
    # Compute the mean of the selected points
    midx = 0.5*(xvals[m1]+xvals[m2])
    # Reduce the list of points by removing the faintest and brightest points.
    tmpx = xvals[m1:m2]
    tmpy = yvals[m1:m2]
    # Fit a Gaussian on the selected points
    popt, pcov = curve_fit(gaussfunc, tmpx, tmpy, p0 = (midx,25))
    # Extract the mean and standard deviation
    mean   = popt[0]
    stddev = popt[1]
    # Compute the brighntness limits of the point to keep
    cliphigh  = mean + sigmaclip*np.abs(stddev)
    cliplow   = mean - sigmaclip*np.abs(stddev)
    # Generate the bad pixel map
    bpmask = np.round(data > cliphigh) + np.round(data < cliplow)
    # Plot the histogram
    if plot:
        # Create figure
        plt.figure(figsize=(5, 3))

        # Add a small offset to avoid log(0) issues
        epsilon = 1e-10
        y_plot = yvals + epsilon
        y_fit = gaussfunc(xvals[:-1], *popt) + epsilon

        # Plot the histogram with log y, symlog x
        plt.plot(xvals[:-1], y_plot, 'k.', label='Pixel intensity histogram', alpha=0.5)

        # Overplot Gaussian fit on the data
        plt.plot(xvals[:-1], y_fit, 'r-',
                 label=f'Gaussian fit (μ={mean:.2f}, σ={stddev:.2f})', alpha=0.5)

        plt.yscale('log')
        plt.xscale('symlog')

        # Add labels
        plt.xlabel('Pixel intensity')
        plt.ylabel('Normalized frequency (log scale)')
        plt.legend()

        # Set y-axis limits to focus on relevant range
        # Find the maximum value in the histogram (excluding outliers)
        y_max = max(np.max(y_plot), 0.1)
        plt.ylim(epsilon, y_max * 1.5)

        # Prepare the title of the figure
        title = f'Bad Pixel Detection (σ-clip = {sigmaclip})'
        title += '\nPixels outside shaded area are considered bad'

        # Add the title
        plt.title(title)

        # Highlight pixels considered as good pixels
        plt.axvspan(cliplow, cliphigh, alpha=0.2, color='grey',
                  label=f'Good pixel range: [{cliplow:.2f}, {cliphigh:.2f}]')

        # Add vertical lines at clip boundaries
        plt.axvline(x=cliplow, color='red', linestyle='--', alpha=0.7)
        plt.axvline(x=cliphigh, color='red', linestyle='--', alpha=0.7)

        # Add text showing percentage of bad pixels
        bad_pixel_percentage = 100.0 * float(np.sum(bpmask)) / bpmask.size
        plt.figtext(0.5, 0.01, f'Bad pixels: {bad_pixel_percentage:.2f}% of image',
                  ha='center', fontsize=10)

        # Show the figure
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        plt.show()

    return np.array(bpmask, dtype=np.float32)


def removebadpix(data, mask, kern=5):
    """Removes bad pixels by replacing them with the local median of
    their kern x kern neighbourhood.  Only patches centred on bad pixels
    are evaluated, so the cost scales with the number of bad pixels rather
    than the size of the image.
    Inputs
    -------
        data--a 2d numpy array
        mask--a 2d numpy binary mask indicating bad pixels
              (ones are bad)
        kern--the kernel size for the local median (must be odd)
    Outputs
    -------
        a 2d numpy array with the bad pixels replaced by the local median
    """
    bad_ys, bad_xs = np.where(mask > 0)
    if len(bad_ys) == 0:
        return data.copy()

    tmp = data.copy()
    half = kern // 2
    # Pad with wrap so edge bad pixels get a full neighbourhood
    padded = np.pad(data, half, mode='wrap')
    # view_as_windows gives a zero-copy view: shape (H, W, kern, kern)
    windows = view_as_windows(padded, (kern, kern))
    # Extract only the patches at bad pixel locations: (n_bad, kern*kern)
    patches = windows[bad_ys, bad_xs].reshape(len(bad_ys), -1)
    tmp[bad_ys, bad_xs] = np.median(patches, axis=1)
    return tmp

def gaussfunc(x, mu, sig):
    """1-d Gaussian  x, mu, sigma"""
    return (1.0/(sig*np.sqrt(2*np.pi))*
            np.exp(-(x-mu)**2/(2*sig**2)))
