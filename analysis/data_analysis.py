import os
import sys
import glob
import numpy as np
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
# import hcipy # dont need it for now
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from analysis.npz_metadata_utils import group_npz_files_by_metadata
except ImportError:  # pragma: no cover - fallback for direct script execution
    from npz_metadata_utils import group_npz_files_by_metadata

path = '/usr/local/aodev/CRED-One/Data/20260715/dark/'

# Example usage: scan the whole folder and group files by metadata.
groups_by_fps = group_npz_files_by_metadata(path, "fps")
list_a = groups_by_fps.get(3500.0, [])



