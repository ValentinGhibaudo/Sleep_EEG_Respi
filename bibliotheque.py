import xarray as xr
import pandas as pd
import numpy as np

from configuration import base_folder
from params import *

def init_da(coords, name = None):
    """
    Just initialize an empty xarray full of NaN according to coordinates.
    """
    dims = list(coords.keys())
    coords = coords

    def size_of(element):
        element = np.array(element)
        size = element.size
        return size

    shape = tuple([size_of(element) for element in list(coords.values())])
    data = np.full(shape, np.nan)
    da = xr.DataArray(data=data, dims=dims, coords=coords, name = name)
    return da