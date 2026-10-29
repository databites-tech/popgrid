"""
popgrid — Population grid maps for any country.

Every country rasterised into N equal-area square cells,
coloured by administrative region.

Quick start
-----------
>>> from popgrid import AreaGrid
>>> import matplotlib.pyplot as plt
>>>
>>> ag = AreaGrid("ESP", n=1000)
>>> fig = ag.plot()
>>> plt.show()
"""

from .core import AreaGrid, PopGrid
from .data import (
    KNOWN_DISSOLVE,
    RECOMMENDED_SETTINGS,
    REGION_LABELS,
    bundled_countries,
    supported_countries,
)
from .exceptions import (
    CountryNotFoundError,
    DataNotFoundError,
    GeometryError,
    PopGridError,
)

__version__ = "0.6.1"
__author__ = "Josep Ferrer"
__email__ = "rfeers@gmail.com"

__all__ = [
    "AreaGrid",
    "PopGrid",
    "KNOWN_DISSOLVE",
    "RECOMMENDED_SETTINGS",
    "REGION_LABELS",
    "bundled_countries",
    "supported_countries",
    "PopGridError",
    "DataNotFoundError",
    "CountryNotFoundError",
    "GeometryError",
]
