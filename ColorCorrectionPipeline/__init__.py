import sys
from ccp import ColorCorrection
from Configs.configs import Config
from models import MyModels
from FFC.FF_correction import FlatFieldCorrection

from pkg_resources import get_distribution, DistributionNotFound


try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    __version__ = 'unknown'

# get all key functions in key_functions.py, append to __all__
__all__ = [
    "ColorCorrection",
    "Config",
    "MyModels",
    "FlatFieldCorrection",
]

if "pdoc" in sys.modules:
    with open("README.md", "r") as fh:
        _readme = fh.read()
    __doc__ = _readme
