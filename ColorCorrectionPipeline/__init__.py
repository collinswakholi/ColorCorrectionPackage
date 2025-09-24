import sys
from pkg_resources import get_distribution, DistributionNotFound
import warnings

# Suppress known PyTorch pynvml deprecation warning
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

# Try multiple import strategies for better compatibility
try:
    # Try explicit relative imports first
    from .ccp import ColorCorrection
    from .Configs.configs import Config
    from .models import MyModels
    from .FFC.FF_correction import FlatFieldCorrection
except ImportError:
    try:
        # Fall back to absolute imports
        from ColorCorrectionPipeline.ccp import ColorCorrection
        from ColorCorrectionPipeline.Configs.configs import Config
        from ColorCorrectionPipeline.models import MyModels
        from ColorCorrectionPipeline.FFC.FF_correction import FlatFieldCorrection
    except ImportError:
        # Final fallback - import from current directory
        import ccp
        import Configs.configs
        import models
        import FFC.FF_correction
        
        ColorCorrection = ccp.ColorCorrection
        Config = Configs.configs.Config
        MyModels = models.MyModels
        FlatFieldCorrection = FFC.FF_correction.FlatFieldCorrection

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
    try:
        with open("README.md", "r") as fh:
            _readme = fh.read()
        __doc__ = _readme
    except FileNotFoundError:
        __doc__ = "ColorCorrectionPipeline - A comprehensive color correction package"
