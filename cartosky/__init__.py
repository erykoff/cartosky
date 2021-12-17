try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("cartosky")
except PackageNotFoundError:
    # package is not installed
    pass

from .skymap import *
from .projections import *
from .formatters import *
