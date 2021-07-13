import sys
from .natural_cubic_spline import NaturalCubicSpline
from .polynomial import Polynomial
from .spline import Spline
from .weights_of_evidence import WeightsOfEvidence, WeightsOfEvidenceModel

if sys.version_info[:2] >= (3, 8):
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "pyspark-utilities"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
