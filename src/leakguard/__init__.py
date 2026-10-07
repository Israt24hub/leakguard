"""LeakGuard: find data leakage before it fools your model."""

from .checks import Finding
from .report import Report
from .scan import scan

__all__ = ["scan", "Report", "Finding"]
__version__ = "0.1.0"
