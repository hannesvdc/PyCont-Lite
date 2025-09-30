# src/pycont/detection/__init__.py
from .base import DetectionModule
from .parammin import ParamMinDetectionModule
from .parammax import ParamMaxDetectionModule

# Re-export detectors you actually ship. Use try/except if some are optional.
# from .hopf import HopfDetector
# from .fold import FoldDetector
# from .bp import BranchPointDetector

__all__ = [
    "DetectionModule",
    "ParamMinDetectionModule",
    "ParamMaxDetectionModule"
]