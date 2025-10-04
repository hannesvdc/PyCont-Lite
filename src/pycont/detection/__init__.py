# src/pycont/detection/__init__.py
from .base import DetectionModule
from .parammin import ParamMinDetectionModule
from .parammax import ParamMaxDetectionModule
from .fold import FoldDetectionModule
from .bifurcation import BifurcationDetectionModule

__all__ = [
    "DetectionModule",
    "ParamMinDetectionModule",
    "ParamMaxDetectionModule",
    "FoldDetectionModule",
    "BifurcationDetectionModule",
]