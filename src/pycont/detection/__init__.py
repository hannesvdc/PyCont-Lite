# src/pycont/detection/__init__.py
from .base import DetectionModule
from .parammin import ParamMinDetectionModule
from .parammax import ParamMaxDetectionModule

__all__ = [
    "DetectionModule",
    "ParamMinDetectionModule",
    "ParamMaxDetectionModule"
]