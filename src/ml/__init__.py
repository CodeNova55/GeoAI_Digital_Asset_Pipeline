"""
Machine Learning Integration Module
===================================

Provides AI/ML integration for geospatial analysis including
spatial classification, object detection, and semantic segmentation.
Compatible with GeoAI plugins like DeepForest and OmniWaterMask.
"""

from .classifier import SpatialClassifier
from .object_detection import ObjectDetector
from .segmentation import SemanticSegmenter
from .change_detection import ChangeDetector
from .clustering import SpatialClusterer

__all__ = [
    "SpatialClassifier",
    "ObjectDetector",
    "SemanticSegmenter",
    "ChangeDetector",
    "SpatialClusterer"
]
