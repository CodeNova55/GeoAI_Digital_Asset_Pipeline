"""
GeoAI Digital Asset Pipeline
============================

A professional QGIS + AI integration project for geospatial analysis
combining PyQGIS automation with machine learning for digital asset creation.

Author: GeoAI Research Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "GeoAI Research Team"
__license__ = "MIT"

from .pyqgis.batch_processor import BatchProcessor
from .pyqgis.layer_styler import LayerStyler
from .pyqgis.crs_transformer import CRSTransformer
from .pyqgis.data_validator import DataValidator
from .ml.classifier import SpatialClassifier
from .ml.object_detection import ObjectDetector
from .ml.segmentation import SemanticSegmenter
from .pipeline.feature_extractor import FeatureExtractor
from .pipeline.asset_manager import AssetManager
from .pipeline.quality_assurance import QualityAssurance
from .workflow.processing_graph import ProcessingGraph
from .workflow.logger import PipelineLogger
from .workflow.progress_tracker import ProgressTracker

__all__ = [
    "BatchProcessor",
    "LayerStyler",
    "CRSTransformer",
    "DataValidator",
    "SpatialClassifier",
    "ObjectDetector",
    "SemanticSegmenter",
    "FeatureExtractor",
    "AssetManager",
    "QualityAssurance",
    "ProcessingGraph",
    "PipelineLogger",
    "ProgressTracker",
]
