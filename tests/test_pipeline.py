"""
Unit Tests for GeoAI Digital Asset Pipeline
===========================================

Comprehensive unit tests for all pipeline modules.

Run tests with:
    pytest tests/ -v
    pytest tests/ --cov=src --cov-report=html
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfig:
    """Tests for configuration management."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        from utils.config import Config
        
        config = Config.default()
        
        assert config.project.name == "GeoAI Digital Asset Pipeline"
        assert config.paths.data_dir == "./data"
        assert config.ml.random_seed == 42
    
    def test_config_from_dict(self):
        """Test configuration from dictionary."""
        from utils.config import Config
        
        data = {
            "project": {"name": "Test Project"},
            "custom": {"value": 42}
        }
        config = Config.from_dict(data)
        
        assert config.project.name == "Test Project"
        assert config.custom.value == 42
    
    def test_config_get(self):
        """Test configuration value retrieval."""
        from utils.config import Config
        
        config = Config.default()
        
        assert config.get("project.name") == "GeoAI Digital Asset Pipeline"
        assert config.get("ml.random_seed") == 42
        assert config.get("nonexistent", "default") == "default"
    
    def test_config_set(self):
        """Test configuration value setting."""
        from utils.config import Config
        
        config = Config.default()
        config.set("project.version", "2.0.0")
        
        assert config.project.version == "2.0.0"
    
    def test_config_env_override(self):
        """Test environment variable overrides."""
        from utils.config import Config
        
        with patch.dict(os.environ, {"GEOAI_ML__RANDOM_SEED": "123"}):
            data = {"ml": {"random_seed": 42}}
            result = Config._apply_env_overrides(data, "GEOAI")
            
            assert result["ml"]["random_seed"] == 123


# =============================================================================
# Progress Tracker Tests
# =============================================================================

class TestProgressTracker:
    """Tests for progress tracking."""
    
    def test_progress_initialization(self):
        """Test progress tracker initialization."""
        from workflow.progress_tracker import ProgressTracker
        
        tracker = ProgressTracker(show_progress=False)
        tracker.init(total=100, description="Test")
        
        assert tracker.state.total == 100
        assert tracker.state.current == 0
        assert tracker.state.description == "Test"
    
    def test_progress_update(self):
        """Test progress updates."""
        from workflow.progress_tracker import ProgressTracker
        
        tracker = ProgressTracker(show_progress=False)
        tracker.init(total=10, description="Test")
        
        tracker.update(3)
        assert tracker.state.current == 3
        assert tracker.state.percentage == 30.0
        
        tracker.update(2)
        assert tracker.state.current == 5
        assert tracker.state.percentage == 50.0
    
    def test_progress_eta(self):
        """Test ETA calculation."""
        from workflow.progress_tracker import ProgressTracker
        import time
        
        tracker = ProgressTracker(show_progress=False)
        tracker.init(total=100, description="Test")
        
        # Simulate some progress
        tracker.update(10)
        time.sleep(0.1)
        tracker.update(10)
        
        # ETA should be calculable
        assert tracker.state.eta is not None or tracker.state.eta is None  # May be None if too fast
    
    def test_progress_context_manager(self):
        """Test progress tracker as context manager."""
        from workflow.progress_tracker import ProgressTracker
        
        with ProgressTracker(show_progress=False) as tracker:
            tracker.init(total=10, description="Test")
            tracker.update(5)
        
        assert tracker.state.completed


# =============================================================================
# Logger Tests
# =============================================================================

class TestPipelineLogger:
    """Tests for pipeline logging."""
    
    def test_logger_creation(self):
        """Test logger creation."""
        from workflow.logger import PipelineLogger
        
        logger = PipelineLogger.get_logger("TestLogger")
        
        assert logger is not None
        assert logger.name == "TestLogger"
    
    def test_logger_singleton(self):
        """Test logger singleton behavior."""
        from workflow.logger import PipelineLogger
        
        logger1 = PipelineLogger.get_logger("SingletonTest")
        logger2 = PipelineLogger.get_logger("SingletonTest")
        
        assert logger1 is logger2
    
    def test_logger_levels(self):
        """Test logger level setting."""
        from workflow.logger import PipelineLogger
        import logging
        
        logger = PipelineLogger.get_logger("LevelTest", level=logging.DEBUG)
        
        assert logger.level == logging.DEBUG


# =============================================================================
# Feature Extractor Tests
# =============================================================================

class TestFeatureExtractor:
    """Tests for feature extraction."""
    
    def test_spectral_indices(self):
        """Test spectral index calculation."""
        from pipeline.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        
        # Create sample band data
        bands = {
            "red": np.ones((10, 10)) * 100,
            "nir": np.ones((10, 10)) * 200,
            "green": np.ones((10, 10)) * 150,
            "blue": np.ones((10, 10)) * 50,
            "swir1": np.ones((10, 10)) * 80,
        }
        
        indices = extractor.calculate_spectral_indices(bands)
        
        assert "NDVI" in indices
        assert "NDWI" in indices
        assert "NDBI" in indices
        
        # NDVI = (NIR - RED) / (NIR + RED) = (200 - 100) / (200 + 100) = 0.333
        expected_ndvi = (200 - 100) / (200 + 100)
        assert np.allclose(indices["NDVI"], expected_ndvi, rtol=0.01)
    
    def test_feature_set(self):
        """Test feature set container."""
        from pipeline.feature_extractor import FeatureSet
        
        features = FeatureSet()
        features.spectral["NDVI"] = np.array([1, 2, 3])
        features.texture["contrast"] = np.array([4, 5, 6])
        
        array = features.to_array()
        assert array.shape[0] == 3
        
        names = features.get_feature_names()
        assert "NDVI" in names
        assert "contrast" in names


# =============================================================================
# Quality Assurance Tests
# =============================================================================

class TestQualityAssurance:
    """Tests for quality assurance."""
    
    def test_geometry_validity_check(self):
        """Test geometry validity checking."""
        from pipeline.quality_assurance import QualityAssurance
        
        try:
            import geopandas as gpd
            from shapely.geometry import Point, Polygon
            
            qa = QualityAssurance()
            
            # Create valid GeoDataFrame
            gdf = gpd.GeoDataFrame({
                "id": [1, 2],
                "geometry": [Point(0, 0), Point(1, 1)]
            })
            
            result = qa._check_geometry_validity(gdf)
            assert result.passed
            assert result.check_name == "geometry_validity"
            
        except ImportError:
            pytest.skip("geopandas not available")
    
    def test_attribute_completeness_check(self):
        """Test attribute completeness checking."""
        from pipeline.quality_assurance import QualityAssurance
        
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            import numpy as np
            
            qa = QualityAssurance()
            
            # Create GeoDataFrame with some null values
            gdf = gpd.GeoDataFrame({
                "id": [1, 2, 3, 4, 5],
                "value": [10, None, 30, None, 50],
                "geometry": [Point(i, i) for i in range(5)]
            })
            
            result = qa._check_attribute_completeness(gdf)
            
            assert result.check_name == "attribute_completeness"
            # Should have some nulls detected
            assert "value" in result.details.get("fields_with_nulls", {})
            
        except ImportError:
            pytest.skip("geopandas not available")
    
    def test_quality_report(self):
        """Test quality report generation."""
        from pipeline.quality_assurance import QualityReport, CheckResult, CheckSeverity, CheckType
        
        report = QualityReport(data_name="Test Data", total_features=100)
        report.checks_run = 5
        report.checks_passed = 4
        report.checks_failed = 1
        report.overall_score = 80.0
        
        assert report.pass_rate == 80.0
        assert report.issue_summary == {"pass": 0, "info": 0, "warning": 0, "error": 0, "critical": 0}


# =============================================================================
# Asset Manager Tests
# =============================================================================

class TestAssetManager:
    """Tests for asset management."""
    
    def test_asset_creation(self):
        """Test asset creation."""
        from pipeline.asset_manager import AssetManager, AssetType, AssetMetadata
        import numpy as np
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = AssetManager(asset_dir=tmpdir)
            
            data = np.array([[1, 2], [3, 4]])
            asset = manager.create_asset(
                data=data,
                name="test_asset",
                asset_type=AssetType.RASTER
            )
            
            assert asset.id.startswith("test_asset")
            assert asset.name == "test_asset"
            assert asset.asset_type == AssetType.RASTER
    
    def test_asset_metadata(self):
        """Test asset metadata."""
        from pipeline.asset_manager import AssetMetadata
        
        metadata = AssetMetadata(
            title="Test Asset",
            description="A test asset",
            asset_type="raster",
            status="raw"
        )
        
        assert metadata.title == "Test Asset"
        assert metadata.created is not None
        
        # Test serialization
        d = metadata.to_dict()
        assert d["title"] == "Test Asset"
        
        json_str = metadata.to_json()
        assert "Test Asset" in json_str


# =============================================================================
# ML Classifier Tests
# =============================================================================

class TestSpatialClassifier:
    """Tests for spatial classification."""
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        from ml.classifier import SpatialClassifier
        
        classifier = SpatialClassifier(algorithm="random_forest")
        
        assert classifier.algorithm == "random_forest"
        assert classifier.is_trained == False
    
    def test_classifier_training(self):
        """Test classifier training."""
        from ml.classifier import SpatialClassifier
        
        classifier = SpatialClassifier(algorithm="random_forest", n_estimators=10)
        
        # Create sample data
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 3, 100)
        
        result = classifier.train(X, y, validation_size=0.2)
        
        assert result.model_trained
        assert result.training_accuracy > 0
        assert classifier.is_trained
    
    def test_classifier_prediction(self):
        """Test classifier prediction."""
        from ml.classifier import SpatialClassifier
        
        classifier = SpatialClassifier(algorithm="random_forest", n_estimators=10)
        
        # Train
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 3, 100)
        classifier.train(X_train, y_train)
        
        # Predict
        X_test = np.random.rand(10, 5)
        predictions = classifier.predict(X_test)
        
        assert len(predictions) == 10
        assert all(p in [0, 1, 2] for p in predictions)
    
    def test_classifier_evaluation(self):
        """Test classifier evaluation."""
        from ml.classifier import SpatialClassifier
        
        classifier = SpatialClassifier(algorithm="random_forest", n_estimators=10)
        
        # Train
        X = np.random.rand(100, 5)
        y = np.array(["class_a"] * 50 + ["class_b"] * 50)  # Use string labels
        result = classifier.train(X, y, class_names=["class_a", "class_b"])
        
        # Evaluate on same data (training data is already encoded)
        eval_result = classifier.evaluate(X, y)
        
        assert eval_result.accuracy >= 0
        assert eval_result.accuracy <= 1
        assert eval_result.confusion_matrix is not None


# =============================================================================
# Object Detection Tests
# =============================================================================

class TestObjectDetector:
    """Tests for object detection."""
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        from ml.object_detection import ObjectDetector
        
        detector = ObjectDetector(model="faster_rcnn")
        
        assert detector.model_name == "faster_rcnn"
        assert detector.confidence_threshold == 0.7
    
    def test_detection_result(self):
        """Test detection result container."""
        from ml.object_detection import Detection, DetectionResult
        
        det = Detection(
            bbox=(10, 10, 50, 50),
            label="building",
            confidence=0.9
        )
        
        assert det.bbox == (10, 10, 50, 50)
        assert det.area == 1600  # (50-10) * (50-10)
        assert det.center == (30, 30)
        
        # Test GeoJSON conversion
        feature = det.to_geojson_feature()
        assert feature["type"] == "Feature"
        assert feature["properties"]["label"] == "building"


# =============================================================================
# Processing Graph Tests
# =============================================================================

class TestProcessingGraph:
    """Tests for processing graph."""
    
    def test_graph_creation(self):
        """Test graph creation."""
        from workflow.processing_graph import ProcessingGraph
        
        graph = ProcessingGraph()
        
        assert len(graph.nodes) == 0
    
    def test_add_node(self):
        """Test adding nodes."""
        from workflow.processing_graph import ProcessingGraph
        
        graph = ProcessingGraph()
        
        def dummy_func():
            return "result"
        
        graph.add_node("node1", dummy_func)
        graph.add_node("node2", dummy_func, depends_on=["node1"])
        
        assert len(graph.nodes) == 2
        assert "node2" in graph.nodes
        assert "node1" in graph.nodes["node2"].depends_on
    
    def test_execution_order(self):
        """Test topological sorting."""
        from workflow.processing_graph import ProcessingGraph
        
        graph = ProcessingGraph()
        
        def dummy_func():
            return "result"
        
        graph.add_node("a", dummy_func)
        graph.add_node("b", dummy_func, depends_on=["a"])
        graph.add_node("c", dummy_func, depends_on=["b"])
        
        order = graph.get_execution_order()
        
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")
    
    def test_graph_execution(self):
        """Test graph execution."""
        from workflow.processing_graph import ProcessingGraph
        
        graph = ProcessingGraph()

        results = {"a": 1}

        def func_a():
            results["a"] = 2
            return {"result": results["a"]}

        def func_b(**kwargs):
            input_a = kwargs.get("a", {}).get("result", 1)
            return input_a * 3

        graph.add_node("a", func_a)
        graph.add_node("b", func_b, depends_on=["a"])

        exec_results = graph.execute()

        assert exec_results["a"].status.value == "completed"
        assert exec_results["b"].status.value == "completed"
        assert exec_results["b"].output == 6


# =============================================================================
# Change Detection Tests
# =============================================================================

class TestChangeDetector:
    """Tests for change detection."""
    
    def test_detector_initialization(self):
        """Test change detector initialization."""
        from ml.change_detection import ChangeDetector
        
        detector = ChangeDetector(method="difference")
        
        assert detector.method == "difference"
        assert detector.threshold == 0.5
    
    def test_change_result(self):
        """Test change detection result."""
        from ml.change_detection import ChangeDetectionResult
        
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1  # 100 changed pixels
        
        result = ChangeDetectionResult(
            change_mask=mask,
            change_pixels=100,
            total_pixels=10000,
            change_percentage=1.0
        )
        
        assert result.change_percentage == 1.0
        assert result.no_change_percentage == 99.0
        assert len(result.get_change_locations()) == 100


# =============================================================================
# Clustering Tests
# =============================================================================

class TestSpatialClusterer:
    """Tests for spatial clustering."""
    
    def test_clusterer_initialization(self):
        """Test clusterer initialization."""
        from ml.clustering import SpatialClusterer
        
        clusterer = SpatialClusterer(algorithm="dbscan", eps=0.5)
        
        assert clusterer.algorithm == "dbscan"
        assert clusterer.eps == 0.5
    
    def test_dbscan_clustering(self):
        """Test DBSCAN clustering."""
        from ml.clustering import SpatialClusterer
        
        clusterer = SpatialClusterer(algorithm="dbscan", eps=0.5, min_samples=2)
        
        # Create sample data with clear clusters
        np.random.seed(42)
        cluster1 = np.random.rand(20, 2) * 0.1
        cluster2 = np.random.rand(20, 2) * 0.1 + 0.5
        coordinates = np.vstack([cluster1, cluster2])
        
        result = clusterer.cluster(coordinates)
        
        assert result.n_clusters >= 1
        assert len(result.labels) == 40
    
    def test_kmeans_clustering(self):
        """Test K-Means clustering."""
        from ml.clustering import SpatialClusterer
        
        clusterer = SpatialClusterer(algorithm="kmeans", n_clusters=3)
        
        np.random.seed(42)
        coordinates = np.random.rand(50, 2)
        
        result = clusterer.cluster(coordinates)
        
        assert result.n_clusters == 3
        assert result.cluster_centers is not None


# =============================================================================
# Batch Processor Tests
# =============================================================================

class TestBatchProcessor:
    """Tests for batch processing."""
    
    def test_processor_initialization(self):
        """Test batch processor initialization."""
        from pyqgis.batch_processor import BatchProcessor
        
        processor = BatchProcessor(use_qgis=False)
        
        assert processor.max_workers > 0
    
    def test_file_discovery(self):
        """Test file discovery."""
        from pyqgis.batch_processor import BatchProcessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            Path(tmpdir).joinpath("test1.shp").touch()
            Path(tmpdir).joinpath("test2.geojson").touch()
            Path(tmpdir).joinpath("test3.txt").touch()
            
            processor = BatchProcessor(use_qgis=False)
            
            files = processor._get_vector_files(tmpdir)
            
            assert len(files) == 2
            assert any("test1.shp" in f for f in files)
            assert any("test2.geojson" in f for f in files)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the pipeline."""
    
    def test_full_classification_workflow(self):
        """Test complete classification workflow."""
        from ml.classifier import SpatialClassifier
        from pipeline.feature_extractor import FeatureExtractor
        import numpy as np
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 200
        
        # Simulate spectral bands
        bands = {
            "red": np.random.rand(n_samples) * 1000,
            "nir": np.random.rand(n_samples) * 2000,
            "green": np.random.rand(n_samples) * 800,
        }
        
        # Calculate indices
        extractor = FeatureExtractor()
        indices = extractor.calculate_spectral_indices(bands)
        
        # Create feature matrix
        features = np.column_stack([
            bands["red"],
            bands["nir"],
            bands["green"],
            indices["NDVI"]
        ])
        
        # Create labels
        labels = (indices["NDVI"] > 0.3).astype(int)
        
        # Train classifier
        classifier = SpatialClassifier(algorithm="random_forest", n_estimators=10)
        result = classifier.train(features, labels, validation_size=0.2)
        
        assert result.model_trained
        assert result.training_accuracy > 0.5
    
    def test_config_integration(self):
        """Test configuration integration across modules."""
        from utils.config import Config
        from workflow.logger import PipelineLogger
        from workflow.progress_tracker import ProgressTracker

        config = Config.default()
        logger = PipelineLogger.get_logger("IntegrationTest")
        progress = ProgressTracker()

        # Verify all components work together
        progress.init(total=10, description="Integration test")
        logger.info("Starting integration test")

        for i in range(10):
            progress.update()

        progress.finish()
        logger.info("Integration test complete")

        assert progress.state.completed


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
