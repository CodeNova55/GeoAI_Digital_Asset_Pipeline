"""
Sample Use Cases for GeoAI Digital Asset Pipeline
=================================================

This module contains sample use cases demonstrating the capabilities
of the GeoAI pipeline for real-world geospatial analysis tasks.

Use Cases:
1. Land Cover Classification from Satellite Imagery
2. Automated Building Footprint Extraction
3. Change Detection Between Time Periods
4. Spatial Pattern Analysis with ML Clustering
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from workflow.logger import PipelineLogger, setup_logging
from workflow.progress_tracker import ProgressTracker


# =============================================================================
# Use Case 1: Land Cover Classification from Satellite Imagery
# =============================================================================

class LandCoverClassification:
    """
    Land cover classification use case.
    
    This use case demonstrates how to classify satellite imagery
    into land cover categories using machine learning.
    
    Workflow:
    1. Load Sentinel-2 or Landsat imagery
    2. Calculate spectral indices (NDVI, NDWI, NDBI, etc.)
    3. Train or load a classification model
    4. Predict land cover classes
    5. Export results and generate accuracy report
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.default()
        self.logger = PipelineLogger.get_logger("LandCoverClassification")
        self.progress = ProgressTracker()
    
    def run(
        self,
        input_raster: str,
        training_data: Optional[str] = None,
        output_path: str = "outputs/land_cover.tif",
        model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run land cover classification.
        
        Args:
            input_raster: Path to input satellite imagery.
            training_data: Optional path to training data.
            output_path: Path for output classification.
            model_path: Optional path to pre-trained model.
            
        Returns:
            Dictionary with results and statistics.
        """
        self.logger.info("=" * 60)
        self.logger.info("Use Case 1: Land Cover Classification")
        self.logger.info("=" * 60)
        
        results = {
            "use_case": "land_cover_classification",
            "input": input_raster,
            "output": output_path,
            "status": "success"
        }
        
        try:
            # Step 1: Load and prepare data
            self.logger.info("Step 1: Loading input data...")
            self.progress.init(total=5, desc="Classification")
            
            # Step 2: Calculate spectral indices
            self.logger.info("Step 2: Calculating spectral indices...")
            from pipeline.feature_extractor import FeatureExtractor
            
            extractor = FeatureExtractor(config=self.config, logger=self.logger)
            bands = extractor._load_bands(input_raster)
            indices = extractor.calculate_spectral_indices(bands)
            
            self.progress.update()
            
            # Step 3: Prepare features
            self.logger.info("Step 3: Preparing feature matrix...")
            feature_matrix = self._prepare_features(bands, indices)
            
            self.progress.update()
            
            # Step 4: Load or train model
            self.logger.info("Step 4: Loading classification model...")
            from ml.classifier import LandCoverClassifier
            
            classifier = LandCoverClassifier(config=self.config, logger=self.logger)
            
            if model_path and Path(model_path).exists():
                classifier.load_model(model_path)
            else:
                # Use default pretrained model
                classifier.load_pretrained("land_cover")
            
            self.progress.update()
            
            # Step 5: Classify
            self.logger.info("Step 5: Classifying pixels...")
            predictions = classifier.predict(feature_matrix)
            
            self.progress.update()
            
            # Step 6: Export results
            self.logger.info("Step 6: Exporting results...")
            self._export_classification(predictions, output_path, input_raster)
            
            self.progress.update()
            
            # Calculate statistics
            results["statistics"] = self._calculate_statistics(predictions, classifier.class_names)
            
            self.progress.finish()
            
            self.logger.info(f"Classification complete. Output: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Classification failed: {e}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)
        
        return results
    
    def _prepare_features(
        self,
        bands: Dict[str, np.ndarray],
        indices: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Prepare feature matrix from bands and indices."""
        features = []
        
        # Add spectral bands
        for band_name, band_data in bands.items():
            features.append(band_data.flatten())
        
        # Add spectral indices
        for index_name, index_data in indices.items():
            features.append(index_data.flatten())
        
        return np.column_stack(features)
    
    def _export_classification(
        self,
        predictions: np.ndarray,
        output_path: str,
        reference_raster: str
    ) -> None:
        """Export classification results as raster."""
        try:
            import rasterio
            
            # Get reference metadata
            with rasterio.open(reference_raster) as src:
                profile = src.profile.copy()
                profile.update({
                    'driver': 'GTiff',
                    'dtype': rasterio.uint8,
                    'count': 1,
                    'compress': 'lzw'
                })
            
            # Reshape predictions to image shape
            with rasterio.open(reference_raster) as src:
                shape = (src.height, src.width)
            
            classification = predictions.reshape(shape).astype(np.uint8)
            
            # Write output
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(classification, 1)
                
        except ImportError:
            self.logger.warning("rasterio not available, skipping export")
    
    def _calculate_statistics(
        self,
        predictions: np.ndarray,
        class_names: List[str]
    ) -> Dict[str, Any]:
        """Calculate classification statistics."""
        unique, counts = np.unique(predictions, return_counts=True)
        total = len(predictions)
        
        stats = {
            "total_pixels": int(total),
            "classes": {}
        }
        
        for class_id, count in zip(unique, counts):
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            stats["classes"][class_name] = {
                "pixels": int(count),
                "percentage": float(count / total * 100)
            }
        
        return stats


# =============================================================================
# Use Case 2: Automated Building Footprint Extraction
# =============================================================================

class BuildingFootprintExtraction:
    """
    Building footprint extraction use case.
    
    This use case demonstrates automated extraction of building
    footprints from aerial or satellite imagery using deep learning.
    
    Workflow:
    1. Load high-resolution imagery
    2. Run object detection or segmentation model
    3. Post-process detections (filter, merge)
    4. Vectorize building footprints
    5. Export as GeoJSON or Shapefile
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.default()
        self.logger = PipelineLogger.get_logger("BuildingExtraction")
        self.progress = ProgressTracker()
    
    def run(
        self,
        input_image: str,
        output_path: str = "outputs/buildings.geojson",
        model_path: Optional[str] = None,
        min_area: float = 20.0,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Run building footprint extraction.
        
        Args:
            input_image: Path to input imagery.
            output_path: Path for output vector file.
            model_path: Optional path to model weights.
            min_area: Minimum building area in square meters.
            confidence_threshold: Detection confidence threshold.
            
        Returns:
            Dictionary with results.
        """
        self.logger.info("=" * 60)
        self.logger.info("Use Case 2: Building Footprint Extraction")
        self.logger.info("=" * 60)
        
        results = {
            "use_case": "building_extraction",
            "input": input_image,
            "output": output_path,
            "status": "success"
        }
        
        try:
            self.progress.init(total=4, desc="Building extraction")
            
            # Step 1: Initialize detector
            self.logger.info("Step 1: Loading detection model...")
            from ml.object_detection import BuildingFootprintDetector
            
            detector = BuildingFootprintDetector(
                config=self.config,
                logger=self.logger,
                confidence_threshold=confidence_threshold
            )
            
            if model_path and Path(model_path).exists():
                detector.load_weights(model_path)
            else:
                detector.load_pretrained("buildings")
            
            self.progress.update()
            
            # Step 2: Detect buildings
            self.logger.info("Step 2: Detecting buildings...")
            detection_result = detector.detect(input_image)
            
            self.progress.update()
            
            # Step 3: Filter by area
            self.logger.info("Step 3: Filtering detections...")
            original_count = detection_result.count
            detection_result.detections = [
                d for d in detection_result.detections
                if d.area >= min_area
            ]
            filtered_count = detection_result.count
            
            self.logger.info(f"Filtered from {original_count} to {filtered_count} buildings")
            
            self.progress.update()
            
            # Step 4: Export results
            self.logger.info("Step 4: Exporting footprints...")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            detection_result.save_geojson(output_path)
            
            self.progress.update()
            self.progress.finish()
            
            # Compile results
            results["buildings_detected"] = filtered_count
            results["filtered_out"] = original_count - filtered_count
            results["by_confidence"] = detection_result.by_class
            
            self.logger.info(f"Extraction complete. Found {filtered_count} buildings.")
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)
        
        return results


# =============================================================================
# Use Case 3: Change Detection Between Time Periods
# =============================================================================

class ChangeDetectionAnalysis:
    """
    Change detection use case.
    
    This use case demonstrates detection of changes between
    multi-temporal satellite imagery for monitoring applications.
    
    Workflow:
    1. Load before and after imagery
    2. Preprocess (co-register, normalize)
    3. Run change detection algorithm
    4. Post-process change mask
    5. Quantify and characterize changes
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.default()
        self.logger = PipelineLogger.get_logger("ChangeDetection")
        self.progress = ProgressTracker()
    
    def run(
        self,
        image_before: str,
        image_after: str,
        output_path: str = "outputs/change_mask.tif",
        method: str = "siamese_network",
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Run change detection analysis.
        
        Args:
            image_before: Path to earlier image.
            image_after: Path to later image.
            output_path: Path for change mask output.
            method: Change detection method.
            threshold: Change threshold.
            
        Returns:
            Dictionary with results.
        """
        self.logger.info("=" * 60)
        self.logger.info("Use Case 3: Change Detection Analysis")
        self.logger.info("=" * 60)
        
        results = {
            "use_case": "change_detection",
            "image_before": image_before,
            "image_after": image_after,
            "output": output_path,
            "method": method,
            "status": "success"
        }
        
        try:
            self.progress.init(total=4, desc="Change detection")
            
            # Step 1: Initialize detector
            self.logger.info(f"Step 1: Initializing {method}...")
            from ml.change_detection import ChangeDetector
            
            detector = ChangeDetector(
                method=method,
                threshold=threshold,
                config=self.config,
                logger=self.logger
            )
            
            self.progress.update()
            
            # Step 2: Detect changes
            self.logger.info("Step 2: Detecting changes...")
            change_result = detector.detect_change(image_before, image_after)
            
            self.progress.update()
            
            # Step 3: Export change mask
            self.logger.info("Step 3: Exporting change mask...")
            self._export_change_mask(change_result, output_path, image_before)
            
            self.progress.update()
            
            # Step 4: Analyze changes
            self.logger.info("Step 4: Analyzing changes...")
            results["analysis"] = self._analyze_changes(change_result, image_before)
            
            self.progress.update()
            self.progress.finish()
            
            results["change_percentage"] = change_result.change_percentage
            results["changed_pixels"] = change_result.change_pixels
            
            self.logger.info(
                f"Change detection complete. "
                f"{change_result.change_percentage:.2f}% area changed."
            )
            
        except Exception as e:
            self.logger.error(f"Change detection failed: {e}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)
        
        return results
    
    def _export_change_mask(
        self,
        result,
        output_path: str,
        reference_image: str
    ) -> None:
        """Export change mask as raster."""
        try:
            from PIL import Image
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save as PNG
            png_path = str(Path(output_path).with_suffix('.png'))
            change_img = Image.fromarray(
                (result.change_mask * 255).astype(np.uint8)
            )
            change_img.save(png_path)
            
        except Exception as e:
            self.logger.warning(f"Could not export change mask: {e}")
    
    def _analyze_changes(
        self,
        result,
        reference_image: str
    ) -> Dict[str, Any]:
        """Analyze detected changes."""
        analysis = {
            "total_change_pixels": result.change_pixels,
            "change_percentage": result.change_percentage,
            "change_locations": len(result.get_change_locations())
        }
        
        # Try to get spatial information
        try:
            import rasterio
            with rasterio.open(reference_image) as src:
                pixel_area = src.transform[0] * abs(src.transform[4])
                analysis["change_area_sqm"] = result.change_pixels * pixel_area
                analysis["total_area_sqm"] = result.total_pixels * pixel_area
        except:
            pass
        
        return analysis


# =============================================================================
# Use Case 4: Spatial Pattern Analysis with ML Clustering
# =============================================================================

class SpatialPatternAnalysis:
    """
    Spatial pattern analysis use case.
    
    This use case demonstrates clustering and pattern analysis
    for understanding spatial distributions and relationships.
    
    Workflow:
    1. Load spatial data with attributes
    2. Extract spatial and attribute features
    3. Run clustering algorithm
    4. Analyze cluster characteristics
    5. Map and visualize patterns
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.default()
        self.logger = PipelineLogger.get_logger("SpatialPatternAnalysis")
        self.progress = ProgressTracker()
    
    def run(
        self,
        input_data: str,
        output_path: str = "outputs/clusters.geojson",
        algorithm: str = "dbscan",
        n_clusters: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run spatial pattern analysis.
        
        Args:
            input_data: Path to input vector data.
            output_path: Path for output with cluster labels.
            algorithm: Clustering algorithm.
            n_clusters: Number of clusters (for K-Means).
            
        Returns:
            Dictionary with results.
        """
        self.logger.info("=" * 60)
        self.logger.info("Use Case 4: Spatial Pattern Analysis")
        self.logger.info("=" * 60)
        
        results = {
            "use_case": "spatial_clustering",
            "input": input_data,
            "output": output_path,
            "algorithm": algorithm,
            "status": "success"
        }
        
        try:
            self.progress.init(total=5, desc="Pattern analysis")
            
            # Step 1: Load data
            self.logger.info("Step 1: Loading spatial data...")
            import geopandas as gpd
            
            gdf = gpd.read_file(input_data)
            self.logger.info(f"Loaded {len(gdf)} features")
            
            self.progress.update()
            
            # Step 2: Extract coordinates and features
            self.logger.info("Step 2: Extracting features...")
            coordinates = self._extract_coordinates(gdf)
            attributes = self._extract_attributes(gdf)
            
            self.progress.update()
            
            # Step 3: Run clustering
            self.logger.info(f"Step 3: Running {algorithm} clustering...")
            from ml.clustering import SpatialClusterer
            
            clusterer = SpatialClusterer(
                algorithm=algorithm,
                n_clusters=n_clusters,
                config=self.config,
                logger=self.logger
            )
            
            clustering_result = clusterer.cluster(coordinates, attributes)
            
            self.progress.update()
            
            # Step 4: Add cluster labels to data
            self.logger.info("Step 4: Adding cluster labels...")
            gdf['cluster_id'] = clustering_result.labels
            gdf['is_noise'] = clustering_result.labels == -1
            
            self.progress.update()
            
            # Step 5: Export results
            self.logger.info("Step 5: Exporting results...")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(output_path, driver="GeoJSON")
            
            self.progress.update()
            self.progress.finish()
            
            # Compile results
            results["n_clusters"] = clustering_result.n_clusters
            results["n_noise"] = clustering_result.n_noise
            results["noise_percentage"] = clustering_result.noise_percentage
            results["cluster_sizes"] = clustering_result.cluster_sizes
            results["silhouette_score"] = clustering_result.silhouette_score
            
            # Spatial autocorrelation
            if len(gdf) > 10:
                autocorr = clusterer.spatial_autocorrelation(
                    coordinates,
                    gdf[gdf.columns[0]].values if len(gdf.columns) > 0 else np.ones(len(gdf))
                )
                results["spatial_autocorrelation"] = autocorr
            
            self.logger.info(
                f"Pattern analysis complete. "
                f"Found {clustering_result.n_clusters} clusters."
            )
            
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)
        
        return results
    
    def _extract_coordinates(self, gdf) -> np.ndarray:
        """Extract coordinates from GeoDataFrame."""
        centroids = gdf.geometry.centroid
        return np.column_stack([centroids.x.values, centroids.y.values])
    
    def _extract_attributes(self, gdf) -> Optional[np.ndarray]:
        """Extract numeric attributes for clustering."""
        numeric_cols = gdf.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            return gdf[numeric_cols].values
        return None


# =============================================================================
# Main execution
# =============================================================================

def run_all_use_cases(
    data_dir: str = "data/raw",
    output_dir: str = "outputs/use_cases"
) -> Dict[str, Dict[str, Any]]:
    """
    Run all use cases with sample data.
    
    Args:
        data_dir: Directory containing sample data.
        output_dir: Directory for outputs.
        
    Returns:
        Dictionary with results from all use cases.
    """
    config = Config.default()
    results = {}
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Note: These use cases require actual data files
    # In production, you would download or generate sample data
    
    sample_data = {
        "satellite_image": f"{data_dir}/sentinel2_sample.tif",
        "aerial_image": f"{data_dir}/aerial_sample.tif",
        "image_before": f"{data_dir}/before.tif",
        "image_after": f"{data_dir}/after.tif",
        "vector_data": f"{data_dir}/sample_data.shp"
    }
    
    # Use Case 1: Land Cover Classification
    if Path(sample_data["satellite_image"]).exists():
        uc1 = LandCoverClassification(config)
        results["land_cover"] = uc1.run(
            input_raster=sample_data["satellite_image"],
            output_path=f"{output_dir}/land_cover.tif"
        )
    
    # Use Case 2: Building Extraction
    if Path(sample_data["aerial_image"]).exists():
        uc2 = BuildingFootprintExtraction(config)
        results["buildings"] = uc2.run(
            input_image=sample_data["aerial_image"],
            output_path=f"{output_dir}/buildings.geojson"
        )
    
    # Use Case 3: Change Detection
    if Path(sample_data["image_before"]).exists() and Path(sample_data["image_after"]).exists():
        uc3 = ChangeDetectionAnalysis(config)
        results["change_detection"] = uc3.run(
            image_before=sample_data["image_before"],
            image_after=sample_data["image_after"],
            output_path=f"{output_dir}/change_mask.tif"
        )
    
    # Use Case 4: Spatial Pattern Analysis
    if Path(sample_data["vector_data"]).exists():
        uc4 = SpatialPatternAnalysis(config)
        results["spatial_clustering"] = uc4.run(
            input_data=sample_data["vector_data"],
            output_path=f"{output_dir}/clusters.geojson"
        )
    
    return results


if __name__ == "__main__":
    setup_logging(level="INFO")
    
    print("\n" + "=" * 60)
    print("GeoAI Digital Asset Pipeline - Sample Use Cases")
    print("=" * 60 + "\n")
    
    results = run_all_use_cases()
    
    print("\n" + "=" * 60)
    print("Use Case Results Summary")
    print("=" * 60)
    
    for use_case, result in results.items():
        status = result.get("status", "unknown")
        print(f"\n{use_case}: {status.upper()}")
        if status == "success":
            for key, value in result.items():
                if key not in ["use_case", "status", "input", "output"]:
                    print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
