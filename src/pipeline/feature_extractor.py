"""
Feature Extractor Module
========================

Provides automated feature extraction from geospatial data.
Supports spectral indices, texture features, geometric features,
and contextual features for ML model input.

Example:
    >>> extractor = FeatureExtractor()
    >>> features = extractor.extract(raster_path, vector_path)
    >>> indices = extractor.calculate_spectral_indices(bands)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import logging
import json

import numpy as np

try:
    from osgeo import gdal, ogr, osr
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

try:
    import rasterio
    from rasterio.features import zonal_stats
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

from src.utils.config import Config
from src.workflow.logger import PipelineLogger


@dataclass
class FeatureSet:
    """Container for extracted features."""
    
    spectral: Dict[str, np.ndarray] = field(default_factory=dict)
    texture: Dict[str, np.ndarray] = field(default_factory=dict)
    geometric: Dict[str, np.ndarray] = field(default_factory=dict)
    contextual: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def to_array(self) -> np.ndarray:
        """Convert all features to a single array."""
        all_features = []
        
        for features in [self.spectral, self.texture, self.geometric, self.contextual]:
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    all_features.append(value.flatten())
        
        if all_features:
            return np.column_stack(all_features)
        return np.array([])
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        names = []
        for features in [self.spectral, self.texture, self.geometric, self.contextual]:
            names.extend(features.keys())
        return names
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "spectral": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.spectral.items()},
            "texture": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.texture.items()},
            "geometric": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.geometric.items()},
            "contextual": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.contextual.items()}
        }
    
    def save(self, path: str) -> bool:
        """Save features to file."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error saving features: {e}")
            return False


class FeatureExtractor:
    """
    Automated feature extraction from geospatial data.
    
    This class provides comprehensive feature extraction capabilities
    for machine learning models, including spectral indices, texture
    analysis, geometric properties, and contextual features.
    
    Attributes:
        config: Configuration object with extraction parameters.
        logger: Logger instance.
        enabled_features: List of enabled feature types.
        
    Example:
        >>> extractor = FeatureExtractor()
        >>> features = extractor.extract_all(
        ...     raster_path="satellite.tif",
        ...     vector_path="polygons.shp"
        ... )
        >>> print(features.get_feature_names())
    """
    
    # Spectral index formulas
    SPECTRAL_INDICES = {
        "NDVI": lambda b: (b["nir"] - b["red"]) / (b["nir"] + b["red"] + 1e-10),
        "NDWI": lambda b: (b["green"] - b["nir"]) / (b["green"] + b["nir"] + 1e-10),
        "NDBI": lambda b: (b["swir1"] - b["nir"]) / (b["swir1"] + b["nir"] + 1e-10),
        "EVI": lambda b: 2.5 * (b["nir"] - b["red"]) / (b["nir"] + 6 * b["red"] - 7.5 * b["blue"] + 10001),
        "SAVI": lambda b: 1.5 * (b["nir"] - b["red"]) / (b["nir"] + b["red"] + 0.5),
        "MNDWI": lambda b: (b["green"] - b["swir1"]) / (b["green"] + b["swir1"] + 1e-10),
        "BAI": lambda b: 1 / ((0.1 - b["red"]) ** 2 + (0.06 - b["nir"]) ** 2 + 1e-10),
        "SI": lambda b: (b["swir1"] + b["red"]) / (b["nir"] + b["red"] + 1e-10),
    }
    
    def __init__(
        self,
        config: Optional[Config] = None,
        logger: Optional[PipelineLogger] = None,
        enabled_features: Optional[List[str]] = None
    ):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration object.
            logger: Logger instance.
            enabled_features: List of feature types to extract.
        """
        self.config = config or Config.default()
        self.logger = logger or PipelineLogger.get_logger("FeatureExtractor")
        
        # Load config
        pipeline_config = self.config.pipeline if hasattr(self.config, 'pipeline') else {}
        fe_config = pipeline_config.get('feature_extraction', {})
        
        self.enabled_features = enabled_features or fe_config.get('enabled_features', [
            "spectral_indices", "texture_features", "geometric_features"
        ])
        
        self.logger.info(f"FeatureExtractor initialized with features: {self.enabled_features}")
    
    def extract_all(
        self,
        raster_path: str,
        vector_path: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> FeatureSet:
        """
        Extract all enabled features.
        
        Args:
            raster_path: Path to raster data.
            vector_path: Optional path to vector data for zonal stats.
            output_path: Optional path to save features.
            
        Returns:
            FeatureSet with all extracted features.
        """
        features = FeatureSet()
        
        # Load raster bands
        bands = self._load_bands(raster_path)
        
        if "spectral_indices" in self.enabled_features:
            features.spectral = self.calculate_spectral_indices(bands)
        
        if "texture_features" in self.enabled_features:
            features.texture = self.calculate_texture_features(bands)
        
        if vector_path and "geometric_features" in self.enabled_features:
            features.geometric = self.calculate_geometric_features(vector_path)
        
        if vector_path and "contextual_features" in self.enabled_features:
            features.contextual = self.calculate_contextual_features(raster_path, vector_path)
        
        if output_path:
            features.save(output_path)
        
        return features
    
    def _load_bands(self, raster_path: str) -> Dict[str, np.ndarray]:
        """Load raster bands from file."""
        bands = {}
        
        if RASTERIO_AVAILABLE:
            with rasterio.open(raster_path) as src:
                # Try to identify bands by name
                band_names = src.descriptions if src.descriptions else [f"band_{i+1}" for i in range(src.count)]
                
                for i, name in enumerate(band_names, 1):
                    band_data = src.read(i).astype(float)
                    
                    # Map common band names
                    name_lower = name.lower()
                    if "blue" in name_lower or "b02" in name_lower:
                        bands["blue"] = band_data
                    elif "green" in name_lower or "b03" in name_lower:
                        bands["green"] = band_data
                    elif "red" in name_lower or "b04" in name_lower:
                        bands["red"] = band_data
                    elif "nir" in name_lower or "b08" in name_lower:
                        bands["nir"] = band_data
                    elif "swir1" in name_lower or "b11" in name_lower:
                        bands["swir1"] = band_data
                    elif "swir2" in name_lower or "b12" in name_lower:
                        bands["swir2"] = band_data
                    else:
                        bands[f"band_{i}"] = band_data
        
        elif GDAL_AVAILABLE:
            ds = gdal.Open(raster_path)
            if ds:
                for i in range(ds.RasterCount):
                    band = ds.GetRasterBand(i + 1)
                    bands[f"band_{i+1}"] = band.ReadAsArray().astype(float)
        
        return bands
    
    def calculate_spectral_indices(
        self,
        bands: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate spectral indices from bands.
        
        Args:
            bands: Dictionary of band arrays.
            
        Returns:
            Dictionary of spectral index arrays.
        """
        indices = {}
        
        # Get configured indices
        pipeline_config = self.config.pipeline if hasattr(self.config, 'pipeline') else {}
        fe_config = pipeline_config.get('feature_extraction', {})
        index_list = fe_config.get('spectral_indices', list(self.SPECTRAL_INDICES.keys()))
        
        for index_name in index_list:
            if index_name in self.SPECTRAL_INDICES:
                try:
                    indices[index_name] = self.SPECTRAL_INDICES[index_name](bands)
                    self.logger.debug(f"Calculated {index_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to calculate {index_name}: {e}")
        
        return indices
    
    def calculate_texture_features(
        self,
        bands: Dict[str, np.ndarray],
        window_size: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Calculate texture features using GLCM.
        
        Args:
            bands: Dictionary of band arrays.
            window_size: Window size for texture calculation.
            
        Returns:
            Dictionary of texture feature arrays.
        """
        textures = {}
        
        # Use NIR or first available band
        band_data = bands.get("nir") or list(bands.values())[0]
        
        try:
            from skimage.feature import graycomatrix, graycoprops
            
            # Normalize to 0-255
            normalized = ((band_data - band_data.min()) / (band_data.max() - band_data.min()) * 255).astype(np.uint8)
            
            # Calculate GLCM
            glcm = graycomatrix(
                normalized,
                distances=[1],
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=256,
                symmetric=True,
                normed=True
            )
            
            # Extract texture properties
            textures["contrast"] = graycoprops(glcm, 'contrast').mean(axis=(0, 1))
            textures["dissimilarity"] = graycoprops(glcm, 'dissimilarity').mean(axis=(0, 1))
            textures["homogeneity"] = graycoprops(glcm, 'homogeneity').mean(axis=(0, 1))
            textures["energy"] = graycoprops(glcm, 'energy').mean(axis=(0, 1))
            textures["correlation"] = graycoprops(glcm, 'correlation').mean(axis=(0, 1))
            textures["ASM"] = graycoprops(glcm, 'ASM').mean(axis=(0, 1))
            
        except ImportError:
            self.logger.warning("scikit-image not available, using simple texture metrics")
            
            # Simple texture metrics using local variance
            from scipy.ndimage import uniform_filter
            
            mean_filter = uniform_filter(band_data, size=window_size)
            variance = uniform_filter(band_data ** 2, size=window_size) - mean_filter ** 2
            
            textures["local_variance"] = variance
            textures["local_mean"] = mean_filter
        
        return textures
    
    def calculate_geometric_features(
        self,
        vector_path: str
    ) -> Dict[str, np.ndarray]:
        """
        Calculate geometric features from vector data.
        
        Args:
            vector_path: Path to vector file.
            
        Returns:
            Dictionary of geometric feature arrays.
        """
        geometric = {}
        
        try:
            import geopandas as gpd
            from shapely.geometry import Polygon
            
            gdf = gpd.read_file(vector_path)
            
            # Area
            geometric["area"] = gdf.geometry.area.values
            
            # Perimeter
            geometric["perimeter"] = gdf.geometry.length.values
            
            # Compactness (4 * pi * area / perimeter^2)
            geometric["compactness"] = (
                4 * np.pi * gdf.geometry.area / (gdf.geometry.length ** 2 + 1e-10)
            )
            
            # Elongation (using minimum rotated rectangle)
            def get_elongation(geom):
                if geom.geom_type != 'Polygon':
                    return 0
                try:
                    min_rect = geom.minimum_rotated_rectangle
                    coords = list(min_rect.exterior.coords)[:4]
                    edges = [np.sqrt((coords[i][0]-coords[i+1][0])**2 + (coords[i][1]-coords[i+1][1])**2) for i in range(3)]
                    return max(edges) / (min(edges) + 1e-10)
                except:
                    return 0
            
            geometric["elongation"] = np.array([get_elongation(geom) for geom in gdf.geometry])
            
            # Convex hull ratio
            def get_convex_ratio(geom):
                if geom.geom_type != 'Polygon':
                    return 0
                return geom.area / (geom.convex_hull.area + 1e-10)
            
            geometric["convex_ratio"] = np.array([get_convex_ratio(geom) for geom in gdf.geometry])
            
        except ImportError:
            self.logger.warning("geopandas not available for geometric features")
        
        return geometric
    
    def calculate_contextual_features(
        self,
        raster_path: str,
        vector_path: str
    ) -> Dict[str, np.ndarray]:
        """
        Calculate contextual features based on neighborhood.
        
        Args:
            raster_path: Path to raster data.
            vector_path: Path to vector data.
            
        Returns:
            Dictionary of contextual feature arrays.
        """
        contextual = {}
        
        try:
            import geopandas as gpd
            from scipy import ndimage
            
            gdf = gpd.read_file(vector_path)
            
            # Load raster for context
            if RASTERIO_AVAILABLE:
                with rasterio.open(raster_path) as src:
                    raster_data = src.read(1)
                    
                    # Calculate distance to edge
                    edges = np.zeros_like(raster_data, dtype=bool)
                    edges[1:-1, 1:-1] = np.abs(np.gradient(raster_data))[0] > 0.1
                    
                    distance_transform = ndimage.distance_transform_edt(~edges)
                    contextual["distance_to_edge"] = distance_transform
                    
                    # Calculate local diversity (if classification exists)
                    unique_classes = len(np.unique(raster_data))
                    if unique_classes > 1:
                        # Neighborhood diversity
                        from skimage.measure import label
                        labeled = label(raster_data)
                        contextual["n_patches"] = np.array([labeled.max()])
        
        except ImportError:
            self.logger.warning("Required libraries not available for contextual features")
        
        return contextual
    
    def extract_zonal_statistics(
        self,
        raster_path: str,
        vector_path: str,
        stats: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract zonal statistics from raster for vector polygons.
        
        Args:
            raster_path: Path to raster data.
            vector_path: Path to vector data.
            stats: List of statistics to calculate.
            
        Returns:
            DataFrame with zonal statistics.
        """
        import pandas as pd
        
        if not RASTERIO_AVAILABLE:
            self.logger.error("rasterio not available for zonal statistics")
            return pd.DataFrame()
        
        if stats is None:
            stats = ['mean', 'min', 'max', 'std', 'sum', 'count']
        
        try:
            gdf = gpd.read_file(vector_path)
            
            with rasterio.open(raster_path) as src:
                results = []
                for idx, geom in enumerate(gdf.geometry):
                    try:
                        stats_result = zonal_stats(
                            geom,
                            src.read(1),
                            affine=src.transform,
                            stats=stats,
                            nodata=src.nodata
                        )
                        results.append(stats_result[0] if stats_result else {})
                    except Exception as e:
                        self.logger.warning(f"Zonal stats failed for feature {idx}: {e}")
                        results.append({stat: None for stat in stats})
            
            return pd.DataFrame(results)
            
        except Exception as e:
            self.logger.error(f"Error in zonal statistics: {e}")
            return pd.DataFrame()
    
    def extract_time_series_features(
        self,
        raster_paths: List[str],
        vector_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from time series of rasters.
        
        Args:
            raster_paths: List of raster paths (temporal sequence).
            vector_path: Optional vector path for zonal aggregation.
            
        Returns:
            Dictionary of time series features.
        """
        ts_features = {}
        
        # Stack all rasters
        data_cube = []
        for path in raster_paths:
            bands = self._load_bands(path)
            if "nir" in bands:
                data_cube.append(bands["nir"])
        
        if not data_cube:
            return ts_features
        
        data_cube = np.stack(data_cube, axis=0)
        
        # Calculate temporal statistics
        ts_features["temporal_mean"] = np.mean(data_cube, axis=0)
        ts_features["temporal_std"] = np.std(data_cube, axis=0)
        ts_features["temporal_min"] = np.min(data_cube, axis=0)
        ts_features["temporal_max"] = np.max(data_cube, axis=0)
        
        # Calculate temporal trend (simple linear regression slope)
        time_indices = np.arange(data_cube.shape[0])
        ts_features["temporal_trend"] = np.apply_along_axis(
            lambda x: np.polyfit(time_indices, x, 1)[0],
            0,
            data_cube.reshape(data_cube.shape[0], -1)
        ).reshape(data_cube.shape[1:])
        
        # Phenology metrics (if vegetation data)
        if "NDVI" in self.enabled_features or "spectral_indices" in self.enabled_features:
            ts_features["growing_season_start"] = self._detect_phenology_start(data_cube)
            ts_features["growing_season_end"] = self._detect_phenology_end(data_cube)
        
        return ts_features
    
    def _detect_phenology_start(self, data: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """Detect start of growing season."""
        # Normalize
        normalized = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)
        
        # Find first time step exceeding threshold
        start = np.argmax(normalized > threshold, axis=0)
        return start
    
    def _detect_phenology_end(self, data: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """Detect end of growing season."""
        # Normalize
        normalized = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)
        
        # Find last time step exceeding threshold
        reversed_data = normalized[::-1]
        end_reversed = np.argmax(reversed_data > threshold, axis=0)
        end = data.shape[0] - 1 - end_reversed
        return end


# Convenience functions
def extract_ndvi(raster_path: str) -> np.ndarray:
    """Quick NDVI extraction."""
    extractor = FeatureExtractor(enabled_features=["spectral_indices"])
    bands = extractor._load_bands(raster_path)
    indices = extractor.calculate_spectral_indices(bands)
    return indices.get("NDVI", np.array([]))


def extract_all_features(
    raster_path: str,
    vector_path: Optional[str] = None
) -> FeatureSet:
    """Extract all features with default settings."""
    extractor = FeatureExtractor()
    return extractor.extract_all(raster_path, vector_path)


# Import pandas at module level for type hints
try:
    import pandas as pd
except ImportError:
    pd = None
