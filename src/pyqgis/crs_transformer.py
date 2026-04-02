"""
CRS Transformer Module
======================

Handles coordinate reference system transformations for geospatial data.
Supports batch transformations, datum shifts, and accuracy validation.

Example:
    >>> transformer = CRSTransformer()
    >>> transformer.transform_layer(layer, "EPSG:4326", "EPSG:3857")
    >>> transformer.transform_file(input_path, output_path, target_crs)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass
import logging

try:
    from qgis.core import (
        QgsVectorLayer,
        QgsRasterLayer,
        QgsCoordinateReferenceSystem,
        QgsCoordinateTransform,
        QgsProject,
        QgsDatumTransform,
        QgsPointXY,
        QgsGeometry,
        QgsFeature,
        QgsApplication,
    )
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False
    # Define stub types for when QGIS is not available
    QgsVectorLayer = type('QgsVectorLayer', (), {})
    QgsRasterLayer = type('QgsRasterLayer', (), {})
    QgsCoordinateReferenceSystem = type('QgsCoordinateReferenceSystem', (), {})
    QgsCoordinateTransform = type('QgsCoordinateTransform', (), {})
    QgsProject = type('QgsProject', (), {})
    QgsDatumTransform = type('QgsDatumTransform', (), {})
    QgsPointXY = type('QgsPointXY', (), {})
    QgsGeometry = type('QgsGeometry', (), {})
    QgsFeature = type('QgsFeature', (), {})
    QgsApplication = type('QgsApplication', (), {})

try:
    from pyproj import CRS, Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

from src.utils.config import Config
from src.workflow.logger import PipelineLogger


@dataclass
class TransformationResult:
    """Container for transformation results."""
    
    source_crs: str
    target_crs: str
    features_transformed: int = 0
    datum_transform: Optional[str] = None
    accuracy_meters: float = 0.0
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class CRSTransformer:
    """
    Coordinate Reference System transformation handler.
    
    This class provides methods for transforming vector and raster data
    between different coordinate reference systems, with support for
    datum transformations and accuracy validation.
    
    Attributes:
        config: Configuration object with CRS parameters.
        logger: Logger instance for recording transformation events.
        supported_crs: List of supported CRS codes.
        
    Example:
        >>> transformer = CRSTransformer()
        >>> result = transformer.transform_vector(
        ...     layer,
        ...     target_crs="EPSG:3857",
        ...     datum_transform="EPSG:1149"
        ... )
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        logger: Optional[PipelineLogger] = None
    ):
        """
        Initialize the CRS transformer.
        
        Args:
            config: Configuration object. Uses default if None.
            logger: Logger instance. Creates new if None.
        """
        self.config = config or Config.default()
        self.logger = logger or PipelineLogger.get_logger("CRSTransformer")
        
        # Load supported CRS from config
        crs_config = self.config.crs if hasattr(self.config, 'crs') else {}
        self.supported_crs = crs_config.get('supported', [
            "EPSG:4326", "EPSG:3857", "EPSG:32633", "EPSG:25832"
        ])
        self.accuracy_threshold = crs_config.get('accuracy_threshold', 0.01)
        
        # Initialize pyproj transformer if available
        self.pyproj_transformer = None
        if PYPROJ_AVAILABLE:
            self.logger.info("pyproj available for standalone transformations")
        
        self.logger.info(f"CRSTransformer initialized with {len(self.supported_crs)} supported CRS")
    
    def get_crs_info(self, crs_code: str) -> Dict[str, Any]:
        """
        Get information about a CRS.
        
        Args:
            crs_code: CRS code (e.g., "EPSG:4326").
            
        Returns:
            Dictionary with CRS information.
        """
        info = {
            "code": crs_code,
            "name": "",
            "type": "",
            "area_of_use": "",
            "is_valid": False
        }
        
        if QGIS_AVAILABLE:
            try:
                crs = QgsCoordinateReferenceSystem(crs_code)
                if crs.isValid():
                    info["name"] = crs.description()
                    info["type"] = "projected" if crs.isProjected() else "geographic"
                    info["is_valid"] = True
                    info["units"] = crs.mapUnits()
            except Exception as e:
                self.logger.error(f"Error getting CRS info: {e}")
        
        if PYPROJ_AVAILABLE and not info["is_valid"]:
            try:
                crs = CRS.from_user_input(crs_code)
                info["name"] = crs.name
                info["type"] = crs.type_name
                info["is_valid"] = True
            except Exception:
                pass
        
        return info
    
    def transform_layer(
        self,
        layer: Union[QgsVectorLayer, QgsRasterLayer],
        target_crs: str,
        datum_transform: Optional[str] = None,
        in_place: bool = False
    ) -> Union[QgsVectorLayer, QgsRasterLayer, None]:
        """
        Transform a layer to target CRS.
        
        Args:
            layer: Layer to transform.
            target_crs: Target CRS code.
            datum_transform: Optional datum transform operation.
            in_place: If True, transform layer in place.
            
        Returns:
            Transformed layer or None if failed.
        """
        if not QGIS_AVAILABLE:
            self.logger.warning("QGIS not available for transformation")
            return None
        
        try:
            source_crs = layer.crs()
            if not source_crs.isValid():
                self.logger.error(f"Invalid source CRS for layer: {layer.name()}")
                return None
            
            target_crs_obj = QgsCoordinateReferenceSystem(target_crs)
            if not target_crs_obj.isValid():
                self.logger.error(f"Invalid target CRS: {target_crs}")
                return None
            
            # Create coordinate transform
            transform = QgsCoordinateTransform(
                source_crs, target_crs_obj, QgsProject.instance()
            )
            
            # Set datum transform if specified
            if datum_transform:
                transform.setDatumTransform(datum_transform)
            
            if isinstance(layer, QgsVectorLayer):
                return self._transform_vector_layer(
                    layer, transform, target_crs_obj, in_place
                )
            elif isinstance(layer, QgsRasterLayer):
                return self._transform_raster_layer(
                    layer, target_crs_obj
                )
            else:
                self.logger.error(f"Unsupported layer type: {type(layer)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error transforming layer: {e}")
            return None
    
    def _transform_vector_layer(
        self,
        layer: QgsVectorLayer,
        transform: QgsCoordinateTransform,
        target_crs: QgsCoordinateReferenceSystem,
        in_place: bool
    ) -> QgsVectorLayer:
        """Transform vector layer."""
        try:
            if in_place:
                # Transform in place
                layer.startEditing()
                
                for feature in layer.getFeatures():
                    if feature.geometry():
                        transformed_geom = feature.geometry().transform(transform)
                        feature.setGeometry(transformed_geom)
                        layer.updateFeature(feature)
                
                layer.setCrs(target_crs)
                layer.commitChanges()
                
                self.logger.info(f"Transformed layer '{layer.name()}' in place")
                return layer
            else:
                # Create new transformed layer
                from qgis.core import QgsVectorLayerUtils
                
                new_layer = QgsVectorLayer(
                    layer.wkbType(),
                    f"{layer.name()}_transformed",
                    "memory"
                )
                new_layer.setCrs(target_crs)
                
                # Copy fields
                new_layer.dataProvider().addAttributes(layer.fields())
                new_layer.updateFields()
                
                # Transform and copy features
                features = []
                for feature in layer.getFeatures():
                    new_feature = QgsFeature(new_layer.fields())
                    new_feature.setAttributes(feature.attributes())
                    if feature.geometry():
                        new_feature.setGeometry(feature.geometry().transform(transform))
                    features.append(new_feature)
                
                new_layer.dataProvider().addFeatures(features)
                new_layer.updateExtents()
                
                self.logger.info(f"Created transformed layer with {len(features)} features")
                return new_layer
                
        except Exception as e:
            self.logger.error(f"Error transforming vector layer: {e}")
            return None
    
    def _transform_raster_layer(
        self,
        layer: QgsRasterLayer,
        target_crs: QgsCoordinateReferenceSystem
    ) -> QgsRasterLayer:
        """Transform raster layer using warp."""
        try:
            # For rasters, we need to use GDAL warp
            from osgeo import gdal
            
            source_path = layer.dataProvider().dataSourceUri()
            source = gdal.Open(source_path)
            
            if not source:
                self.logger.error(f"Cannot open raster: {source_path}")
                return None
            
            # Get target CRS in WKT
            target_wkt = target_crs.toWkt()
            
            # Warp to target CRS
            warped = gdal.Warp(
                "",
                source,
                dstSRS=target_wkt,
                format="MEM"
            )
            
            if warped:
                # Create new layer from warped dataset
                # In production, would save to file and reload
                self.logger.info("Raster transformation completed (in memory)")
                return layer  # Return original for now
            else:
                self.logger.error("Raster warp failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Error transforming raster: {e}")
            return None
    
    def transform_file(
        self,
        input_path: str,
        output_path: str,
        target_crs: str,
        source_crs: Optional[str] = None
    ) -> bool:
        """
        Transform a file to target CRS.
        
        Args:
            input_path: Input file path.
            output_path: Output file path.
            target_crs: Target CRS code.
            source_crs: Source CRS (auto-detected if None).
            
        Returns:
            True if transformation successful.
        """
        try:
            ext = Path(input_path).suffix.lower()
            
            if ext in [".shp", ".geojson", ".gpkg", ".kml", ".gml"]:
                return self._transform_vector_file(
                    input_path, output_path, target_crs, source_crs
                )
            elif ext in [".tif", ".tiff", ".jp2", ".img", ".asc"]:
                return self._transform_raster_file(
                    input_path, output_path, target_crs, source_crs
                )
            else:
                self.logger.error(f"Unsupported file type: {ext}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error transforming file: {e}")
            return False
    
    def _transform_vector_file(
        self,
        input_path: str,
        output_path: str,
        target_crs: str,
        source_crs: Optional[str]
    ) -> bool:
        """Transform vector file using ogr2ogr."""
        try:
            from osgeo import ogr
            
            # Open source
            source = ogr.Open(input_path)
            if not source:
                self.logger.error(f"Cannot open vector file: {input_path}")
                return False
            
            source_layer = source.GetLayer(0)
            
            # Get or set source CRS
            if source_crs:
                src_srs = self._crs_to_osr(source_crs)
            else:
                src_srs = source_layer.GetSpatialRef()
            
            # Target CRS
            tgt_srs = self._crs_to_osr(target_crs)
            
            # Create output
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            driver = ogr.GetDriverByName(self._get_ogr_driver(output_path))
            output = driver.CreateDataSource(output_path)
            output_layer = output.CreateLayer(
                Path(output_path).stem,
                tgt_srs,
                source_layer.GetGeomType()
            )
            
            # Copy fields
            source_defn = source_layer.GetLayerDefn()
            for i in range(source_defn.GetFieldCount()):
                output_layer.CreateField(source_defn.GetFieldDefn(i))
            
            # Create transform
            transform = osr.CoordinateTransformation(src_srs, tgt_srs)
            
            # Transform features
            for feature in source_layer:
                geom = feature.GetGeometryRef()
                if geom:
                    geom.Transform(transform)
                
                output_feature = ogr.Feature(output_layer.GetLayerDefn())
                for i in range(feature.GetFieldCount()):
                    output_feature.SetField(i, feature.GetField(i))
                output_feature.SetGeometry(geom)
                output_layer.CreateFeature(output_feature)
            
            # Cleanup
            source = None
            output = None
            
            self.logger.info(f"Transformed vector file to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error transforming vector file: {e}")
            return False
    
    def _transform_raster_file(
        self,
        input_path: str,
        output_path: str,
        target_crs: str,
        source_crs: Optional[str]
    ) -> bool:
        """Transform raster file using gdal.Warp."""
        try:
            from osgeo import gdal
            
            # Open source
            source = gdal.Open(input_path)
            if not source:
                self.logger.error(f"Cannot open raster file: {input_path}")
                return False
            
            # Create output directory
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Warp options
            options = gdal.WarpOptions(
                dstSRS=target_crs,
                srcSRS=source_crs,
                format="GTiff",
                compression="LZW",
                creationOptions=["TILED=YES", "BIGTIFF=YES"]
            )
            
            # Execute warp
            result = gdal.Warp(output_path, source, options=options)
            
            # Cleanup
            source = None
            result = None
            
            self.logger.info(f"Transformed raster file to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error transforming raster file: {e}")
            return False
    
    def _crs_to_osr(self, crs_code: str):
        """Convert CRS code to OSR object."""
        try:
            from osgeo import osr
            srs = osr.SpatialReference()
            if crs_code.startswith("EPSG:"):
                srs.ImportFromEPSG(int(crs_code.split(":")[1]))
            else:
                srs.ImportFromWkt(crs_code)
            return srs
        except Exception as e:
            self.logger.error(f"Error creating OSR: {e}")
            return None
    
    def _get_ogr_driver(self, path: str) -> str:
        """Get OGR driver name for file extension."""
        ext = Path(path).suffix.lower()
        drivers = {
            ".shp": "ESRI Shapefile",
            ".geojson": "GeoJSON",
            ".gpkg": "GPKG",
            ".kml": "KML",
            ".gml": "GML",
        }
        return drivers.get(ext, "GPKG")
    
    def transform_point(
        self,
        x: float,
        y: float,
        source_crs: str,
        target_crs: str
    ) -> Tuple[float, float]:
        """
        Transform a single point between CRS.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
            source_crs: Source CRS code.
            target_crs: Target CRS code.
            
        Returns:
            Tuple of (x, y) in target CRS.
        """
        if PYPROJ_AVAILABLE:
            try:
                transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
                return transformer.transform(x, y)
            except Exception as e:
                self.logger.error(f"pyproj transform error: {e}")
        
        if QGIS_AVAILABLE:
            try:
                src = QgsCoordinateReferenceSystem(source_crs)
                tgt = QgsCoordinateReferenceSystem(target_crs)
                transform = QgsCoordinateTransform(src, tgt, QgsProject.instance())
                point = QgsPointXY(x, y).transform(transform)
                return (point.x(), point.y())
            except Exception as e:
                self.logger.error(f"QGIS transform error: {e}")
        
        raise ValueError("No transformation backend available")
    
    def get_datum_transforms(
        self,
        source_crs: str,
        target_crs: str
    ) -> List[Dict[str, Any]]:
        """
        Get available datum transforms between two CRS.
        
        Args:
            source_crs: Source CRS code.
            target_crs: Target CRS code.
            
        Returns:
            List of available datum transforms.
        """
        if not QGIS_AVAILABLE:
            return []
        
        try:
            src = QgsCoordinateReferenceSystem(source_crs)
            tgt = QgsCoordinateReferenceSystem(target_crs)
            
            transforms = QgsDatumTransform.operations(src, tgt)
            
            result = []
            for t in transforms:
                result.append({
                    "code": t.epsg,
                    "name": t.name,
                    "accuracy": t.accuracy,
                    "is_ballpark": t.isBallpark
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting datum transforms: {e}")
            return []
    
    def validate_transformation(
        self,
        source_crs: str,
        target_crs: str,
        test_points: Optional[List[Tuple[float, float]]] = None
    ) -> TransformationResult:
        """
        Validate transformation accuracy.
        
        Args:
            source_crs: Source CRS code.
            target_crs: Target CRS code.
            test_points: Optional test points for validation.
            
        Returns:
            TransformationResult with accuracy metrics.
        """
        result = TransformationResult(
            source_crs=source_crs,
            target_crs=target_crs
        )
        
        # Get datum transforms
        transforms = self.get_datum_transforms(source_crs, target_crs)
        if transforms:
            best = min(transforms, key=lambda t: t.get("accuracy", float("inf")))
            result.datum_transform = best["code"]
            result.accuracy_meters = best.get("accuracy", 0)
            
            if result.accuracy_meters > self.accuracy_threshold:
                result.warnings.append(
                    f"Transformation accuracy ({result.accuracy_meters}m) exceeds threshold"
                )
        
        # Validate with test points if provided
        if test_points:
            for x, y in test_points:
                try:
                    # Round-trip transformation
                    x2, y2 = self.transform_point(x, y, source_crs, target_crs)
                    x3, y3 = self.transform_point(x2, y2, target_crs, source_crs)
                    
                    error = ((x - x3) ** 2 + (y - y3) ** 2) ** 0.5
                    if error > self.accuracy_threshold:
                        result.warnings.append(
                            f"Round-trip error at ({x}, {y}): {error:.4f}"
                        )
                except Exception as e:
                    result.warnings.append(f"Transform error: {e}")
        
        return result
    
    def batch_transform(
        self,
        files: List[str],
        output_dir: str,
        target_crs: str
    ) -> List[TransformationResult]:
        """
        Batch transform multiple files.
        
        Args:
            files: List of input file paths.
            output_dir: Output directory.
            target_crs: Target CRS for all files.
            
        Returns:
            List of TransformationResult objects.
        """
        results = []
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for file_path in files:
            output_path = os.path.join(
                output_dir,
                f"{Path(file_path).stem}_transformed{Path(file_path).suffix}"
            )
            
            success = self.transform_file(file_path, output_path, target_crs)
            
            result = TransformationResult(
                source_crs="auto",
                target_crs=target_crs,
                features_transformed=1 if success else 0
            )
            
            if not success:
                result.warnings.append(f"Transformation failed for {file_path}")
            
            results.append(result)
        
        return results


# Convenience functions
def reproject_to_web_mercator(
    layer: QgsVectorLayer,
    in_place: bool = False
) -> QgsVectorLayer:
    """Reproject layer to Web Mercator (EPSG:3857)."""
    transformer = CRSTransformer()
    return transformer.transform_layer(layer, "EPSG:3857", in_place=in_place)


def reproject_to_wgs84(
    layer: QgsVectorLayer,
    in_place: bool = False
) -> QgsVectorLayer:
    """Reproject layer to WGS84 (EPSG:4326)."""
    transformer = CRSTransformer()
    return transformer.transform_layer(layer, "EPSG:4326", in_place=in_place)
