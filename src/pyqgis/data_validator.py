"""
Data Validator Module
=====================

Provides data validation and quality checks for geospatial data.
Supports geometry validation, attribute checks, topology validation,
and comprehensive quality reporting.

Example:
    >>> validator = DataValidator()
    >>> results = validator.validate_layer(layer)
    >>> validator.run_topology_checks(layer, ["overlap", "gap"])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

try:
    from qgis.core import (
        QgsVectorLayer,
        QgsFeature,
        QgsGeometry,
        QgsGeometryValidator,
        QgsGeometryValidatorResult,
        QgsFeatureRequest,
        QgsSpatialIndex,
        QgsDistanceArea,
        QgsProject,
        QgsCoordinateReferenceSystem,
        QgsExpression,
    )
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False
    # Define stub types for when QGIS is not available
    QgsVectorLayer = type('QgsVectorLayer', (), {})
    QgsFeature = type('QgsFeature', (), {})
    QgsGeometry = type('QgsGeometry', (), {})
    QgsGeometryValidator = type('QgsGeometryValidator', (), {})
    QgsGeometryValidatorResult = type('QgsGeometryValidatorResult', (), {})
    QgsFeatureRequest = type('QgsFeatureRequest', (), {})
    QgsSpatialIndex = type('QgsSpatialIndex', (), {})
    QgsDistanceArea = type('QgsDistanceArea', (), {})
    QgsProject = type('QgsProject', (), {})
    QgsCoordinateReferenceSystem = type('QgsCoordinateReferenceSystem', (), {})
    QgsExpression = type('QgsExpression', (), {})

from shapely.geometry import shape, mapping
from shapely.validation import make_valid, explain_validity
import geopandas as gpd

from src.utils.config import Config
from src.workflow.logger import PipelineLogger


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationType(Enum):
    """Types of validation checks."""
    GEOMETRY = "geometry"
    ATTRIBUTE = "attribute"
    TOPOLOGY = "topology"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"


@dataclass
class ValidationIssue:
    """Container for a single validation issue."""
    
    issue_type: ValidationType
    severity: ValidationSeverity
    message: str
    feature_id: Optional[int] = None
    geometry_error: Optional[str] = None
    field_name: Optional[str] = None
    field_value: Optional[Any] = None
    location: Optional[Tuple[float, float]] = None
    fixable: bool = False
    fix_applied: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.issue_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "feature_id": self.feature_id,
            "geometry_error": self.geometry_error,
            "field_name": self.field_name,
            "field_value": self.field_value,
            "location": self.location,
            "fixable": self.fixable,
            "fix_applied": self.fix_applied
        }


@dataclass
class ValidationResult:
    """Container for complete validation results."""
    
    layer_name: str
    total_features: int = 0
    valid_features: int = 0
    invalid_features: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    validation_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def validity_rate(self) -> float:
        """Calculate validity rate percentage."""
        if self.total_features == 0:
            return 100.0
        return (self.valid_features / self.total_features) * 100
    
    @property
    def issue_summary(self) -> Dict[str, int]:
        """Get count of issues by severity."""
        summary = {s.value: 0 for s in ValidationSeverity}
        for issue in self.issues:
            summary[issue.severity.value] += 1
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer_name": self.layer_name,
            "total_features": self.total_features,
            "valid_features": self.valid_features,
            "invalid_features": self.invalid_features,
            "validity_rate": self.validity_rate,
            "issue_summary": self.issue_summary,
            "issues": [i.to_dict() for i in self.issues[:100]],  # Limit for serialization
            "validation_time": self.validation_time,
            "timestamp": self.timestamp
        }


class DataValidator:
    """
    Geospatial data validation and quality assurance.
    
    This class provides comprehensive validation for vector and raster
    data, including geometry validation, attribute checks, topology
    validation, and spatial accuracy assessment.
    
    Attributes:
        config: Configuration object with validation parameters.
        logger: Logger instance for recording validation events.
        tolerance: Tolerance values for various checks.
        
    Example:
        >>> validator = DataValidator()
        >>> result = validator.validate_layer(layer)
        >>> if result.validity_rate < 95:
        ...     validator.auto_repair(layer)
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        logger: Optional[PipelineLogger] = None
    ):
        """
        Initialize the data validator.
        
        Args:
            config: Configuration object. Uses default if None.
            logger: Logger instance. Creates new if None.
        """
        self.config = config or Config.default()
        self.logger = logger or PipelineLogger.get_logger("DataValidator")
        
        # Load validation settings from config
        qa_config = self.config.pipeline.quality_assurance if hasattr(self.config, 'pipeline') else {}
        self.tolerance = qa_config.get('tolerance', {})
        self.auto_repair_enabled = qa_config.get('auto_repair', True)
        
        # Default tolerances
        self.positional_tolerance = self.tolerance.get('positional', 1.0)  # meters
        self.area_tolerance = self.tolerance.get('area', 0.05)  # 5%
        
        self.logger.info("DataValidator initialized")
    
    def validate_layer(
        self,
        layer: Union[QgsVectorLayer, gpd.GeoDataFrame],
        checks: Optional[List[str]] = None,
        sample_size: Optional[int] = None
    ) -> ValidationResult:
        """
        Run comprehensive validation on a layer.
        
        Args:
            layer: Layer to validate (QGIS or GeoDataFrame).
            checks: List of checks to run. None runs all.
            sample_size: Optional sample size for large datasets.
            
        Returns:
            ValidationResult with all issues found.
            
        Example:
            >>> result = validator.validate_layer(
            ...     layer,
            ...     checks=["geometry", "attributes", "topology"]
            ... )
            >>> print(f"Validity rate: {result.validity_rate:.1f}%")
        """
        import time
        start_time = time.time()
        
        # Convert to GeoDataFrame if QGIS layer
        if QGIS_AVAILABLE and isinstance(layer, QgsVectorLayer):
            gdf = self._qgis_to_geopandas(layer)
            layer_name = layer.name()
            total_features = layer.featureCount()
        elif isinstance(layer, gpd.GeoDataFrame):
            gdf = layer
            layer_name = "GeoDataFrame"
            total_features = len(gdf)
        else:
            raise ValueError("Unsupported layer type")
        
        # Apply sampling if specified
        if sample_size and sample_size < total_features:
            gdf = gdf.sample(n=sample_size, random_state=42)
            self.logger.info(f"Sampled {sample_size} features for validation")
        
        result = ValidationResult(
            layer_name=layer_name,
            total_features=len(gdf)
        )
        
        # Default checks
        if checks is None:
            checks = ["geometry", "attributes", "topology"]
        
        # Run checks
        if "geometry" in checks:
            self._check_geometry(gdf, result)
        
        if "attributes" in checks:
            self._check_attributes(gdf, result)
        
        if "topology" in checks:
            self._check_topology(gdf, result)
        
        if "spatial" in checks:
            self._check_spatial(gdf, result)
        
        # Calculate valid features
        invalid_ids = set()
        for issue in result.issues:
            if issue.feature_id is not None:
                invalid_ids.add(issue.feature_id)
        
        result.invalid_features = len(invalid_ids)
        result.valid_features = result.total_features - result.invalid_features
        result.validation_time = time.time() - start_time
        
        self.logger.info(
            f"Validation complete: {result.validity_rate:.1f}% valid "
            f"({result.valid_features}/{result.total_features})"
        )
        
        return result
    
    def _qgis_to_geopandas(self, layer: QgsVectorLayer) -> gpd.GeoDataFrame:
        """Convert QGIS layer to GeoDataFrame."""
        features = []
        for feature in layer.getFeatures():
            geom = feature.geometry()
            if geom:
                features.append({
                    "geometry": shape(geom.asJson()),
                    **{field.name(): feature[field.name()] for field in layer.fields()}
                })
        return gpd.GeoDataFrame(features, crs=layer.crs().authid() if layer.crs() else None)
    
    def _check_geometry(
        self,
        gdf: gpd.GeoDataFrame,
        result: ValidationResult
    ) -> None:
        """Check geometry validity."""
        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            if geom is None or geom.is_empty:
                result.issues.append(ValidationIssue(
                    issue_type=ValidationType.GEOMETRY,
                    severity=ValidationSeverity.ERROR,
                    message="Empty or null geometry",
                    feature_id=idx,
                    fixable=False
                ))
                continue
            
            # Check validity using shapely
            if not geom.is_valid:
                validity_msg = explain_validity(geom)
                result.issues.append(ValidationIssue(
                    issue_type=ValidationType.GEOMETRY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid geometry: {validity_msg}",
                    feature_id=idx,
                    geometry_error=validity_msg,
                    fixable=True
                ))
            
            # Check geometry type consistency
            expected_type = gdf.geometry.geom_type.mode()
            if geom.geom_type != expected_type:
                result.issues.append(ValidationIssue(
                    issue_type=ValidationType.GEOMETRY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Unexpected geometry type: {geom.geom_type}",
                    feature_id=idx,
                    fixable=False
                ))
            
            # Check for self-intersection
            if geom.geom_type in ["Polygon", "MultiPolygon"]:
                if geom.intersects(geom.boundary):
                    result.issues.append(ValidationIssue(
                        issue_type=ValidationType.GEOMETRY,
                        severity=ValidationSeverity.ERROR,
                        message="Self-intersecting polygon",
                        feature_id=idx,
                        fixable=True
                    ))
    
    def _check_attributes(
        self,
        gdf: gpd.GeoDataFrame,
        result: ValidationResult
    ) -> None:
        """Check attribute completeness and validity."""
        # Check for null values in required fields
        for col in gdf.columns:
            if col == "geometry":
                continue
            
            null_count = gdf[col].isnull().sum()
            if null_count > 0:
                null_pct = (null_count / len(gdf)) * 100
                
                severity = ValidationSeverity.WARNING
                if null_pct > 50:
                    severity = ValidationSeverity.ERROR
                elif null_pct > 20:
                    severity = ValidationSeverity.WARNING
                
                result.issues.append(ValidationIssue(
                    issue_type=ValidationType.ATTRIBUTE,
                    severity=severity,
                    message=f"Column '{col}' has {null_count} null values ({null_pct:.1f}%)",
                    field_name=col,
                    field_value=null_count,
                    fixable=False
                ))
            
            # Check for empty strings
            if gdf[col].dtype == object:
                empty_count = (gdf[col] == "").sum()
                if empty_count > 0:
                    result.issues.append(ValidationIssue(
                        issue_type=ValidationType.ATTRIBUTE,
                        severity=ValidationSeverity.WARNING,
                        message=f"Column '{col}' has {empty_count} empty strings",
                        field_name=col,
                        field_value=empty_count,
                        fixable=True
                    ))
            
            # Check for duplicate values in potentially unique fields
            if "id" in col.lower() or "fid" in col.lower():
                dup_count = gdf[col].duplicated().sum()
                if dup_count > 0:
                    result.issues.append(ValidationIssue(
                        issue_type=ValidationType.ATTRIBUTE,
                        severity=ValidationSeverity.ERROR,
                        message=f"Column '{col}' has {dup_count} duplicate values",
                        field_name=col,
                        fixable=False
                    ))
    
    def _check_topology(
        self,
        gdf: gpd.GeoDataFrame,
        result: ValidationResult
    ) -> None:
        """Check topological relationships."""
        if len(gdf) < 2:
            return
        
        # Check for overlaps (polygons)
        if gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            self._check_overlaps(gdf, result)
        
        # Check for gaps (polygons)
        if gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            self._check_gaps(gdf, result)
        
        # Check for duplicate geometries
        self._check_duplicates(gdf, result)
    
    def _check_overlaps(
        self,
        gdf: gpd.GeoDataFrame,
        result: ValidationResult
    ) -> None:
        """Check for overlapping polygons."""
        # Build spatial index for efficiency
        from shapely.strtree import STRtree
        
        geometries = gdf.geometry.tolist()
        tree = STRtree(geometries)
        
        checked = set()
        for i, geom in enumerate(geometries):
            if geom is None:
                continue
            
            # Query potential overlaps
            potential = tree.query(geom)
            
            for j in potential:
                if i >= j or (i, j) in checked:
                    continue
                checked.add((i, j))
                
                other_geom = geometries[j]
                if other_geom is None:
                    continue
                
                if geom.intersects(other_geom):
                    intersection = geom.intersection(other_geom)
                    if intersection.area > 0.001:  # Tolerance
                        result.issues.append(ValidationIssue(
                            issue_type=ValidationType.TOPOLOGY,
                            severity=ValidationSeverity.WARNING,
                            message=f"Overlap between features {i} and {j}",
                            feature_id=i,
                            location=(geom.centroid.x, geom.centroid.y),
                            fixable=True
                        ))
    
    def _check_gaps(
        self,
        gdf: gpd.GeoDataFrame,
        result: ValidationResult
    ) -> None:
        """Check for gaps between polygons."""
        # Union all geometries
        union = gdf.geometry.unary_union
        
        if union.geom_type == "MultiPolygon":
            # Check if there are gaps within the bounding box
            bounds = union.bounds
            from shapely.geometry import box
            bbox = box(*bounds)
            
            # Difference should be empty if no gaps
            diff = bbox.difference(union)
            if diff.area > 0:
                # Check if gaps are significant
                gap_ratio = diff.area / bbox.area
                if gap_ratio > 0.01:  # More than 1% gaps
                    result.issues.append(ValidationIssue(
                        issue_type=ValidationType.TOPOLOGY,
                        severity=ValidationSeverity.INFO,
                        message=f"Gaps detected: {gap_ratio*100:.1f}% of area",
                        fixable=False
                    ))
    
    def _check_duplicates(
        self,
        gdf: gpd.GeoDataFrame,
        result: ValidationResult
    ) -> None:
        """Check for duplicate geometries."""
        # Create geometry hashes
        geom_hashes = []
        for geom in gdf.geometry:
            if geom is not None:
                geom_hashes.append(hash(geom.wkb))
            else:
                geom_hashes.append(None)
        
        # Find duplicates
        seen = {}
        for i, h in enumerate(geom_hashes):
            if h is None:
                continue
            if h in seen:
                result.issues.append(ValidationIssue(
                    issue_type=ValidationType.TOPOLOGY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Duplicate geometry: features {seen[h]} and {i}",
                    feature_id=i,
                    fixable=True
                ))
            else:
                seen[h] = i
    
    def _check_spatial(
        self,
        gdf: gpd.GeoDataFrame,
        result: ValidationResult
    ) -> None:
        """Check spatial accuracy and extent."""
        # Check coordinate ranges
        bounds = gdf.total_bounds
        if len(bounds) == 4:
            minx, miny, maxx, maxy = bounds
            
            # Check for coordinates outside valid range
            if minx < -180 or maxx > 180:
                # Might be projected CRS
                if minx < -1000000 or maxx > 1000000:
                    result.issues.append(ValidationIssue(
                        issue_type=ValidationType.SPATIAL,
                        severity=ValidationSeverity.INFO,
                        message="Coordinates appear to be in projected CRS",
                        fixable=False
                    ))
            
            # Check for zero-area features
            if gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
                areas = gdf.geometry.area
                zero_area = (areas == 0).sum()
                if zero_area > 0:
                    result.issues.append(ValidationIssue(
                        issue_type=ValidationType.SPATIAL,
                        severity=ValidationSeverity.ERROR,
                        message=f"{zero_area} features have zero area",
                        fixable=False
                    ))
    
    def run_topology_checks(
        self,
        layer: Union[QgsVectorLayer, gpd.GeoDataFrame],
        check_types: List[str]
    ) -> ValidationResult:
        """
        Run specific topology checks.
        
        Args:
            layer: Layer to check.
            check_types: List of topology checks (overlap, gap, duplicate, self_intersection).
            
        Returns:
            ValidationResult with topology issues.
        """
        # Convert to GeoDataFrame if needed
        if QGIS_AVAILABLE and isinstance(layer, QgsVectorLayer):
            gdf = self._qgis_to_geopandas(layer)
        else:
            gdf = layer
        
        result = ValidationResult(
            layer_name=layer.name() if hasattr(layer, 'name') else "Layer",
            total_features=len(gdf)
        )
        
        if "overlap" in check_types:
            self._check_overlaps(gdf, result)
        
        if "gap" in check_types:
            self._check_gaps(gdf, result)
        
        if "duplicate" in check_types:
            self._check_duplicates(gdf, result)
        
        if "self_intersection" in check_types:
            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom and not geom.is_valid:
                    result.issues.append(ValidationIssue(
                        issue_type=ValidationType.TOPOLOGY,
                        severity=ValidationSeverity.ERROR,
                        message="Self-intersection detected",
                        feature_id=idx,
                        fixable=True
                    ))
        
        return result
    
    def auto_repair(
        self,
        layer: Union[QgsVectorLayer, gpd.GeoDataFrame],
        output_path: Optional[str] = None
    ) -> Union[QgsVectorLayer, gpd.GeoDataFrame]:
        """
        Automatically repair fixable geometry issues.
        
        Args:
            layer: Layer to repair.
            output_path: Optional path to save repaired layer.
            
        Returns:
            Repaired layer.
        """
        # Convert to GeoDataFrame if needed
        if QGIS_AVAILABLE and isinstance(layer, QgsVectorLayer):
            gdf = self._qgis_to_geopandas(layer)
            is_qgis = True
        else:
            gdf = layer.copy()
            is_qgis = False
        
        repaired_count = 0
        
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or not geom.is_valid:
                try:
                    # Use shapely's make_valid
                    gdf.at[idx, 'geometry'] = make_valid(geom)
                    repaired_count += 1
                except Exception as e:
                    self.logger.warning(f"Could not repair feature {idx}: {e}")
        
        self.logger.info(f"Repaired {repaired_count} geometries")
        
        # Save if output path specified
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(output_path, driver="GPKG")
            self.logger.info(f"Saved repaired layer to {output_path}")
        
        if is_qgis and QGIS_AVAILABLE:
            # Convert back to QGIS layer
            return self._geopandas_to_qgis(gdf, layer)
        
        return gdf
    
    def _geopandas_to_qgis(
        self,
        gdf: gpd.GeoDataFrame,
        template_layer: QgsVectorLayer
    ) -> QgsVectorLayer:
        """Convert GeoDataFrame back to QGIS layer."""
        new_layer = QgsVectorLayer(
            f"Polygon?crs={template_layer.crs().authid()}",
            f"{template_layer.name()}_repaired",
            "memory"
        )
        
        provider = new_layer.dataProvider()
        provider.addAttributes(template_layer.fields())
        new_layer.updateFields()
        
        features = []
        for idx, row in gdf.iterrows():
            feature = QgsFeature(new_layer.fields())
            feature.setAttributes([row[col] for col in gdf.columns if col != 'geometry'])
            
            # Convert shapely geometry to QGIS
            from qgis.core import QgsGeometry
            feature.setGeometry(QgsGeometry.fromWkt(row.geometry.wkt))
            features.append(feature)
        
        provider.addFeatures(features)
        new_layer.updateExtents()
        
        return new_layer
    
    def generate_quality_report(
        self,
        result: ValidationResult,
        output_path: str,
        format: str = "json"
    ) -> bool:
        """
        Generate quality report from validation results.
        
        Args:
            result: ValidationResult to report on.
            output_path: Output file path.
            format: Report format (json, html, txt).
            
        Returns:
            True if report generated successfully.
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                import json
                with open(output_path, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)
            
            elif format == "html":
                self._generate_html_report(result, output_path)
            
            elif format == "txt":
                self._generate_text_report(result, output_path)
            
            else:
                self.logger.error(f"Unknown report format: {format}")
                return False
            
            self.logger.info(f"Quality report saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return False
    
    def _generate_html_report(
        self,
        result: ValidationResult,
        output_path: str
    ) -> None:
        """Generate HTML quality report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quality Report - {result.layer_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .validity {{ font-size: 24px; font-weight: bold; }}
                .valid {{ color: #28a745; }}
                .invalid {{ color: #dc3545; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background: #f2f2f2; }}
                .error {{ color: #dc3545; }}
                .warning {{ color: #ffc107; }}
                .info {{ color: #17a2b8; }}
            </style>
        </head>
        <body>
            <h1>Geospatial Data Quality Report</h1>
            <p><strong>Layer:</strong> {result.layer_name}</p>
            <p><strong>Timestamp:</strong> {result.timestamp}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p class="validity {'valid' if result.validity_rate >= 95 else 'invalid'}">
                    Validity Rate: {result.validity_rate:.1f}%
                </p>
                <p>Total Features: {result.total_features}</p>
                <p>Valid Features: {result.valid_features}</p>
                <p>Invalid Features: {result.invalid_features}</p>
                <p>Validation Time: {result.validation_time:.2f}s</p>
                
                <h3>Issues by Severity</h3>
                <ul>
                    <li class="error">Critical: {result.issue_summary.get('critical', 0)}</li>
                    <li class="error">Errors: {result.issue_summary.get('error', 0)}</li>
                    <li class="warning">Warnings: {result.issue_summary.get('warning', 0)}</li>
                    <li class="info">Info: {result.issue_summary.get('info', 0)}</li>
                </ul>
            </div>
            
            <h2>Issues</h2>
            <table>
                <tr>
                    <th>Type</th>
                    <th>Severity</th>
                    <th>Message</th>
                    <th>Feature ID</th>
                    <th>Fixable</th>
                </tr>
        """
        
        for issue in result.issues[:100]:  # Limit to 100 issues
            html += f"""
                <tr>
                    <td>{issue.issue_type.value}</td>
                    <td class="{issue.severity.value}">{issue.severity.value}</td>
                    <td>{issue.message}</td>
                    <td>{issue.feature_id or 'N/A'}</td>
                    <td>{'Yes' if issue.fixable else 'No'}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
    
    def _generate_text_report(
        self,
        result: ValidationResult,
        output_path: str
    ) -> None:
        """Generate text quality report."""
        lines = [
            "=" * 60,
            "GEOSPATIAL DATA QUALITY REPORT",
            "=" * 60,
            f"Layer: {result.layer_name}",
            f"Timestamp: {result.timestamp}",
            f"Validation Time: {result.validation_time:.2f}s",
            "",
            "SUMMARY",
            "-" * 40,
            f"Total Features: {result.total_features}",
            f"Valid Features: {result.valid_features}",
            f"Invalid Features: {result.invalid_features}",
            f"Validity Rate: {result.validity_rate:.1f}%",
            "",
            "ISSUES BY SEVERITY",
            "-" * 40,
        ]
        
        for severity, count in result.issue_summary.items():
            lines.append(f"  {severity.upper()}: {count}")
        
        lines.extend([
            "",
            "DETAILED ISSUES",
            "-" * 40,
        ])
        
        for i, issue in enumerate(result.issues[:50], 1):
            lines.append(
                f"{i}. [{issue.severity.value.upper()}] {issue.message}"
                f" (Feature: {issue.feature_id or 'N/A'})"
            )
        
        if len(result.issues) > 50:
            lines.append(f"... and {len(result.issues) - 50} more issues")
        
        lines.append("=" * 60)
        
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
    
    def validate_against_schema(
        self,
        layer: Union[QgsVectorLayer, gpd.GeoDataFrame],
        schema: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate layer against a schema definition.
        
        Args:
            layer: Layer to validate.
            schema: Schema definition with field types and constraints.
            
        Returns:
            ValidationResult with schema validation issues.
        """
        # Convert to GeoDataFrame if needed
        if QGIS_AVAILABLE and isinstance(layer, QgsVectorLayer):
            gdf = self._qgis_to_geopandas(layer)
            layer_name = layer.name()
        else:
            gdf = layer
            layer_name = "Layer"
        
        result = ValidationResult(
            layer_name=layer_name,
            total_features=len(gdf)
        )
        
        # Check required fields
        required_fields = schema.get("required_fields", [])
        for field in required_fields:
            if field not in gdf.columns:
                result.issues.append(ValidationIssue(
                    issue_type=ValidationType.ATTRIBUTE,
                    severity=ValidationSeverity.ERROR,
                    message=f"Missing required field: {field}",
                    field_name=field,
                    fixable=False
                ))
        
        # Check field types
        field_types = schema.get("field_types", {})
        for field, expected_type in field_types.items():
            if field not in gdf.columns:
                continue
            
            actual_type = str(gdf[field].dtype)
            if not self._type_matches(actual_type, expected_type):
                result.issues.append(ValidationIssue(
                    issue_type=ValidationType.ATTRIBUTE,
                    severity=ValidationSeverity.WARNING,
                    message=f"Field '{field}' type mismatch: expected {expected_type}, got {actual_type}",
                    field_name=field,
                    field_value=actual_type,
                    fixable=False
                ))
        
        # Check value constraints
        constraints = schema.get("constraints", {})
        for field, constraint in constraints.items():
            if field not in gdf.columns:
                continue
            
            # Check min/max values
            if "min" in constraint:
                invalid = gdf[gdf[field] < constraint["min"]]
                if len(invalid) > 0:
                    result.issues.append(ValidationIssue(
                        issue_type=ValidationType.ATTRIBUTE,
                        severity=ValidationSeverity.ERROR,
                        message=f"Field '{field}' has {len(invalid)} values below minimum ({constraint['min']})",
                        field_name=field,
                        fixable=False
                    ))
            
            if "max" in constraint:
                invalid = gdf[gdf[field] > constraint["max"]]
                if len(invalid) > 0:
                    result.issues.append(ValidationIssue(
                        issue_type=ValidationType.ATTRIBUTE,
                        severity=ValidationSeverity.ERROR,
                        message=f"Field '{field}' has {len(invalid)} values above maximum ({constraint['max']})",
                        field_name=field,
                        fixable=False
                    ))
            
            # Check allowed values
            if "allowed" in constraint:
                invalid = gdf[~gdf[field].isin(constraint["allowed"])]
                if len(invalid) > 0:
                    result.issues.append(ValidationIssue(
                        issue_type=ValidationType.ATTRIBUTE,
                        severity=ValidationSeverity.ERROR,
                        message=f"Field '{field}' has {len(invalid)} invalid values",
                        field_name=field,
                        fixable=False
                    ))
        
        return result
    
    def _type_matches(self, actual: str, expected: str) -> bool:
        """Check if actual type matches expected type."""
        type_map = {
            "int": ["int8", "int16", "int32", "int64"],
            "float": ["float32", "float64"],
            "str": ["object", "string"],
            "bool": ["bool"],
        }
        return actual in type_map.get(expected, [expected])


# Convenience functions
def quick_validate(
    layer: Union[QgsVectorLayer, gpd.GeoDataFrame]
) -> ValidationResult:
    """Quick validation with default settings."""
    validator = DataValidator()
    return validator.validate_layer(layer)


def validate_and_repair(
    layer: Union[QgsVectorLayer, gpd.GeoDataFrame],
    output_path: Optional[str] = None
) -> Tuple[ValidationResult, Union[QgsVectorLayer, gpd.GeoDataFrame]]:
    """Validate and automatically repair issues."""
    validator = DataValidator()
    result = validator.validate_layer(layer)
    repaired = validator.auto_repair(layer, output_path)
    return result, repaired
