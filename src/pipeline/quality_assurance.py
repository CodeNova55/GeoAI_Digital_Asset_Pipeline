"""
Quality Assurance Module
========================

Provides comprehensive quality assurance workflows for geospatial data.
Supports geometry validation, attribute checks, spatial accuracy,
temporal consistency, and topology error detection.

Example:
    >>> qa = QualityAssurance()
    >>> results = qa.run_checks(gdf)
    >>> qa.generate_report(results, "qa_report.html")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import json

import numpy as np

try:
    import geopandas as gpd
    from shapely.geometry import shape, mapping
    from shapely.validation import make_valid, explain_validity
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    from osgeo import ogr, osr
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

from src.utils.config import Config
from src.workflow.logger import PipelineLogger


class CheckType(Enum):
    """Types of quality checks."""
    GEOMETRY = "geometry"
    ATTRIBUTE = "attribute"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    TOPOLOGY = "topology"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"


class CheckSeverity(Enum):
    """Severity levels for check results."""
    PASS = "pass"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CheckResult:
    """Result of a single quality check."""
    
    check_name: str
    check_type: CheckType
    severity: CheckSeverity
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    affected_features: Optional[List[int]] = None
    fixable: bool = False
    fix_applied: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_name": self.check_name,
            "check_type": self.check_type.value,
            "severity": self.severity.value,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "affected_features": self.affected_features,
            "fixable": self.fixable,
            "fix_applied": self.fix_applied,
            "timestamp": self.timestamp
        }


@dataclass
class QualityReport:
    """Complete quality assurance report."""
    
    data_name: str
    total_features: int = 0
    checks_run: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    results: List[CheckResult] = field(default_factory=list)
    overall_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def pass_rate(self) -> float:
        """Calculate check pass rate."""
        if self.checks_run == 0:
            return 100.0
        return (self.checks_passed / self.checks_run) * 100
    
    @property
    def issue_summary(self) -> Dict[str, int]:
        """Get count of issues by severity."""
        summary = {s.value: 0 for s in CheckSeverity}
        for result in self.results:
            if not result.passed:
                summary[result.severity.value] += 1
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data_name": self.data_name,
            "total_features": self.total_features,
            "checks_run": self.checks_run,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "pass_rate": self.pass_rate,
            "overall_score": self.overall_score,
            "issue_summary": self.issue_summary,
            "results": [r.to_dict() for r in self.results],
            "timestamp": self.timestamp
        }


class QualityAssurance:
    """
    Comprehensive quality assurance for geospatial data.
    
    This class provides a complete QA workflow including geometry
    validation, attribute checks, spatial accuracy assessment,
    temporal consistency checks, and topology error detection.
    
    Attributes:
        config: Configuration object with QA parameters.
        logger: Logger instance.
        checks: List of configured quality checks.
        
    Example:
        >>> qa = QualityAssurance()
        >>> report = qa.run_all_checks(gdf)
        >>> print(f"Overall score: {report.overall_score:.1f}%")
        >>> qa.generate_report(report, "qa_report.html")
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        logger: Optional[PipelineLogger] = None
    ):
        """
        Initialize the quality assurance module.
        
        Args:
            config: Configuration object.
            logger: Logger instance.
        """
        self.config = config or Config.default()
        self.logger = logger or PipelineLogger.get_logger("QualityAssurance")
        
        # Load QA config
        qa_config = {}
        if hasattr(self.config, 'pipeline'):
            qa_config = self.config.pipeline.get('quality_assurance', {})
        
        self.enabled_checks = qa_config.get('checks', [
            "geometry_validity",
            "attribute_completeness",
            "spatial_accuracy",
            "topology_errors"
        ])
        
        self.tolerance = qa_config.get('tolerance', {
            "positional": 1.0,
            "area": 0.05
        })
        
        self.auto_repair = qa_config.get('auto_repair', True)
        
        self.logger.info(f"QualityAssurance initialized with {len(self.enabled_checks)} checks")
    
    def run_all_checks(
        self,
        data: Union[gpd.GeoDataFrame, str],
        schema: Optional[Dict[str, Any]] = None
    ) -> QualityReport:
        """
        Run all enabled quality checks.
        
        Args:
            data: Data to check (GeoDataFrame or path).
            schema: Optional schema for validation.
            
        Returns:
            QualityReport with all check results.
        """
        # Load data
        gdf = self._load_data(data)
        
        report = QualityReport(
            data_name=gdf.name if hasattr(gdf, 'name') else "Dataset",
            total_features=len(gdf)
        )
        
        # Run geometry checks
        if "geometry_validity" in self.enabled_checks:
            report.results.append(self._check_geometry_validity(gdf))
        
        if "empty_geometry" in self.enabled_checks:
            report.results.append(self._check_empty_geometry(gdf))
        
        # Run attribute checks
        if "attribute_completeness" in self.enabled_checks:
            report.results.append(self._check_attribute_completeness(gdf))
        
        if "unique_ids" in self.enabled_checks:
            report.results.append(self._check_unique_ids(gdf))
        
        # Run spatial checks
        if "spatial_accuracy" in self.enabled_checks:
            report.results.append(self._check_spatial_accuracy(gdf))
        
        if "coordinate_range" in self.enabled_checks:
            report.results.append(self._check_coordinate_range(gdf))
        
        # Run topology checks
        if "topology_errors" in self.enabled_checks:
            report.results.extend(self._check_topology(gdf))
        
        if "overlaps" in self.enabled_checks:
            report.results.append(self._check_overlaps(gdf))
        
        # Run temporal checks
        if "temporal_consistency" in self.enabled_checks:
            report.results.append(self._check_temporal_consistency(gdf))
        
        # Run schema validation
        if schema:
            report.results.append(self._check_schema(gdf, schema))
        
        # Calculate summary
        report.checks_run = len(report.results)
        report.checks_passed = sum(1 for r in report.results if r.passed)
        report.checks_failed = report.checks_run - report.checks_passed
        
        # Calculate overall score
        report.overall_score = self._calculate_overall_score(report)
        
        self.logger.info(
            f"QA complete: {report.overall_score:.1f}% score "
            f"({report.checks_passed}/{report.checks_run} checks passed)"
        )
        
        return report
    
    def _load_data(self, data: Union[gpd.GeoDataFrame, str]) -> gpd.GeoDataFrame:
        """Load data from path or return GeoDataFrame."""
        if isinstance(data, str):
            if not GEOPANDAS_AVAILABLE:
                raise ImportError("geopandas required for data loading")
            return gpd.read_file(data)
        return data
    
    def _check_geometry_validity(self, gdf: gpd.GeoDataFrame) -> CheckResult:
        """Check geometry validity."""
        invalid_count = 0
        invalid_ids = []
        
        for idx, geom in enumerate(gdf.geometry):
            if geom is not None and not geom.is_valid:
                invalid_count += 1
                invalid_ids.append(idx)
        
        passed = invalid_count == 0
        severity = CheckSeverity.ERROR if invalid_count > len(gdf) * 0.1 else CheckSeverity.WARNING
        
        return CheckResult(
            check_name="geometry_validity",
            check_type=CheckType.GEOMETRY,
            severity=severity,
            passed=passed,
            message=f"{invalid_count} invalid geometries found ({invalid_count/len(gdf)*100:.1f}%)",
            details={
                "invalid_count": invalid_count,
                "total_features": len(gdf)
            },
            affected_features=invalid_ids[:100],  # Limit to first 100
            fixable=True
        )
    
    def _check_empty_geometry(self, gdf: gpd.GeoDataFrame) -> CheckResult:
        """Check for empty geometries."""
        empty_count = gdf.geometry.is_empty.sum()
        
        return CheckResult(
            check_name="empty_geometry",
            check_type=CheckType.GEOMETRY,
            severity=CheckSeverity.ERROR if empty_count > 0 else CheckSeverity.PASS,
            passed=empty_count == 0,
            message=f"{empty_count} empty geometries found",
            details={"empty_count": int(empty_count)},
            fixable=False
        )
    
    def _check_attribute_completeness(self, gdf: gpd.GeoDataFrame) -> CheckResult:
        """Check attribute completeness."""
        completeness_scores = []
        null_fields = {}
        
        for col in gdf.columns:
            if col == "geometry":
                continue
            
            null_count = gdf[col].isnull().sum()
            null_pct = (null_count / len(gdf)) * 100
            completeness_scores.append(100 - null_pct)
            
            if null_count > 0:
                null_fields[col] = {"null_count": int(null_count), "null_pct": float(null_pct)}
        
        avg_completeness = np.mean(completeness_scores) if completeness_scores else 100.0
        
        return CheckResult(
            check_name="attribute_completeness",
            check_type=CheckType.ATTRIBUTE,
            severity=CheckSeverity.WARNING if avg_completeness < 95 else CheckSeverity.PASS,
            passed=avg_completeness >= 90,
            message=f"Average attribute completeness: {avg_completeness:.1f}%",
            details={
                "average_completeness": float(avg_completeness),
                "fields_with_nulls": null_fields
            },
            fixable=False
        )
    
    def _check_unique_ids(self, gdf: gpd.GeoDataFrame) -> CheckResult:
        """Check for unique IDs."""
        # Look for ID-like columns
        id_cols = [col for col in gdf.columns if 'id' in col.lower()]
        
        if not id_cols:
            return CheckResult(
                check_name="unique_ids",
                check_type=CheckType.ATTRIBUTE,
                severity=CheckSeverity.INFO,
                passed=True,
                message="No ID column found, skipping uniqueness check",
                details={"id_columns": id_cols}
            )
        
        duplicates_found = False
        duplicate_counts = {}
        
        for col in id_cols:
            dup_count = gdf[col].duplicated().sum()
            if dup_count > 0:
                duplicates_found = True
                duplicate_counts[col] = int(dup_count)
        
        return CheckResult(
            check_name="unique_ids",
            check_type=CheckType.ATTRIBUTE,
            severity=CheckSeverity.ERROR if duplicates_found else CheckSeverity.PASS,
            passed=not duplicates_found,
            message=f"Duplicate IDs found in {len(duplicate_counts)} column(s)",
            details={"duplicate_counts": duplicate_counts},
            fixable=False
        )
    
    def _check_spatial_accuracy(self, gdf: gpd.GeoDataFrame) -> CheckResult:
        """Check spatial accuracy metrics."""
        issues = []
        
        # Check for zero-area polygons
        if gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            zero_area = (gdf.geometry.area == 0).sum()
            if zero_area > 0:
                issues.append(f"{zero_area} features with zero area")
        
        # Check for very small features
        if len(gdf) > 0 and gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            areas = gdf.geometry.area
            if len(areas) > 0:
                min_area = areas.min()
                if min_area < self.tolerance.get("area", 0.05):
                    issues.append(f"Minimum area ({min_area:.6f}) below tolerance")
        
        passed = len(issues) == 0
        
        return CheckResult(
            check_name="spatial_accuracy",
            check_type=CheckType.SPATIAL,
            severity=CheckSeverity.WARNING if issues else CheckSeverity.PASS,
            passed=passed,
            message="; ".join(issues) if issues else "Spatial accuracy checks passed",
            details={"issues": issues},
            fixable=False
        )
    
    def _check_coordinate_range(self, gdf: gpd.GeoDataFrame) -> CheckResult:
        """Check coordinate ranges."""
        bounds = gdf.total_bounds
        
        issues = []
        
        # Check for coordinates outside valid WGS84 range
        if gdf.crs and "EPSG:4326" in str(gdf.crs):
            if bounds[0] < -180 or bounds[2] > 180:
                issues.append("X coordinates outside valid longitude range")
            if bounds[1] < -90 or bounds[3] > 90:
                issues.append("Y coordinates outside valid latitude range")
        
        # Check for suspiciously large coordinates
        if abs(bounds[0]) > 1000000 or abs(bounds[3]) > 1000000:
            issues.append("Coordinates suggest projected CRS (not WGS84)")
        
        return CheckResult(
            check_name="coordinate_range",
            check_type=CheckType.SPATIAL,
            severity=CheckSeverity.WARNING if issues else CheckSeverity.PASS,
            passed=len(issues) == 0,
            message="; ".join(issues) if issues else "Coordinate range valid",
            details={"bounds": bounds.tolist(), "issues": issues}
        )
    
    def _check_topology(self, gdf: gpd.GeoDataFrame) -> List[CheckResult]:
        """Check topological relationships."""
        results = []
        
        # Check for duplicate geometries
        geom_hashes = []
        for geom in gdf.geometry:
            if geom is not None:
                try:
                    geom_hashes.append(hash(geom.wkb))
                except:
                    geom_hashes.append(None)
            else:
                geom_hashes.append(None)
        
        duplicates = sum(1 for i, h in enumerate(geom_hashes) if h is not None and geom_hashes[:i].count(h) > 0)
        
        results.append(CheckResult(
            check_name="duplicate_geometries",
            check_type=CheckType.TOPOLOGY,
            severity=CheckSeverity.WARNING if duplicates > 0 else CheckSeverity.PASS,
            passed=duplicates == 0,
            message=f"{duplicates} duplicate geometries found",
            details={"duplicate_count": duplicates},
            fixable=True
        ))
        
        return results
    
    def _check_overlaps(self, gdf: gpd.GeoDataFrame) -> CheckResult:
        """Check for overlapping polygons."""
        if not gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            return CheckResult(
                check_name="overlaps",
                check_type=CheckType.TOPOLOGY,
                severity=CheckSeverity.INFO,
                passed=True,
                message="Skipping overlap check (not polygon data)",
                details={}
            )
        
        overlap_count = 0
        
        # Simple overlap check (for small datasets)
        if len(gdf) < 1000:
            from shapely.strtree import STRtree
            
            geometries = gdf.geometry.tolist()
            tree = STRtree(geometries)
            
            checked = set()
            for i, geom in enumerate(geometries):
                if geom is None:
                    continue
                
                potential = tree.query(geom)
                for j in potential:
                    if i >= j or (i, j) in checked:
                        continue
                    checked.add((i, j))
                    
                    if geometries[j] is not None and geom.intersects(geometries[j]):
                        intersection = geom.intersection(geometries[j])
                        if intersection.area > 0.001:
                            overlap_count += 1
        
        return CheckResult(
            check_name="overlaps",
            check_type=CheckType.TOPOLOGY,
            severity=CheckSeverity.WARNING if overlap_count > 0 else CheckSeverity.PASS,
            passed=overlap_count == 0,
            message=f"{overlap_count} overlapping polygon pairs found",
            details={"overlap_count": overlap_count},
            fixable=True
        )
    
    def _check_temporal_consistency(self, gdf: gpd.GeoDataFrame) -> CheckResult:
        """Check temporal consistency."""
        # Look for date/time columns
        date_cols = [col for col in gdf.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if not date_cols:
            return CheckResult(
                check_name="temporal_consistency",
                check_type=CheckType.TEMPORAL,
                severity=CheckSeverity.INFO,
                passed=True,
                message="No temporal columns found",
                details={}
            )
        
        issues = []
        
        for col in date_cols:
            # Check for future dates
            try:
                max_date = gdf[col].max()
                if pd.notna(max_date) and max_date > pd.Timestamp.now():
                    issues.append(f"Future dates found in {col}")
            except:
                pass
            
            # Check for very old dates
            try:
                min_date = gdf[col].min()
                if pd.notna(min_date) and min_date < pd.Timestamp("1900-01-01"):
                    issues.append(f"Suspiciously old dates in {col}")
            except:
                pass
        
        return CheckResult(
            check_name="temporal_consistency",
            check_type=CheckType.TEMPORAL,
            severity=CheckSeverity.WARNING if issues else CheckSeverity.PASS,
            passed=len(issues) == 0,
            message="; ".join(issues) if issues else "Temporal consistency checks passed",
            details={"issues": issues}
        )
    
    def _check_schema(self, gdf: gpd.GeoDataFrame, schema: Dict[str, Any]) -> CheckResult:
        """Check data against schema."""
        issues = []
        
        # Check required fields
        required = schema.get("required_fields", [])
        missing = [f for f in required if f not in gdf.columns]
        if missing:
            issues.append(f"Missing required fields: {missing}")
        
        # Check field types
        field_types = schema.get("field_types", {})
        for field, expected_type in field_types.items():
            if field in gdf.columns:
                actual_type = str(gdf[field].dtype)
                if not self._type_matches(actual_type, expected_type):
                    issues.append(f"Field '{field}' type mismatch: expected {expected_type}, got {actual_type}")
        
        # Check value constraints
        constraints = schema.get("constraints", {})
        for field, constraint in constraints.items():
            if field in gdf.columns:
                if "min" in constraint:
                    invalid = (gdf[field] < constraint["min"]).sum()
                    if invalid > 0:
                        issues.append(f"{invalid} values below minimum in {field}")
                
                if "max" in constraint:
                    invalid = (gdf[field] > constraint["max"]).sum()
                    if invalid > 0:
                        issues.append(f"{invalid} values above maximum in {field}")
        
        return CheckResult(
            check_name="schema_validation",
            check_type=CheckType.CONSISTENCY,
            severity=CheckSeverity.ERROR if issues else CheckSeverity.PASS,
            passed=len(issues) == 0,
            message="; ".join(issues) if issues else "Schema validation passed",
            details={"issues": issues}
        )
    
    def _type_matches(self, actual: str, expected: str) -> bool:
        """Check if actual type matches expected."""
        type_map = {
            "int": ["int8", "int16", "int32", "int64"],
            "float": ["float32", "float64"],
            "str": ["object", "string"],
            "bool": ["bool"],
            "datetime": ["datetime64"]
        }
        return actual in type_map.get(expected, [expected])
    
    def _calculate_overall_score(self, report: QualityReport) -> float:
        """Calculate overall quality score."""
        if not report.results:
            return 100.0
        
        weights = {
            CheckSeverity.CRITICAL: 0,
            CheckSeverity.ERROR: 20,
            CheckSeverity.WARNING: 50,
            CheckSeverity.INFO: 80,
            CheckSeverity.PASS: 100
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for result in report.results:
            weight = 1.0
            if result.severity == CheckSeverity.CRITICAL:
                weight = 3.0
            elif result.severity == CheckSeverity.ERROR:
                weight = 2.0
            
            score = weights.get(result.severity, 50)
            weighted_sum += score * weight
            total_weight += weight
        
        return (weighted_sum / total_weight) if total_weight > 0 else 0.0
    
    def auto_fix(self, gdf: gpd.GeoDataFrame, report: QualityReport) -> gpd.GeoDataFrame:
        """
        Automatically fix fixable issues.
        
        Args:
            gdf: Input GeoDataFrame.
            report: Quality report with issues.
            
        Returns:
            Fixed GeoDataFrame.
        """
        if not self.auto_repair:
            return gdf
        
        fixed_gdf = gdf.copy()
        
        for result in report.results:
            if result.fixable and not result.passed:
                if result.check_name == "geometry_validity":
                    fixed_gdf = self._fix_invalid_geometries(fixed_gdf)
                elif result.check_name == "duplicate_geometries":
                    fixed_gdf = self._remove_duplicate_geometries(fixed_gdf)
        
        return fixed_gdf
    
    def _fix_invalid_geometries(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Fix invalid geometries."""
        for idx, geom in enumerate(gdf.geometry):
            if geom is not None and not geom.is_valid:
                try:
                    gdf.at[idx, 'geometry'] = make_valid(geom)
                except Exception as e:
                    self.logger.warning(f"Could not fix geometry at index {idx}: {e}")
        return gdf
    
    def _remove_duplicate_geometries(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Remove duplicate geometries."""
        return gdf.drop_duplicates(subset=['geometry'])
    
    def generate_report(
        self,
        report: QualityReport,
        output_path: str,
        format: str = "html"
    ) -> bool:
        """
        Generate quality report.
        
        Args:
            report: Quality report to output.
            output_path: Output file path.
            format: Report format (html, json, txt).
            
        Returns:
            True if generation successful.
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                with open(output_path, 'w') as f:
                    json.dump(report.to_dict(), f, indent=2)
            
            elif format == "html":
                self._generate_html_report(report, output_path)
            
            elif format == "txt":
                self._generate_text_report(report, output_path)
            
            else:
                self.logger.error(f"Unknown format: {format}")
                return False
            
            self.logger.info(f"Generated report: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return False
    
    def _generate_html_report(self, report: QualityReport, output_path: str) -> None:
        """Generate HTML report."""
        score_color = "#28a745" if report.overall_score >= 80 else "#ffc107" if report.overall_score >= 60 else "#dc3545"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quality Assurance Report - {report.data_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
                .score {{ font-size: 48px; font-weight: bold; color: {score_color}; text-align: center; padding: 20px; }}
                .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .summary-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
                .summary-card h3 {{ margin: 0; color: #666; font-size: 14px; }}
                .summary-card p {{ margin: 10px 0 0; font-size: 24px; font-weight: bold; color: #333; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #007bff; color: white; }}
                tr:hover {{ background: #f5f5f5; }}
                .pass {{ color: #28a745; }}
                .fail {{ color: #dc3545; }}
                .warning {{ color: #ffc107; }}
                .timestamp {{ color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Quality Assurance Report</h1>
                <p><strong>Dataset:</strong> {report.data_name}</p>
                <p class="timestamp">Generated: {report.timestamp}</p>
                
                <div class="score">Overall Score: {report.overall_score:.1f}%</div>
                
                <div class="summary">
                    <div class="summary-card">
                        <h3>Total Features</h3>
                        <p>{report.total_features}</p>
                    </div>
                    <div class="summary-card">
                        <h3>Checks Run</h3>
                        <p>{report.checks_run}</p>
                    </div>
                    <div class="summary-card">
                        <h3>Checks Passed</h3>
                        <p class="pass">{report.checks_passed}</p>
                    </div>
                    <div class="summary-card">
                        <h3>Checks Failed</h3>
                        <p class="fail">{report.checks_failed}</p>
                    </div>
                    <div class="summary-card">
                        <h3>Pass Rate</h3>
                        <p>{report.pass_rate:.1f}%</p>
                    </div>
                </div>
                
                <h2>Issue Summary</h2>
                <table>
                    <tr><th>Severity</th><th>Count</th></tr>
                    <tr><td class="fail">Critical</td><td>{report.issue_summary.get('critical', 0)}</td></tr>
                    <tr><td class="fail">Error</td><td>{report.issue_summary.get('error', 0)}</td></tr>
                    <tr><td class="warning">Warning</td><td>{report.issue_summary.get('warning', 0)}</td></tr>
                    <tr><td class="pass">Info</td><td>{report.issue_summary.get('info', 0)}</td></tr>
                </table>
                
                <h2>Detailed Results</h2>
                <table>
                    <tr>
                        <th>Check</th>
                        <th>Type</th>
                        <th>Status</th>
                        <th>Message</th>
                    </tr>
        """
        
        for result in report.results:
            status_class = "pass" if result.passed else "fail"
            status_text = "PASS" if result.passed else "FAIL"
            
            html += f"""
                    <tr>
                        <td>{result.check_name}</td>
                        <td>{result.check_type.value}</td>
                        <td class="{status_class}">{status_text}</td>
                        <td>{result.message}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
    
    def _generate_text_report(self, report: QualityReport, output_path: str) -> None:
        """Generate text report."""
        lines = [
            "=" * 60,
            "QUALITY ASSURANCE REPORT",
            "=" * 60,
            f"Dataset: {report.data_name}",
            f"Generated: {report.timestamp}",
            "",
            f"OVERALL SCORE: {report.overall_score:.1f}%",
            "",
            "SUMMARY",
            "-" * 40,
            f"Total Features: {report.total_features}",
            f"Checks Run: {report.checks_run}",
            f"Checks Passed: {report.checks_passed}",
            f"Checks Failed: {report.checks_failed}",
            f"Pass Rate: {report.pass_rate:.1f}%",
            "",
            "ISSUE SUMMARY",
            "-" * 40,
        ]
        
        for severity, count in report.issue_summary.items():
            lines.append(f"  {severity.upper()}: {count}")
        
        lines.extend([
            "",
            "DETAILED RESULTS",
            "-" * 40,
        ])
        
        for result in report.results:
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"[{status}] {result.check_name}: {result.message}")
        
        lines.append("=" * 60)
        
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))


# Import pandas for temporal checks
try:
    import pandas as pd
except ImportError:
    pd = None


# Convenience functions
def quick_qa(data: Union[gpd.GeoDataFrame, str]) -> QualityReport:
    """Quick quality assessment with default checks."""
    qa = QualityAssurance()
    return qa.run_all_checks(data)


def validate_and_fix(data: Union[gpd.GeoDataFrame, str]) -> Tuple[gpd.GeoDataFrame, QualityReport]:
    """Validate data and auto-fix issues."""
    qa = QualityAssurance()
    report = qa.run_all_checks(data)
    gdf = qa._load_data(data)
    fixed_gdf = qa.auto_fix(gdf, report)
    return fixed_gdf, report
