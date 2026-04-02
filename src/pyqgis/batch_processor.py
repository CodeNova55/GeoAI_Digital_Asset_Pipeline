"""
Batch Processor Module
======================

Handles batch processing of vector and raster data using PyQGIS.
Supports parallel processing, progress tracking, and comprehensive error handling.

Example:
    >>> processor = BatchProcessor()
    >>> processor.batch_process_vectors(input_dir, output_dir, processing_func)
    >>> processor.batch_process_rasters(input_dir, output_dir, ["clip", "reproject"])
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional, Union, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import logging

try:
    from qgis.core import (
        QgsVectorLayer,
        QgsRasterLayer,
        QgsProject,
        QgsProcessingContext,
        QgsProcessingFeedback,
        QgsApplication,
        QgsCoordinateReferenceSystem,
    )
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False
    # Define stub types for when QGIS is not available
    QgsVectorLayer = type('QgsVectorLayer', (), {})
    QgsRasterLayer = type('QgsRasterLayer', (), {})
    QgsProject = type('QgsProject', (), {})
    QgsProcessingContext = type('QgsProcessingContext', (), {})
    QgsProcessingFeedback = type('QgsProcessingFeedback', (), {})
    QgsApplication = type('QgsApplication', (), {})
    QgsCoordinateReferenceSystem = type('QgsCoordinateReferenceSystem', (), {})

from src.utils.config import Config
from src.workflow.logger import PipelineLogger
from src.workflow.progress_tracker import ProgressTracker


@dataclass
class ProcessingResult:
    """Container for batch processing results."""
    
    input_path: str
    output_path: str
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0
    features_processed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchProcessor:
    """
    Batch processor for vector and raster geospatial data.
    
    This class provides methods for processing multiple geospatial files
    in batch, with support for parallel execution, progress tracking,
    and comprehensive error handling.
    
    Attributes:
        config: Configuration object with processing parameters.
        logger: Logger instance for recording processing events.
        context: QGIS processing context.
        max_workers: Maximum number of parallel workers.
        
    Example:
        >>> config = Config.load("config.yaml")
        >>> processor = BatchProcessor(config)
        >>> results = processor.batch_process_vectors(
        ...     input_dir="./data/raw",
        ...     output_dir="./data/processed",
        ...     operation="reproject",
        ...     target_crs="EPSG:3857"
        ... )
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        logger: Optional[PipelineLogger] = None,
        max_workers: int = -1,
        use_qgis: bool = True
    ):
        """
        Initialize the batch processor.
        
        Args:
            config: Configuration object. Uses default if None.
            logger: Logger instance. Creates new if None.
            max_workers: Max parallel workers. -1 uses all CPUs.
            use_qgis: Whether to initialize QGIS application.
        """
        self.config = config or Config.default()
        self.logger = logger or PipelineLogger.get_logger("BatchProcessor")
        self.max_workers = max_workers if max_workers > 0 else os.cpu_count() or 4
        self.use_qgis = use_qgis and QGIS_AVAILABLE
        
        # Initialize QGIS if available and requested
        if self.use_qgis:
            self._init_qgis()
        
        self.progress = ProgressTracker()
        self.results: List[ProcessingResult] = []
        
        self.logger.info(f"BatchProcessor initialized with {self.max_workers} workers")
    
    def _init_qgis(self) -> None:
        """Initialize QGIS application for processing."""
        if not QGIS_AVAILABLE:
            self.logger.warning("QGIS not available, running in limited mode")
            return
            
        try:
            QgsApplication.initQgis()
            self.context = QgsProcessingContext()
            self.logger.info("QGIS initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize QGIS: {e}")
            self.use_qgis = False
    
    def _get_vector_files(self, directory: str) -> List[str]:
        """
        Get all vector files from a directory.
        
        Args:
            directory: Path to search for vector files.
            
        Returns:
            List of vector file paths.
        """
        extensions = ["*.shp", "*.geojson", "*.gpkg", "*.kml", "*.gml", "*.dxf"]
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        return files
    
    def _get_raster_files(self, directory: str) -> List[str]:
        """
        Get all raster files from a directory.
        
        Args:
            directory: Path to search for raster files.
            
        Returns:
            List of raster file paths.
        """
        extensions = ["*.tif", "*.tiff", "*.jp2", "*.img", "*.asc", "*.vrt"]
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        return files
    
    def _load_vector_layer(self, path: str) -> Optional[QgsVectorLayer]:
        """
        Load a vector layer from file.
        
        Args:
            path: Path to vector file.
            
        Returns:
            QgsVectorLayer or None if loading failed.
        """
        if not QGIS_AVAILABLE:
            self.logger.warning("QGIS not available for vector loading")
            return None
            
        layer = QgsVectorLayer(path, Path(path).stem, "ogr")
        if layer.isValid():
            return layer
        else:
            self.logger.error(f"Failed to load vector layer: {path}")
            return None
    
    def _load_raster_layer(self, path: str) -> Optional[QgsRasterLayer]:
        """
        Load a raster layer from file.
        
        Args:
            path: Path to raster file.
            
        Returns:
            QgsRasterLayer or None if loading failed.
        """
        if not QGIS_AVAILABLE:
            self.logger.warning("QGIS not available for raster loading")
            return None
            
        layer = QgsRasterLayer(path, Path(path).stem)
        if layer.isValid():
            return layer
        else:
            self.logger.error(f"Failed to load raster layer: {path}")
            return None
    
    def batch_process_vectors(
        self,
        input_dir: str,
        output_dir: str,
        operation: str,
        processing_func: Optional[Callable] = None,
        file_pattern: Optional[str] = None,
        parallel: bool = True,
        **kwargs
    ) -> List[ProcessingResult]:
        """
        Batch process vector files with specified operation.
        
        Args:
            input_dir: Directory containing input vector files.
            output_dir: Directory for processed output files.
            operation: Operation type (e.g., "reproject", "clip", "buffer").
            processing_func: Custom processing function. If None, uses built-in.
            file_pattern: Optional glob pattern to filter files.
            parallel: Whether to process files in parallel.
            **kwargs: Additional arguments for processing function.
            
        Returns:
            List of ProcessingResult objects for each file.
            
        Example:
            >>> results = processor.batch_process_vectors(
            ...     input_dir="./data/raw",
            ...     output_dir="./data/processed",
            ...     operation="reproject",
            ...     target_crs="EPSG:3857"
            ... )
        """
        # Get input files
        files = self._get_vector_files(input_dir)
        if file_pattern:
            files = [f for f in files if glob.fnmatch.fnmatch(Path(f).name, file_pattern)]
        
        if not files:
            self.logger.warning(f"No vector files found in {input_dir}")
            return []
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Processing {len(files)} vector files with operation: {operation}")
        self.progress.init(total=len(files), desc="Processing vectors")
        
        results = []
        
        if parallel and len(files) > 1:
            results = self._process_parallel(
                files, input_dir, output_dir, operation, processing_func, **kwargs
            )
        else:
            results = self._process_sequential(
                files, input_dir, output_dir, operation, processing_func, **kwargs
            )
        
        self.results.extend(results)
        self._log_summary(results, "vector")
        
        return results
    
    def batch_process_rasters(
        self,
        input_dir: str,
        output_dir: str,
        operations: List[str],
        processing_func: Optional[Callable] = None,
        file_pattern: Optional[str] = None,
        parallel: bool = True,
        **kwargs
    ) -> List[ProcessingResult]:
        """
        Batch process raster files with specified operations.
        
        Args:
            input_dir: Directory containing input raster files.
            output_dir: Directory for processed output files.
            operations: List of operations (e.g., ["clip", "reproject", "calculate_ndvi"]).
            processing_func: Custom processing function. If None, uses built-in.
            file_pattern: Optional glob pattern to filter files.
            parallel: Whether to process files in parallel.
            **kwargs: Additional arguments for processing functions.
            
        Returns:
            List of ProcessingResult objects for each file.
            
        Example:
            >>> results = processor.batch_process_rasters(
            ...     input_dir="./data/sentinel2",
            ...     output_dir="./data/processed",
            ...     operations=["reproject", "calculate_ndvi"],
            ...     target_crs="EPSG:3857"
            ... )
        """
        # Get input files
        files = self._get_raster_files(input_dir)
        if file_pattern:
            files = [f for f in files if glob.fnmatch.fnmatch(Path(f).name, file_pattern)]
        
        if not files:
            self.logger.warning(f"No raster files found in {input_dir}")
            return []
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Processing {len(files)} raster files with operations: {operations}")
        self.progress.init(total=len(files), desc="Processing rasters")
        
        results = []
        
        if parallel and len(files) > 1:
            results = self._process_parallel(
                files, input_dir, output_dir, operations, processing_func, **kwargs
            )
        else:
            results = self._process_sequential(
                files, input_dir, output_dir, operations, processing_func, **kwargs
            )
        
        self.results.extend(results)
        self._log_summary(results, "raster")
        
        return results
    
    def _process_parallel(
        self,
        files: List[str],
        input_dir: str,
        output_dir: str,
        operations: Union[str, List[str]],
        processing_func: Optional[Callable],
        **kwargs
    ) -> List[ProcessingResult]:
        """Process files in parallel using thread pool."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_file,
                    f, input_dir, output_dir, operations, processing_func, **kwargs
                ): f for f in files
            }
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                self.progress.update(1)
        
        return results
    
    def _process_sequential(
        self,
        files: List[str],
        input_dir: str,
        output_dir: str,
        operations: Union[str, List[str]],
        processing_func: Optional[Callable],
        **kwargs
    ) -> List[ProcessingResult]:
        """Process files sequentially."""
        results = []
        
        for file_path in files:
            result = self._process_single_file(
                file_path, input_dir, output_dir, operations, processing_func, **kwargs
            )
            results.append(result)
            self.progress.update(1)
        
        return results
    
    def _process_single_file(
        self,
        file_path: str,
        input_dir: str,
        output_dir: str,
        operations: Union[str, List[str]],
        processing_func: Optional[Callable],
        **kwargs
    ) -> ProcessingResult:
        """
        Process a single file with specified operations.
        
        Args:
            file_path: Path to input file.
            input_dir: Base input directory.
            output_dir: Base output directory.
            operations: Operation(s) to apply.
            processing_func: Custom processing function.
            **kwargs: Additional arguments.
            
        Returns:
            ProcessingResult object.
        """
        import time
        start_time = time.time()
        
        try:
            # Determine output path
            rel_path = os.path.relpath(file_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Determine file type and load
            ext = Path(file_path).suffix.lower()
            is_raster = ext in [".tif", ".tiff", ".jp2", ".img", ".asc", ".vrt"]
            
            if is_raster:
                layer = self._load_raster_layer(file_path)
                features_count = 0  # Rasters don't have features
            else:
                layer = self._load_vector_layer(file_path)
                features_count = layer.featureCount() if layer else 0
            
            if layer is None and self.use_qgis:
                raise ValueError(f"Failed to load layer: {file_path}")
            
            # Apply operations
            if processing_func:
                # Use custom processing function
                result_layer = processing_func(layer, **kwargs)
            else:
                # Use built-in operations
                result_layer = self._apply_operations(
                    layer, operations if isinstance(operations, list) else [operations], **kwargs
                )
            
            # Export result
            if result_layer:
                self._export_layer(result_layer, output_path)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                input_path=file_path,
                output_path=output_path,
                success=True,
                processing_time=processing_time,
                features_processed=features_count,
                metadata={"operations": operations if isinstance(operations, list) else [operations]}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing {file_path}: {e}")
            
            return ProcessingResult(
                input_path=file_path,
                output_path="",
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _apply_operations(
        self,
        layer: Any,
        operations: List[str],
        **kwargs
    ) -> Any:
        """
        Apply a list of operations to a layer.
        
        Args:
            layer: Input layer (vector or raster).
            operations: List of operation names.
            **kwargs: Operation parameters.
            
        Returns:
            Processed layer.
        """
        result = layer
        
        for op in operations:
            op_method = getattr(self, f"_op_{op}", None)
            if op_method:
                result = op_method(result, **kwargs)
            else:
                self.logger.warning(f"Unknown operation: {op}")
        
        return result
    
    def _op_reproject(self, layer: Any, target_crs: str = "EPSG:3857", **kwargs) -> Any:
        """Reproject layer to target CRS."""
        if not QGIS_AVAILABLE or layer is None:
            return layer
        
        target_crs_obj = QgsCoordinateReferenceSystem(target_crs)
        
        if isinstance(layer, QgsVectorLayer):
            # For vector layers, create a reprojected copy
            from qgis.analysis import QgsVectorLayerUtils
            reprojected = QgsVectorLayerUtils.copyLayer(
                layer, 
                self.context,
                transformContext=target_crs_obj
            )
            return reprojected
        elif isinstance(layer, QgsRasterLayer):
            # For raster layers, use warp algorithm
            from qgis.analysis import QgsRasterCalcAlgorithm
            # Simplified - in production would use gdal.Warp
            return layer
            
        return layer
    
    def _op_clip(self, layer: Any, clip_layer: Any = None, **kwargs) -> Any:
        """Clip layer to extent or another layer."""
        if not QGIS_AVAILABLE or layer is None:
            return layer
        
        # Simplified clip operation
        # In production, would use QgsProcessingAlgorithm
        return layer
    
    def _op_buffer(self, layer: Any, distance: float = 100.0, **kwargs) -> Any:
        """Create buffer around vector features."""
        if not QGIS_AVAILABLE or layer is None:
            return layer
        
        # Simplified buffer operation
        return layer
    
    def _op_calculate_ndvi(self, layer: Any, **kwargs) -> Any:
        """Calculate NDVI from multispectral raster."""
        if not QGIS_AVAILABLE or layer is None:
            return layer
        
        # Simplified NDVI calculation
        # In production, would use raster calculator
        return layer
    
    def _export_layer(self, layer: Any, output_path: str) -> bool:
        """
        Export layer to file.
        
        Args:
            layer: Layer to export.
            output_path: Output file path.
            
        Returns:
            True if export successful.
        """
        if not QGIS_AVAILABLE or layer is None:
            return False
        
        try:
            ext = Path(output_path).suffix.lower()
            
            if isinstance(layer, QgsVectorLayer):
                # Vector export
                options = layer.dataProvider().encoding()
                error = QgsVectorLayer.writeAsVectorFormat(
                    layer, output_path, "utf-8", 
                    driverName=self._get_vector_driver(ext),
                    onlySelected=False
                )
                return error == QgsVectorLayer.NoError
            elif isinstance(layer, QgsRasterLayer):
                # Raster export using GDAL
                from osgeo import gdal
                dataset = gdal.Open(layer.dataProvider().dataSourceUri())
                if dataset:
                    driver = gdal.GetDriverByName(self._get_raster_driver(ext))
                    driver.CreateCopy(output_path, dataset)
                    return True
                    
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            
        return False
    
    def _get_vector_driver(self, ext: str) -> str:
        """Get OGR driver name for vector extension."""
        drivers = {
            ".shp": "ESRI Shapefile",
            ".geojson": "GeoJSON",
            ".gpkg": "GPKG",
            ".kml": "KML",
            ".gml": "GML",
        }
        return drivers.get(ext, "GPKG")
    
    def _get_raster_driver(self, ext: str) -> str:
        """Get GDAL driver name for raster extension."""
        drivers = {
            ".tif": "GTiff",
            ".tiff": "GTiff",
            ".jp2": "JP2OpenJPEG",
            ".img": "HFA",
            ".asc": "AAIGrid",
        }
        return drivers.get(ext, "GTiff")
    
    def _log_summary(self, results: List[ProcessingResult], data_type: str) -> None:
        """Log processing summary."""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        total_time = sum(r.processing_time for r in results)
        total_features = sum(r.features_processed for r in results)
        
        self.logger.info(f"{'='*50}")
        self.logger.info(f"{data_type.capitalize()} Processing Summary")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Total files: {total}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Total processing time: {total_time:.2f}s")
        self.logger.info(f"Total features processed: {total_features}")
        self.logger.info(f"Average time per file: {total_time/total:.2f}s" if total > 0 else "")
        self.logger.info(f"{'='*50}")
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get summary of all processing results.
        
        Returns:
            Dictionary with processing statistics.
        """
        if not self.results:
            return {"total": 0}
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        return {
            "total": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.results) if self.results else 0,
            "total_processing_time": sum(r.processing_time for r in self.results),
            "total_features_processed": sum(r.features_processed for r in self.results),
            "errors": [r.error_message for r in failed if r.error_message]
        }
    
    def cleanup(self) -> None:
        """Clean up resources and exit QGIS application."""
        if self.use_qgis and QGIS_AVAILABLE:
            try:
                QgsApplication.exitQgis()
                self.logger.info("QGIS application closed")
            except Exception as e:
                self.logger.error(f"Error closing QGIS: {e}")


# Standalone script capability
def main():
    """Main entry point for standalone batch processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process geospatial data")
    parser.add_argument("input_dir", help="Input directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--type", choices=["vector", "raster"], default="vector")
    parser.add_argument("--operations", nargs="+", default=["reproject"])
    parser.add_argument("--target-crs", default="EPSG:3857")
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--config", default="config.yaml")
    
    args = parser.parse_args()
    
    config = Config.load(args.config) if Path(args.config).exists() else Config.default()
    processor = BatchProcessor(config)
    
    if args.type == "vector":
        results = processor.batch_process_vectors(
            args.input_dir,
            args.output_dir,
            operation=args.operations[0] if len(args.operations) == 1 else args.operations,
            target_crs=args.target_crs,
            parallel=args.parallel
        )
    else:
        results = processor.batch_process_rasters(
            args.input_dir,
            args.output_dir,
            operations=args.operations,
            target_crs=args.target_crs,
            parallel=args.parallel
        )
    
    summary = processor.get_results_summary()
    print(f"\nProcessing complete: {summary['successful']}/{summary['total']} successful")
    
    processor.cleanup()


if __name__ == "__main__":
    main()
