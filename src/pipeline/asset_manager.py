"""
Asset Manager Module
====================

Provides digital asset management including metadata generation,
versioning, and export to multiple formats.

Example:
    >>> manager = AssetManager()
    >>> asset = manager.create_asset(data, metadata)
    >>> manager.export(asset, output_path, format="geojson")
"""

from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

import numpy as np

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

from src.utils.config import Config
from src.workflow.logger import PipelineLogger


class AssetType(Enum):
    """Types of digital assets."""
    VECTOR = "vector"
    RASTER = "raster"
    TABLE = "table"
    MODEL = "model"
    METADATA = "metadata"
    REPORT = "report"


class AssetStatus(Enum):
    """Asset processing status."""
    RAW = "raw"
    PROCESSING = "processing"
    VALIDATED = "validated"
    PUBLISHED = "published"
    ARCHIVED = "archived"


@dataclass
class AssetMetadata:
    """ISO 19115-inspired metadata for geospatial assets."""
    
    # Identification
    title: str
    description: str
    asset_type: str
    status: str
    
    # Temporal
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())
    temporal_extent_start: Optional[str] = None
    temporal_extent_end: Optional[str] = None
    
    # Spatial
    spatial_extent: Optional[Dict[str, float]] = None  # bbox
    coordinate_reference_system: str = "EPSG:4326"
    
    # Provenance
    source_data: List[str] = field(default_factory=list)
    processing_steps: List[Dict[str, Any]] = field(default_factory=list)
    algorithm_info: Optional[Dict[str, Any]] = None
    
    # Quality
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    validation_results: Optional[Dict[str, Any]] = None
    
    # Technical
    file_format: str = ""
    file_size_bytes: int = 0
    checksum: str = ""
    version: str = "1.0.0"
    
    # Contact
    creator: str = ""
    organization: str = ""
    contact_email: str = ""
    
    # Keywords and categories
    keywords: List[str] = field(default_factory=list)
    category: str = ""
    tags: List[str] = field(default_factory=list)
    
    # License and access
    license: str = "MIT"
    access_constraints: str = "none"
    
    def to_iso19115(self) -> Dict[str, Any]:
        """Convert to ISO 19115 format."""
        return {
            "MD_Metadata": {
                "fileIdentifier": self.checksum,
                "language": "eng",
                "characterSet": "utf8",
                "parentIdentifier": "",
                "hierarchyLevel": "dataset",
                "contact": [{
                    "role": "originator",
                    "organisationName": self.organization,
                    "individualName": self.creator,
                    "contactInfo": {"email": self.contact_email}
                }],
                "dateStamp": self.modified,
                "metadataStandardName": "ISO 19115",
                "metadataStandardVersion": "2014",
                "identificationInfo": [{
                    "MD_DataIdentification": {
                        "citation": {
                            "title": self.title,
                            "date": self.created
                        },
                        "abstract": self.description,
                        "status": self.status,
                        "topicCategory": self.category,
                        "extent": {
                            "geographicElement": {
                                "boundingBox": self.spatial_extent
                            } if self.spatial_extent else None,
                            "temporalElement": {
                                "beginPosition": self.temporal_extent_start,
                                "endPosition": self.temporal_extent_end
                            } if self.temporal_extent_start else None
                        },
                        "descriptiveKeywords": {
                            "keyword": self.keywords
                        }
                    }
                }],
                "dataQualityInfo": {
                    "report": self.validation_results
                } if self.validation_results else None
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class DigitalAsset:
    """Container for a digital asset with metadata."""
    
    id: str
    name: str
    asset_type: AssetType
    data: Any
    metadata: AssetMetadata
    path: Optional[str] = None
    status: AssetStatus = AssetStatus.RAW
    
    @property
    def checksum(self) -> str:
        """Calculate checksum of data."""
        if isinstance(self.data, np.ndarray):
            return hashlib.md5(self.data.tobytes()).hexdigest()
        elif isinstance(self.data, str):
            return hashlib.md5(self.data.encode()).hexdigest()
        elif self.path and Path(self.path).exists():
            return self._file_checksum(self.path)
        return ""
    
    def _file_checksum(self, path: str) -> str:
        """Calculate file checksum."""
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def update_metadata(self, **kwargs) -> None:
        """Update metadata fields."""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
        self.metadata.modified = datetime.now().isoformat()
    
    def add_processing_step(self, step: Dict[str, Any]) -> None:
        """Add a processing step to metadata."""
        self.metadata.processing_steps.append({
            "step": len(self.metadata.processing_steps) + 1,
            "timestamp": datetime.now().isoformat(),
            **step
        })
        self.metadata.modified = datetime.now().isoformat()


class AssetManager:
    """
    Digital asset management for geospatial data.
    
    This class provides comprehensive asset management capabilities
    including creation, metadata generation, versioning, and export
    to multiple formats.
    
    Attributes:
        config: Configuration object.
        logger: Logger instance.
        assets: Dictionary of managed assets.
        asset_dir: Base directory for assets.
        
    Example:
        >>> manager = AssetManager()
        >>> asset = manager.create_asset(
        ...     data=gdf,
        ...     name="buildings_2024",
        ...     asset_type=AssetType.VECTOR
        ... )
        >>> manager.export(asset, "output/buildings.geojson")
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        logger: Optional[PipelineLogger] = None,
        asset_dir: Optional[str] = None
    ):
        """
        Initialize the asset manager.
        
        Args:
            config: Configuration object.
            logger: Logger instance.
            asset_dir: Base directory for assets.
        """
        self.config = config or Config.default()
        self.logger = logger or PipelineLogger.get_logger("AssetManager")
        
        # Asset directory
        self.asset_dir = Path(asset_dir or self.config.paths.outputs_dir if hasattr(self.config, 'paths') else "./outputs")
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        
        # Asset registry
        self.assets: Dict[str, DigitalAsset] = {}
        self.asset_registry_path = self.asset_dir / "asset_registry.json"
        
        # Load existing registry
        self._load_registry()
        
        self.logger.info(f"AssetManager initialized at {self.asset_dir}")
    
    def _load_registry(self) -> None:
        """Load asset registry from file."""
        if self.asset_registry_path.exists():
            try:
                with open(self.asset_registry_path, 'r') as f:
                    registry = json.load(f)
                self.logger.info(f"Loaded {len(registry)} assets from registry")
            except Exception as e:
                self.logger.warning(f"Failed to load registry: {e}")
    
    def _save_registry(self) -> None:
        """Save asset registry to file."""
        try:
            registry = {
                asset_id: {
                    "name": asset.name,
                    "type": asset.asset_type.value,
                    "status": asset.status.value,
                    "path": str(asset.path),
                    "metadata": asset.metadata.to_dict()
                }
                for asset_id, asset in self.assets.items()
            }
            
            with open(self.asset_registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
    
    def create_asset(
        self,
        data: Any,
        name: str,
        asset_type: AssetType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DigitalAsset:
        """
        Create a new digital asset.
        
        Args:
            data: Asset data (GeoDataFrame, array, etc.).
            name: Asset name.
            asset_type: Type of asset.
            metadata: Optional metadata dictionary.
            
        Returns:
            Created DigitalAsset.
        """
        # Generate unique ID
        asset_id = self._generate_id(name)
        
        # Create metadata
        asset_metadata = self._create_metadata(name, asset_type, metadata)
        
        # Create asset
        asset = DigitalAsset(
            id=asset_id,
            name=name,
            asset_type=asset_type,
            data=data,
            metadata=asset_metadata
        )
        
        # Register asset
        self.assets[asset_id] = asset
        self._save_registry()
        
        self.logger.info(f"Created asset: {asset_id} ({name})")
        
        return asset
    
    def _generate_id(self, name: str) -> str:
        """Generate unique asset ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(f"{name}_{timestamp}".encode()).hexdigest()[:8]
        return f"{name.lower().replace(' ', '_')}_{name_hash}"
    
    def _create_metadata(
        self,
        name: str,
        asset_type: AssetType,
        custom_metadata: Optional[Dict[str, Any]]
    ) -> AssetMetadata:
        """Create metadata for asset."""
        metadata = AssetMetadata(
            title=name,
            description=f"Auto-generated asset: {name}",
            asset_type=asset_type.value,
            status=AssetStatus.RAW.value,
            **self._get_spatial_info(custom_metadata)
        )
        
        # Update with custom metadata
        if custom_metadata:
            for key, value in custom_metadata.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
        
        return metadata
    
    def _get_spatial_info(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract spatial information from data."""
        info = {}
        
        if metadata and "spatial_extent" in metadata:
            info["spatial_extent"] = metadata["spatial_extent"]
        
        if metadata and "coordinate_reference_system" in metadata:
            info["coordinate_reference_system"] = metadata["coordinate_reference_system"]
        
        return info
    
    def load_asset(self, path: str, asset_type: Optional[AssetType] = None) -> DigitalAsset:
        """
        Load an asset from file.
        
        Args:
            path: Path to asset file.
            asset_type: Type of asset (auto-detected if None).
            
        Returns:
            Loaded DigitalAsset.
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Asset not found: {path}")
        
        # Auto-detect type
        if asset_type is None:
            asset_type = self._detect_asset_type(path)
        
        # Load data based on type
        data = self._load_data(path, asset_type)
        
        # Create asset
        name = path.stem
        asset = self.create_asset(data, name, asset_type)
        asset.path = str(path)
        asset.status = AssetStatus.VALIDATED
        
        # Update metadata
        asset.metadata.file_format = path.suffix
        asset.metadata.file_size_bytes = path.stat().st_size
        asset.metadata.checksum = asset.checksum
        
        self.logger.info(f"Loaded asset from {path}")
        
        return asset
    
    def _detect_asset_type(self, path: Path) -> AssetType:
        """Detect asset type from file extension."""
        ext = path.suffix.lower()
        
        vector_exts = [".shp", ".geojson", ".gpkg", ".kml", ".gml"]
        raster_exts = [".tif", ".tiff", ".jp2", ".img", ".asc", ".vrt"]
        table_exts = [".csv", ".parquet", ".feather"]
        model_exts = [".pth", ".pt", ".h5", ".onnx", ".pkl"]
        
        if ext in vector_exts:
            return AssetType.VECTOR
        elif ext in raster_exts:
            return AssetType.RASTER
        elif ext in table_exts:
            return AssetType.TABLE
        elif ext in model_exts:
            return AssetType.MODEL
        else:
            return AssetType.VECTOR  # Default
    
    def _load_data(self, path: Path, asset_type: AssetType) -> Any:
        """Load data from file based on type."""
        if asset_type == AssetType.VECTOR:
            if GEOPANDAS_AVAILABLE:
                return gpd.read_file(path)
            else:
                raise ImportError("geopandas required for vector loading")
        
        elif asset_type == AssetType.RASTER:
            if RASTERIO_AVAILABLE:
                with rasterio.open(path) as src:
                    return src.read()
            else:
                raise ImportError("rasterio required for raster loading")
        
        elif asset_type == AssetType.TABLE:
            import pandas as pd
            if path.suffix == ".csv":
                return pd.read_csv(path)
            elif path.suffix == ".parquet":
                return pd.read_parquet(path)
            else:
                return pd.read_feather(path)
        
        else:
            # For other types, just store path
            return str(path)
    
    def export(
        self,
        asset: DigitalAsset,
        output_path: str,
        format: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Export asset to file.
        
        Args:
            asset: Asset to export.
            output_path: Output file path.
            format: Output format (auto-detected if None).
            **kwargs: Format-specific options.
            
        Returns:
            True if export successful.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format is None:
            format = output_path.suffix.lower().lstrip('.')
        
        try:
            if asset.asset_type == AssetType.VECTOR:
                return self._export_vector(asset, output_path, format, **kwargs)
            elif asset.asset_type == AssetType.RASTER:
                return self._export_raster(asset, output_path, format, **kwargs)
            elif asset.asset_type == AssetType.TABLE:
                return self._export_table(asset, output_path, format, **kwargs)
            elif asset.asset_type == AssetType.METADATA:
                return self._export_metadata(asset, output_path, **kwargs)
            else:
                self.logger.error(f"Cannot export asset type: {asset.asset_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False
    
    def _export_vector(
        self,
        asset: DigitalAsset,
        output_path: Path,
        format: str,
        **kwargs
    ) -> bool:
        """Export vector asset."""
        if not GEOPANDAS_AVAILABLE:
            self.logger.error("geopandas not available")
            return False
        
        data = asset.data
        
        # Handle different data types
        if isinstance(data, gpd.GeoDataFrame):
            gdf = data
        elif isinstance(data, np.ndarray):
            # Convert array to points
            from shapely.geometry import Point
            gdf = gpd.GeoDataFrame(
                {"geometry": [Point(x, y) for x, y in data[:, :2]]}
            )
        else:
            self.logger.error(f"Unsupported vector data type: {type(data)}")
            return False
        
        # Map format to driver
        drivers = {
            "geojson": "GeoJSON",
            "json": "GeoJSON",
            "shp": "ESRI Shapefile",
            "shapefile": "ESRI Shapefile",
            "gpkg": "GPKG",
            "geopackage": "GPKG",
            "kml": "KML",
            "gml": "GML"
        }
        
        driver = drivers.get(format, "GPKG")
        
        # Adjust path for shapefile
        if driver == "ESRI Shapefile":
            output_path = output_path.with_suffix(".shp")
        
        gdf.to_file(output_path, driver=driver, **kwargs)
        
        # Export metadata alongside
        metadata_path = output_path.with_suffix(".json")
        self._export_metadata(asset, metadata_path)
        
        asset.path = str(output_path)
        self.logger.info(f"Exported vector to {output_path}")
        
        return True
    
    def _export_raster(
        self,
        asset: DigitalAsset,
        output_path: Path,
        format: str,
        **kwargs
    ) -> bool:
        """Export raster asset."""
        if not RASTERIO_AVAILABLE:
            self.logger.error("rasterio not available")
            return False
        
        data = asset.data
        
        if not isinstance(data, np.ndarray):
            self.logger.error("Raster data must be numpy array")
            return False
        
        # Default profile
        profile = {
            'driver': 'GTiff',
            'height': data.shape[-2] if len(data.shape) == 3 else data.shape[0],
            'width': data.shape[-1] if len(data.shape) == 3 else data.shape[1],
            'count': data.shape[0] if len(data.shape) == 3 else 1,
            'dtype': data.dtype,
            'crs': asset.metadata.coordinate_reference_system,
            'compress': 'lzw'
        }
        
        profile.update(kwargs)
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)
        
        asset.path = str(output_path)
        self.logger.info(f"Exported raster to {output_path}")
        
        return True
    
    def _export_table(
        self,
        asset: DigitalAsset,
        output_path: Path,
        format: str,
        **kwargs
    ) -> bool:
        """Export table asset."""
        import pandas as pd
        
        data = asset.data
        
        if not isinstance(data, pd.DataFrame):
            self.logger.error("Table data must be DataFrame")
            return False
        
        if format == "csv":
            data.to_csv(output_path, index=False, **kwargs)
        elif format == "parquet":
            data.to_parquet(output_path, **kwargs)
        elif format == "excel" or format == "xlsx":
            data.to_excel(output_path, index=False, **kwargs)
        else:
            data.to_csv(output_path, index=False)
        
        asset.path = str(output_path)
        self.logger.info(f"Exported table to {output_path}")
        
        return True
    
    def _export_metadata(
        self,
        asset: DigitalAsset,
        output_path: Path,
        format: str = "json"
    ) -> bool:
        """Export asset metadata."""
        try:
            output_path = Path(output_path)
            
            if format == "json":
                with open(output_path, 'w') as f:
                    json.dump(asset.metadata.to_dict(), f, indent=2)
            elif format == "xml" or output_path.suffix == ".xml":
                # ISO 19115 XML format
                self._export_iso19115_xml(asset, output_path)
            else:
                with open(output_path, 'w') as f:
                    json.dump(asset.metadata.to_dict(), f, indent=2)
            
            self.logger.info(f"Exported metadata to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Metadata export failed: {e}")
            return False
    
    def _export_iso19115_xml(self, asset: DigitalAsset, output_path: Path) -> None:
        """Export metadata as ISO 19115 XML."""
        iso_dict = asset.metadata.to_iso19115()
        
        # Simple XML generation (in production, use proper XML library)
        xml_content = self._dict_to_xml(iso_dict)
        
        with open(output_path, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(xml_content)
    
    def _dict_to_xml(self, d: Dict, parent: str = "") -> str:
        """Convert dictionary to XML string."""
        xml = ""
        for key, value in d.items():
            tag = key.replace("_", "")
            if isinstance(value, dict):
                xml += f"<{tag}>\n{self._dict_to_xml(value, tag)}\n</{tag}>"
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        xml += f"<{tag}>\n{self._dict_to_xml(item, tag)}\n</{tag}>"
                    else:
                        xml += f"<{tag}>{item}</{tag}>"
            elif value is not None:
                xml += f"<{tag}>{value}</{tag}>"
        return xml
    
    def get_asset(self, asset_id: str) -> Optional[DigitalAsset]:
        """Get asset by ID."""
        return self.assets.get(asset_id)
    
    def list_assets(
        self,
        asset_type: Optional[AssetType] = None,
        status: Optional[AssetStatus] = None
    ) -> List[DigitalAsset]:
        """List assets with optional filters."""
        assets = list(self.assets.values())
        
        if asset_type:
            assets = [a for a in assets if a.asset_type == asset_type]
        
        if status:
            assets = [a for a in assets if a.status == status]
        
        return assets
    
    def update_asset_status(
        self,
        asset_id: str,
        status: AssetStatus
    ) -> bool:
        """Update asset status."""
        if asset_id in self.assets:
            self.assets[asset_id].status = status
            self.assets[asset_id].metadata.status = status.value
            self._save_registry()
            return True
        return False
    
    def delete_asset(self, asset_id: str, delete_files: bool = False) -> bool:
        """
        Delete an asset.
        
        Args:
            asset_id: ID of asset to delete.
            delete_files: Whether to delete associated files.
            
        Returns:
            True if deletion successful.
        """
        if asset_id not in self.assets:
            return False
        
        asset = self.assets[asset_id]
        
        if delete_files and asset.path:
            try:
                Path(asset.path).unlink()
            except Exception as e:
                self.logger.warning(f"Failed to delete file: {e}")
        
        del self.assets[asset_id]
        self._save_registry()
        
        self.logger.info(f"Deleted asset: {asset_id}")
        return True
    
    def generate_catalog(self, output_path: str) -> bool:
        """
        Generate asset catalog.
        
        Args:
            output_path: Path for catalog file.
            
        Returns:
            True if generation successful.
        """
        try:
            catalog = {
                "generated": datetime.now().isoformat(),
                "total_assets": len(self.assets),
                "assets": [
                    {
                        "id": asset.id,
                        "name": asset.name,
                        "type": asset.asset_type.value,
                        "status": asset.status.value,
                        "path": asset.path,
                        "metadata": asset.metadata.to_dict()
                    }
                    for asset in self.assets.values()
                ]
            }
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(catalog, f, indent=2)
            
            self.logger.info(f"Generated catalog with {len(self.assets)} assets")
            return True
            
        except Exception as e:
            self.logger.error(f"Catalog generation failed: {e}")
            return False


# Convenience functions
def create_vector_asset(
    data: Any,
    name: str,
    output_dir: Optional[str] = None
) -> DigitalAsset:
    """Create a vector asset."""
    manager = AssetManager(asset_dir=output_dir)
    return manager.create_asset(data, name, AssetType.VECTOR)


def create_raster_asset(
    data: np.ndarray,
    name: str,
    output_dir: Optional[str] = None
) -> DigitalAsset:
    """Create a raster asset."""
    manager = AssetManager(asset_dir=output_dir)
    return manager.create_asset(data, name, AssetType.RASTER)
