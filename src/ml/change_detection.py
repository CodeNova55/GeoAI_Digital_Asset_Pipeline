"""
Change Detection Module
=======================

Provides change detection between multi-temporal satellite imagery.
Supports various methods including Siamese networks, image differencing,
and classification-based approaches.

Example:
    >>> detector = ChangeDetector(method="siamese")
    >>> changes = detector.detect_change(before_image, after_image)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import logging

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Define stub types
    class _MockNN:
        class Module: pass
    class _MockTorch:
        class device:
            def __init__(self, *args): pass
            def __str__(self): return 'cpu'
        @staticmethod
        def cuda(): return False
    nn = _MockNN()
    F = type('F', (), {'sigmoid': lambda x: x, 'max_pool2d': lambda x, k: x, 'interpolate': lambda x, **kw: x})()
    torch = _MockTorch()

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from src.utils.config import Config
from src.workflow.logger import PipelineLogger


@dataclass
class ChangeDetectionResult:
    """Container for change detection results."""
    
    change_mask: np.ndarray
    change_map: Optional[np.ndarray] = None  # Probabilistic change map
    change_pixels: int = 0
    total_pixels: int = 0
    change_percentage: float = 0.0
    processing_time: float = 0.0
    method: str = ""
    
    # Change statistics by region
    change_regions: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def no_change_percentage(self) -> float:
        """Percentage of unchanged area."""
        return 100.0 - self.change_percentage
    
    def get_change_locations(self) -> List[Tuple[int, int]]:
        """Get coordinates of changed pixels."""
        return list(zip(*np.where(self.change_mask > 0)))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "change_pixels": self.change_pixels,
            "total_pixels": self.total_pixels,
            "change_percentage": self.change_percentage,
            "no_change_percentage": self.no_change_percentage,
            "processing_time": self.processing_time,
            "method": self.method
        }


class SiameseUNet(nn.Module):
    """Siamese U-Net for change detection."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        
        # Encoder (shared weights)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x1, x2):
        # Encode both inputs with shared weights
        enc1 = self.encoder(x1)
        enc2 = self.encoder(x2)
        
        # Pool
        pool1 = F.max_pool2d(enc1, 2)
        pool2 = F.max_pool2d(enc2, 2)
        
        # Bottleneck - concatenate encoded features
        bottleneck_input = torch.cat([pool1, pool2], dim=1)
        bottleneck = self.bottleneck(bottleneck_input)
        
        # Decode
        upsampled = F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Concatenate with skip connections
        skip_input = torch.cat([enc1, enc2, upsampled], dim=1)
        decoded = self.decoder(skip_input)
        
        return torch.sigmoid(self.final(decoded))


class ChangeDetector:
    """
    Change detection for multi-temporal satellite imagery.
    
    This class provides methods for detecting changes between
    two or more temporal images, useful for monitoring urban
    growth, deforestation, disaster assessment, etc.
    
    Attributes:
        method: Detection method (siamese, difference, classification).
        config: Configuration object.
        logger: Logger instance.
        
    Example:
        >>> detector = ChangeDetector(method="siamese_network")
        >>> result = detector.detect_change(
        ...     "image_2020.tif",
        ...     "image_2024.tif",
        ...     threshold=0.5
        ... )
        >>> print(f"Change detected: {result.change_percentage:.1f}%")
    """
    
    def __init__(
        self,
        method: str = "siamese_network",
        threshold: float = 0.5,
        min_change_area: int = 50,
        config: Optional[Config] = None,
        logger: Optional[PipelineLogger] = None
    ):
        """
        Initialize the change detector.
        
        Args:
            method: Detection method.
            threshold: Change detection threshold.
            min_change_area: Minimum change area in pixels.
            config: Configuration object.
            logger: Logger instance.
        """
        self.method = method
        self.threshold = threshold
        self.min_change_area = min_change_area
        
        self.config = config or Config.default()
        self.logger = logger or PipelineLogger.get_logger("ChangeDetector")
        
        # Load config
        ml_config = self.config.ml if hasattr(self.config, 'ml') else {}
        cd_config = ml_config.get('change_detection', {})
        
        self.threshold = cd_config.get('threshold', threshold)
        self.min_change_area = cd_config.get('min_change_area', min_change_area)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else torch.device('cpu')
        
        # Model
        self.model = None
        self.is_loaded = False
        
        self.logger.info(f"ChangeDetector initialized: {method} on {self.device}")
    
    def _create_model(self) -> nn.Module:
        """Create change detection model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        return SiameseUNet(in_channels=3, out_channels=1).to(self.device)
    
    def load_pretrained(self, weights_path: Optional[str] = None) -> bool:
        """Load pretrained model."""
        if not TORCH_AVAILABLE:
            return False
        
        try:
            self.model = self._create_model()
            
            if weights_path and Path(weights_path).exists():
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"Loaded pretrained weights from {weights_path}")
            
            self.model.eval()
            self.is_loaded = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading pretrained model: {e}")
            return False
    
    def detect_change(
        self,
        image_before: Union[str, np.ndarray, Image.Image],
        image_after: Union[str, np.ndarray, Image.Image],
        threshold: Optional[float] = None
    ) -> ChangeDetectionResult:
        """
        Detect changes between two images.
        
        Args:
            image_before: Earlier image.
            image_after: Later image.
            threshold: Override default threshold.
            
        Returns:
            ChangeDetectionResult with change mask and statistics.
        """
        import time
        start_time = time.time()
        
        threshold = threshold or self.threshold
        
        # Load images
        img1 = self._load_image(image_before)
        img2 = self._load_image(image_after)
        
        # Ensure same size
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
        
        original_size = img1.size
        
        if self.method == "siamese_network" and TORCH_AVAILABLE and self.is_loaded:
            change_map = self._detect_siamese(img1, img2)
        elif self.method == "difference":
            change_map = self._detect_difference(img1, img2)
        elif self.method == "classification":
            change_map = self._detect_classification(img1, img2)
        else:
            # Default to image differencing
            change_map = self._detect_difference(img1, img2)
        
        # Apply threshold
        change_mask = (change_map >= threshold).astype(np.uint8)
        
        # Remove small regions
        change_mask = self._remove_small_regions(change_mask, self.min_change_area)
        
        # Calculate statistics
        change_pixels = int(np.sum(change_mask))
        total_pixels = change_mask.size
        
        processing_time = time.time() - start_time
        
        result = ChangeDetectionResult(
            change_mask=change_mask,
            change_map=change_map,
            change_pixels=change_pixels,
            total_pixels=total_pixels,
            change_percentage=(change_pixels / total_pixels) * 100,
            processing_time=processing_time,
            method=self.method
        )
        
        self.logger.info(
            f"Change detection complete: {result.change_percentage:.2f}% changed "
            f"in {processing_time:.3f}s"
        )
        
        return result
    
    def _load_image(self, image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """Load image from various sources."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL not available")
        
        if isinstance(image, str):
            return Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _detect_siamese(
        self,
        img1: Image.Image,
        img2: Image.Image
    ) -> np.ndarray:
        """Detect changes using Siamese network."""
        if not TORCH_AVAILABLE or self.model is None:
            return self._detect_difference(img1, img2)
        
        from torchvision.transforms import functional as F
        
        # Preprocess
        t1 = F.to_tensor(img1).unsqueeze(0).to(self.device)
        t2 = F.to_tensor(img2).unsqueeze(0).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            change_map = self.model(t1, t2)
            change_map = F.interpolate(
                change_map,
                size=(img1.size[1], img1.size[0]),
                mode='bilinear',
                align_corners=False
            )
        
        return change_map[0, 0].cpu().numpy()
    
    def _detect_difference(
        self,
        img1: Image.Image,
        img2: Image.Image
    ) -> np.ndarray:
        """Detect changes using image differencing."""
        arr1 = np.array(img1).astype(float)
        arr2 = np.array(img2).astype(float)
        
        # Calculate difference
        diff = np.abs(arr1 - arr2)
        
        # Convert to grayscale difference
        if len(diff.shape) == 3:
            diff = np.mean(diff, axis=2)
        
        # Normalize to 0-1
        diff = diff / 255.0
        
        return diff
    
    def _detect_classification(
        self,
        img1: Image.Image,
        img2: Image.Image
    ) -> np.ndarray:
        """Detect changes using classification approach."""
        # Stack images and use simple thresholding
        arr1 = np.array(img1).astype(float) / 255.0
        arr2 = np.array(img2).astype(float) / 255.0
        
        # Calculate normalized difference
        diff = np.abs(arr1 - arr2)
        
        if len(diff.shape) == 3:
            diff = np.mean(diff, axis=2)
        
        return diff
    
    def _remove_small_regions(
        self,
        mask: np.ndarray,
        min_area: int
    ) -> np.ndarray:
        """Remove small connected components from mask."""
        try:
            from scipy import ndimage
            
            labeled, num_features = ndimage.label(mask)
            
            for i in range(1, num_features + 1):
                region_area = np.sum(labeled == i)
                if region_area < min_area:
                    mask[labeled == i] = 0
            
            return mask
            
        except ImportError:
            # Simple morphological operations if scipy not available
            return mask
    
    def detect_change_batch(
        self,
        image_pairs: List[Tuple[Union[str, np.ndarray], Union[str, np.ndarray]]],
        output_dir: Optional[str] = None
    ) -> List[ChangeDetectionResult]:
        """
        Detect changes in multiple image pairs.
        
        Args:
            image_pairs: List of (before, after) image pairs.
            output_dir: Optional directory to save results.
            
        Returns:
            List of ChangeDetectionResult objects.
        """
        results = []
        
        for i, (img_before, img_after) in enumerate(image_pairs):
            result = self.detect_change(img_before, img_after)
            results.append(result)
            
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
                # Save change mask
                mask_path = Path(output_dir) / f"change_mask_{i:04d}.png"
                Image.fromarray((result.change_mask * 255).astype(np.uint8)).save(mask_path)
                
                # Save metadata
                meta_path = Path(output_dir) / f"change_meta_{i:04d}.json"
                import json
                with open(meta_path, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
        
        return results
    
    def get_change_statistics(
        self,
        result: ChangeDetectionResult,
        reference_image: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed change statistics.
        
        Args:
            result: ChangeDetectionResult to analyze.
            reference_image: Optional reference for geospatial stats.
            
        Returns:
            Dictionary with detailed statistics.
        """
        stats = {
            "total_change_pixels": result.change_pixels,
            "total_pixels": result.total_pixels,
            "change_percentage": result.change_percentage,
            "change_locations_count": len(result.get_change_locations()),
            "method": result.method
        }
        
        if reference_image:
            try:
                import rasterio
                with rasterio.open(reference_image) as src:
                    pixel_area = src.transform[0] * abs(src.transform[4])
                    stats["change_area_sqm"] = result.change_pixels * pixel_area
                    stats["total_area_sqm"] = result.total_pixels * pixel_area
            except (ImportError, Exception):
                pass
        
        return stats


# Convenience functions
def quick_change_detection(
    image_before: Union[str, np.ndarray],
    image_after: Union[str, np.ndarray],
    method: str = "difference"
) -> ChangeDetectionResult:
    """Quick change detection with default settings."""
    detector = ChangeDetector(method=method)
    return detector.detect_change(image_before, image_after)


def compare_time_series(
    images: List[Union[str, np.ndarray]],
    dates: Optional[List[str]] = None
) -> List[ChangeDetectionResult]:
    """Compare a time series of images."""
    detector = ChangeDetector(method="difference")
    results = []
    
    for i in range(len(images) - 1):
        result = detector.detect_change(images[i], images[i + 1])
        results.append(result)
    
    return results
