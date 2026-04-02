"""
Semantic Segmentation Module
============================

Provides semantic segmentation for land use/land cover classification.
Supports U-Net, DeepLabV3, and FCN architectures.

Example:
    >>> segmenter = SemanticSegmenter(model="unet")
    >>> segmenter.load_pretrained("land_cover")
    >>> mask = segmenter.segment(image_path)
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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import torchvision
    from torchvision.models.segmentation import (
        deeplabv3_resnet50,
        fcn_resnet50
    )
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
    F = type('F', (), {'softmax': lambda x, dim: x, 'interpolate': lambda x, **kw: x, 'one_hot': lambda x, n: x})()
    torch = _MockTorch()
    torchvision = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from src.utils.config import Config
from src.workflow.logger import PipelineLogger
from src.workflow.progress_tracker import ProgressTracker


@dataclass
class SegmentationResult:
    """Container for segmentation results."""
    
    mask: np.ndarray
    class_names: List[str] = field(default_factory=list)
    confidence: Optional[np.ndarray] = None
    image_size: Tuple[int, int] = (0, 0)
    processing_time: float = 0.0
    model_name: str = ""
    
    @property
    def class_counts(self) -> Dict[str, int]:
        """Count pixels per class."""
        counts = {}
        for i, name in enumerate(self.class_names):
            counts[name] = int(np.sum(self.mask == i))
        return counts
    
    @property
    def class_percentages(self) -> Dict[str, float]:
        """Calculate percentage of each class."""
        total = self.mask.size
        return {k: (v / total) * 100 for k, v in self.class_counts.items()}
    
    def get_class_mask(self, class_name: str) -> np.ndarray:
        """Get binary mask for a specific class."""
        if class_name in self.class_names:
            class_idx = self.class_names.index(class_name)
            return (self.mask == class_idx).astype(np.uint8)
        return np.zeros_like(self.mask)
    
    def to_geojson(
        self,
        transform: Optional[Any] = None,
        crs: str = "EPSG:4326"
    ) -> Dict[str, Any]:
        """
        Convert segmentation to GeoJSON polygons.
        
        Args:
            transform: Rasterio transform for georeferencing.
            crs: Coordinate reference system.
            
        Returns:
            GeoJSON FeatureCollection.
        """
        try:
            import rasterio
            from rasterio.features import shapes
            from shapely.geometry import mapping, shape
            import geopandas as gpd
            
            features = []
            
            for class_idx, class_name in enumerate(self.class_names):
                class_mask = (self.mask == class_idx).astype(np.uint8)
                
                # Vectorize
                results = shapes(
                    class_mask,
                    mask=class_mask,
                    transform=transform if transform else None
                )
                
                for geom, value in results:
                    features.append({
                        "type": "Feature",
                        "properties": {
                            "class": class_name,
                            "class_id": class_idx,
                            "area": geom.get('coordinates', []) and sum(
                                self._polygon_area(coords) for coords in geom.get('coordinates', [[]])
                            ) if geom.get('type') == 'Polygon' else 0
                        },
                        "geometry": geom
                    })
            
            return {
                "type": "FeatureCollection",
                "features": features,
                "properties": {
                    "image_size": self.image_size,
                    "class_names": self.class_names,
                    "class_percentages": self.class_percentages
                }
            }
            
        except ImportError:
            self.logger.warning("rasterio/geopandas not available for GeoJSON conversion")
            return {"type": "FeatureCollection", "features": []}
    
    def _polygon_area(self, coords: List) -> float:
        """Calculate polygon area using shoelace formula."""
        if len(coords) < 3:
            return 0.0
        
        area = 0.0
        n = len(coords)
        for i in range(n):
            j = (i + 1) % n
            area += coords[i][0] * coords[j][1]
            area -= coords[j][0] * coords[i][1]
        return abs(area) / 2.0
    
    def save_mask(self, path: str, format: str = "png") -> bool:
        """Save segmentation mask to file."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create RGB visualization
            if len(self.class_names) <= 256:
                # Use colormap
                from matplotlib.colors import ListedColormap
                import matplotlib.pyplot as plt
                
                cmap = plt.get_cmap('tab20', len(self.class_names))
                rgb_mask = (cmap(self.mask)[:, :, :3] * 255).astype(np.uint8)
                
                img = Image.fromarray(rgb_mask)
                img.save(path)
                return True
            else:
                # Save as grayscale
                img = Image.fromarray((self.mask * 255 / len(self.class_names)).astype(np.uint8))
                img.save(path)
                return True
                
        except Exception as e:
            logging.error(f"Error saving mask: {e}")
            return False
    
    def save_raster(
        self,
        path: str,
        reference_raster: str,
        dtype: int = 1  # Byte
    ) -> bool:
        """Save segmentation as georeferenced raster."""
        try:
            import rasterio
            from rasterio.transform import from_bounds
            
            with rasterio.open(reference_raster) as src:
                profile = src.profile.copy()
                profile.update({
                    'driver': 'GTiff',
                    'height': self.mask.shape[0],
                    'width': self.mask.shape[1],
                    'count': 1,
                    'dtype': rasterio.uint8 if dtype == 1 else rasterio.uint16,
                    'compress': 'lzw'
                })
                
                with rasterio.open(path, 'w', **profile) as dst:
                    dst.write(self.mask.astype(rasterio.uint8), 1)
            
            return True
            
        except ImportError:
            logging.error("rasterio not available")
            return False


class UNet(nn.Module):
    """U-Net architecture for semantic segmentation."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 6,
        features: List[int] = None,
        pretrained_backbone: bool = False
    ):
        super().__init__()
        
        if features is None:
            features = [64, 128, 256, 512]
        
        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        prev_channels = in_channels
        
        for features_count in features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, features_count, kernel_size=3, padding=1),
                    nn.BatchNorm2d(features_count),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(features_count, features_count, kernel_size=3, padding=1),
                    nn.BatchNorm2d(features_count),
                    nn.ReLU(inplace=True)
                )
            )
            prev_channels = features_count
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1] * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1] * 2, features[-1] * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsampling)
        self.decoder = nn.ModuleList()
        for features_count in reversed(features):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(features_count * 2, features_count, kernel_size=2, stride=2),
                    nn.Conv2d(features_count * 2, features_count, kernel_size=3, padding=1),
                    nn.BatchNorm2d(features_count),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(features_count, features_count, kernel_size=3, padding=1),
                    nn.BatchNorm2d(features_count),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        encoder_features = []
        for enc in self.encoder:
            x = enc(x)
            encoder_features.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for dec, enc_feat in zip(self.decoder, reversed(encoder_features)):
            x = dec(x)
            x = torch.cat([x, enc_feat], dim=1)
        
        return self.final_conv(x)


class SemanticSegmenter:
    """
    Semantic segmentation for land use/land cover classification.
    
    This class provides deep learning-based semantic segmentation
    optimized for geospatial imagery, including land cover mapping,
    urban area detection, and vegetation analysis.
    
    Attributes:
        model: Segmentation model architecture.
        backbone: Feature extractor backbone.
        config: Configuration object.
        logger: Logger instance.
        device: Computing device (cuda/cpu).
        
    Example:
        >>> segmenter = SemanticSegmenter(
        ...     model="unet",
        ...     backbone="resnet34",
        ...     classes=["background", "buildings", "roads", "vegetation", "water"]
        ... )
        >>> segmenter.load_pretrained("land_cover")
        >>> result = segmenter.segment("satellite_image.tif")
        >>> print(result.class_percentages)
    """
    
    # Predefined class mappings for common segmentation tasks
    CLASS_MAPPINGS = {
        "land_cover": [
            "water", "forest", "grassland", "cropland",
            "urban", "bare_soil", "wetland", "snow_ice"
        ],
        "urban": [
            "background", "buildings", "roads", "parking",
            "vegetation", "water", "bare_soil"
        ],
        "agriculture": [
            "background", "cropland", "forest", "grassland",
            "water", "bare_soil", "urban"
        ],
        "binary": ["background", "foreground"]
    }
    
    def __init__(
        self,
        model: str = "unet",
        backbone: str = "resnet34",
        classes: Optional[List[str]] = None,
        input_size: Tuple[int, int] = (512, 512),
        config: Optional[Config] = None,
        logger: Optional[PipelineLogger] = None
    ):
        """
        Initialize the semantic segmenter.
        
        Args:
            model: Model architecture (unet, deeplabv3, fcn).
            backbone: Feature backbone (resnet34, resnet50, efficientnet).
            classes: List of class names.
            input_size: Input image size.
            config: Configuration object.
            logger: Logger instance.
        """
        self.model_name = model
        self.backbone = backbone
        self.input_size = input_size
        self.config = config or Config.default()
        self.logger = logger or PipelineLogger.get_logger("SemanticSegmenter")
        
        # Load ML config
        ml_config = self.config.ml if hasattr(self.config, 'ml') else {}
        seg_config = ml_config.get('segmentation', {})
        
        # Set classes
        if classes:
            self.class_names = classes
        else:
            task = seg_config.get('task', 'land_cover')
            self.class_names = self.CLASS_MAPPINGS.get(task, self.CLASS_MAPPINGS["land_cover"])
        
        self.num_classes = len(self.class_names)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else torch.device('cpu')
        
        # Initialize model
        self.model = None
        self.is_loaded = False
        
        self.progress = ProgressTracker()
        
        self.logger.info(
            f"SemanticSegmenter initialized: {model}/{backbone} "
            f"with {self.num_classes} classes on {self.device}"
        )
    
    def _create_model(self) -> nn.Module:
        """Create segmentation model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        if self.model_name == "unet":
            model = UNet(
                in_channels=3,
                out_channels=self.num_classes,
                pretrained_backbone='efficientnet' in self.backbone
            )
        
        elif self.model_name == "deeplabv3":
            model = deeplabv3_resnet50(
                pretrained=True,
                progress=True
            )
            # Replace classifier
            model.classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)
            model.aux_classifier = None  # Remove auxiliary classifier
        
        elif self.model_name == "fcn":
            model = fcn_resnet50(
                pretrained=True,
                progress=True
            )
            # Replace classifier
            model.classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)
            model.aux_classifier = None
        
        else:
            self.logger.warning(f"Unknown model: {self.model_name}, using U-Net")
            model = UNet(in_channels=3, out_channels=self.num_classes)
        
        return model.to(self.device)
    
    def load_pretrained(self, task: str = "land_cover") -> bool:
        """
        Load pretrained weights for a specific task.
        
        Args:
            task: Segmentation task (land_cover, urban, agriculture).
            
        Returns:
            True if loading successful.
        """
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available")
            return False
        
        try:
            # Update classes for task
            self.class_names = self.CLASS_MAPPINGS.get(task, self.CLASS_MAPPINGS["land_cover"])
            self.num_classes = len(self.class_names)
            
            self.model = self._create_model()
            
            # Try to load weights
            weights_path = Path(self.config.paths.models_dir if hasattr(self.config, 'paths') else "./models")
            weights_file = weights_path / f"{task}_segmentation.pth"
            
            if weights_file.exists():
                checkpoint = torch.load(weights_file, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"Loaded pretrained weights from {weights_file}")
            else:
                self.logger.info(f"No pretrained weights found for {task}, using ImageNet backbone")
            
            self.model.eval()
            self.is_loaded = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading pretrained model: {e}")
            return False
    
    def load_weights(self, path: str) -> bool:
        """
        Load model weights from file.
        
        Args:
            path: Path to weights file.
            
        Returns:
            True if loading successful.
        """
        if not TORCH_AVAILABLE:
            return False
        
        try:
            if self.model is None:
                self.model = self._create_model()
            
            checkpoint = torch.load(path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            self.is_loaded = True
            
            if 'class_names' in checkpoint:
                self.class_names = checkpoint['class_names']
                self.num_classes = len(self.class_names)
            
            self.logger.info(f"Loaded weights from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading weights: {e}")
            return False
    
    def save_weights(self, path: str) -> bool:
        """
        Save model weights to file.
        
        Args:
            path: Output path for weights file.
            
        Returns:
            True if save successful.
        """
        if not TORCH_AVAILABLE or self.model is None:
            return False
        
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'class_names': self.class_names,
                'config': {
                    'model_name': self.model_name,
                    'backbone': self.backbone,
                    'num_classes': self.num_classes
                }
            }
            
            torch.save(checkpoint, path)
            self.logger.info(f"Saved weights to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving weights: {e}")
            return False
    
    def segment(
        self,
        image: Union[str, np.ndarray, Image.Image],
        return_confidence: bool = False
    ) -> SegmentationResult:
        """
        Perform semantic segmentation on an image.
        
        Args:
            image: Image path, numpy array, or PIL Image.
            return_confidence: Whether to return confidence map.
            
        Returns:
            SegmentationResult with class mask.
        """
        import time
        start_time = time.time()
        
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_pretrained() or load_weights() first.")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        # Load image
        image = self._load_image(image)
        original_size = image.size
        
        # Preprocess
        image_tensor = self._preprocess(image)
        
        # Segment
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor.unsqueeze(0))
            
            if isinstance(output, dict):
                output = output['out']
            
            # Get predictions
            predictions = F.interpolate(
                output,
                size=(original_size[1], original_size[0]),
                mode='bilinear',
                align_corners=False
            )
            mask = torch.argmax(predictions[0], dim=0).cpu().numpy()
            
            # Get confidence if requested
            confidence = None
            if return_confidence:
                probs = F.softmax(predictions[0], dim=0)
                confidence = torch.max(probs, dim=0).values.cpu().numpy()
        
        processing_time = time.time() - start_time
        
        result = SegmentationResult(
            mask=mask,
            class_names=self.class_names,
            confidence=confidence,
            image_size=original_size,
            processing_time=processing_time,
            model_name=self.model_name
        )
        
        self.logger.info(
            f"Segmentation complete in {processing_time:.3f}s - "
            f"Image: {original_size}, Classes: {self.num_classes}"
        )
        
        return result
    
    def segment_batch(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        batch_size: int = 8
    ) -> List[SegmentationResult]:
        """
        Segment multiple images.
        
        Args:
            images: List of images.
            batch_size: Batch size for processing.
            
        Returns:
            List of SegmentationResult objects.
        """
        self.progress.init(total=len(images), desc="Batch segmentation")
        
        results = []
        for image in images:
            result = self.segment(image)
            results.append(result)
            self.progress.update(1)
        
        return results
    
    def _load_image(
        self,
        image: Union[str, np.ndarray, Image.Image]
    ) -> Image.Image:
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
    
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for segmentation."""
        from torchvision.transforms import functional as F
        
        # Resize to input size
        image = image.resize(self.input_size, Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        return image_tensor.to(self.device)
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        epochs: int = 50,
        learning_rate: float = 0.001,
        batch_size: int = 8,
        loss_function: str = "cross_entropy"
    ) -> Dict[str, List[float]]:
        """
        Train the segmenter on custom dataset.
        
        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            epochs: Number of training epochs.
            learning_rate: Learning rate.
            batch_size: Batch size.
            loss_function: Loss function (cross_entropy, dice, focal).
            
        Returns:
            Dictionary with training history.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        if self.model is None:
            self.model = self._create_model()
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Loss function
        if loss_function == "dice":
            criterion = DiceLoss()
        elif loss_function == "focal":
            criterion = FocalLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
        
        self.progress.init(total=epochs, desc="Training segmenter")
        
        best_iou = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                images = batch['image'].to(self.device) if isinstance(batch, dict) else batch[0].to(self.device)
                masks = batch['mask'].to(self.device) if isinstance(batch, dict) else batch[1].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                
                loss = criterion(outputs, masks.long())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_iou = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(self.device) if isinstance(batch, dict) else batch[0].to(self.device)
                    masks = batch['mask'].to(self.device) if isinstance(batch, dict) else batch[1].to(self.device)
                    
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        outputs = outputs['out']
                    
                    loss = criterion(outputs, masks.long())
                    val_loss += loss.item()
                    
                    # Calculate IoU
                    preds = torch.argmax(outputs, dim=1)
                    iou = self._calculate_iou(preds, masks.long())
                    val_iou += iou
            
            val_loss /= len(val_loader)
            val_iou /= len(val_loader)
            
            scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_iou'].append(val_iou)
            
            self.logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}"
            )
            
            if val_iou > best_iou:
                best_iou = val_iou
            
            self.progress.update(1)
        
        self.is_loaded = True
        self.logger.info(f"Training complete. Best validation IoU: {best_iou:.4f}")
        
        return history
    
    def _calculate_iou(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate mean Intersection over Union."""
        ious = []
        
        for class_idx in range(self.num_classes):
            pred_class = (preds == class_idx)
            target_class = (targets == class_idx)
            
            intersection = (pred_class & target_class).sum().item()
            union = (pred_class | target_class).sum().item()
            
            if union > 0:
                ious.append(intersection / union)
        
        return np.mean(ious) if ious else 0.0


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        outputs = F.softmax(outputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, outputs.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (outputs * targets_one_hot).sum(dim=(2, 3))
        union = outputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for segmentation."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# Land cover segmentation convenience class
class LandCoverSegmenter(SemanticSegmenter):
    """Specialized segmenter for land cover classification."""
    
    def __init__(self, **kwargs):
        super().__init__(
            model="unet",
            classes=["water", "forest", "grassland", "cropland", "urban", "bare_soil"],
            **kwargs
        )
    
    def get_land_cover_stats(self, result: SegmentationResult) -> Dict[str, Any]:
        """Get detailed land cover statistics."""
        return {
            "class_percentages": result.class_percentages,
            "class_areas": {k: v for k, v in result.class_counts.items()},
            "dominant_class": max(result.class_percentages, key=result.class_percentages.get),
            "water_percentage": result.class_percentages.get("water", 0),
            "vegetation_percentage": (
                result.class_percentages.get("forest", 0) +
                result.class_percentages.get("grassland", 0) +
                result.class_percentages.get("cropland", 0)
            ),
            "urban_percentage": result.class_percentages.get("urban", 0)
        }


# Building segmentation convenience class
class BuildingSegmenter(SemanticSegmenter):
    """Specialized segmenter for building footprint extraction."""
    
    def __init__(self, **kwargs):
        super().__init__(
            model="unet",
            classes=["background", "buildings"],
            **kwargs
        )
    
    def extract_building_footprints(
        self,
        result: SegmentationResult,
        min_area: int = 10,
        simplify: bool = True
    ) -> List[Dict[str, Any]]:
        """Extract building footprints as polygons."""
        building_mask = result.get_class_mask("buildings")
        
        try:
            import cv2
            from shapely.geometry import Polygon, mapping
            
            # Find contours
            contours, _ = cv2.findContours(
                building_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            footprints = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area:
                    if simplify:
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        contour = cv2.approxPolyDP(contour, epsilon, True)
                    
                    polygon = Polygon(contour.reshape(-1, 2))
                    footprints.append({
                        "geometry": mapping(polygon),
                        "properties": {"area": area}
                    })
            
            return footprints
            
        except ImportError:
            self.logger.warning("cv2 not available for footprint extraction")
            return []
