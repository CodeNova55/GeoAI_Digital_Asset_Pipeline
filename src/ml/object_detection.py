"""
Object Detection Module
=======================

Provides object detection for satellite and aerial imagery.
Supports building detection, vehicle detection, and other geospatial objects.

Compatible with GeoAI plugins and supports multiple architectures:
Faster R-CNN, YOLO, SSD.

Example:
    >>> detector = ObjectDetector(model="faster_rcnn")
    >>> detector.load_pretrained("buildings")
    >>> results = detector.detect(image_path)
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
    import torchvision
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        retinanet_resnet50_fpn,
        ssd300_vgg16
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.transforms import functional as F
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
    torch = _MockTorch()
    torchvision = None
    F = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from src.utils.config import Config
from src.workflow.logger import PipelineLogger
from src.workflow.progress_tracker import ProgressTracker


@dataclass
class Detection:
    """Container for a single detection."""
    
    bbox: Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max
    label: str
    confidence: float
    class_id: int = 0
    area: float = 0.0
    center: Tuple[float, float] = (0.0, 0.0)
    
    def __post_init__(self):
        self.area = (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
        self.center = (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bbox": self.bbox,
            "label": self.label,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "area": self.area,
            "center": self.center
        }
    
    def to_geojson_feature(self) -> Dict[str, Any]:
        """Convert to GeoJSON feature."""
        return {
            "type": "Feature",
            "properties": {
                "label": self.label,
                "confidence": self.confidence,
                "class_id": self.class_id,
                "area": self.area
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [self.bbox[0], self.bbox[1]],
                    [self.bbox[2], self.bbox[1]],
                    [self.bbox[2], self.bbox[3]],
                    [self.bbox[0], self.bbox[3]],
                    [self.bbox[0], self.bbox[1]]
                ]]
            }
        }


@dataclass
class DetectionResult:
    """Container for detection results."""
    
    detections: List[Detection] = field(default_factory=list)
    image_size: Tuple[int, int] = (0, 0)
    processing_time: float = 0.0
    model_name: str = ""
    
    @property
    def count(self) -> int:
        """Number of detections."""
        return len(self.detections)
    
    @property
    def by_class(self) -> Dict[str, int]:
        """Count detections by class."""
        counts = {}
        for det in self.detections:
            counts[det.label] = counts.get(det.label, 0) + 1
        return counts
    
    def to_geojson(self) -> Dict[str, Any]:
        """Convert to GeoJSON FeatureCollection."""
        return {
            "type": "FeatureCollection",
            "features": [d.to_geojson_feature() for d in self.detections],
            "properties": {
                "image_size": self.image_size,
                "processing_time": self.processing_time,
                "model_name": self.model_name,
                "total_detections": self.count,
                "by_class": self.by_class
            }
        }
    
    def save_geojson(self, path: str) -> bool:
        """Save results as GeoJSON."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(self.to_geojson(), f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error saving GeoJSON: {e}")
            return False


class ObjectDetector:
    """
    Object detection for satellite and aerial imagery.
    
    This class provides deep learning-based object detection optimized
    for geospatial imagery, including building footprint extraction,
    vehicle detection, and other geospatial objects.
    
    Attributes:
        model: Detection model architecture name.
        backbone: Feature extractor backbone.
        config: Configuration object.
        logger: Logger instance.
        device: Computing device (cuda/cpu).
        
    Example:
        >>> detector = ObjectDetector(
        ...     model="faster_rcnn",
        ...     backbone="resnet50",
        ...     confidence_threshold=0.7
        ... )
        >>> detector.load_pretrained("buildings")
        >>> results = detector.detect("satellite_image.tif")
        >>> print(f"Found {results.count} buildings")
    """
    
    # Predefined class mappings for common geospatial detection tasks
    CLASS_MAPPINGS = {
        "buildings": {0: "building"},
        "vehicles": {0: "car", 1: "truck", 2: "bus"},
        "trees": {0: "tree"},
        "roads": {0: "road_segment"},
        "solar_panels": {0: "solar_panel"},
        "swimming_pools": {0: "pool"},
        "aircraft": {0: "airplane", 1: "helicopter"},
        "ships": {0: "ship", 1: "boat"},
        "general": {
            0: "building", 1: "vehicle", 2: "tree",
            3: "road", 4: "water", 5: "vegetation"
        }
    }
    
    def __init__(
        self,
        model: str = "faster_rcnn",
        backbone: str = "resnet50",
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (512, 512),
        config: Optional[Config] = None,
        logger: Optional[PipelineLogger] = None,
        num_classes: int = 2
    ):
        """
        Initialize the object detector.
        
        Args:
            model: Model architecture (faster_rcnn, retinanet, ssd).
            backbone: Feature backbone (resnet50, resnet101, vgg16).
            confidence_threshold: Minimum confidence for detections.
            nms_threshold: Non-maximum suppression threshold.
            input_size: Input image size for detection.
            config: Configuration object.
            logger: Logger instance.
            num_classes: Number of object classes.
        """
        self.model_name = model
        self.backbone = backbone
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.num_classes = num_classes
        
        self.config = config or Config.default()
        self.logger = logger or PipelineLogger.get_logger("ObjectDetector")
        
        # Load ML config
        ml_config = self.config.ml if hasattr(self.config, 'ml') else {}
        od_config = ml_config.get('object_detection', {})
        
        # Update from config
        self.confidence_threshold = od_config.get('confidence_threshold', confidence_threshold)
        self.nms_threshold = od_config.get('nms_threshold', nms_threshold)
        self.input_size = tuple(od_config.get('input_size', input_size))
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else torch.device('cpu')
        
        # Initialize model
        self.model = None
        self.class_mapping = self.CLASS_MAPPINGS.get("general")
        self.is_loaded = False
        
        self.progress = ProgressTracker()
        
        self.logger.info(f"ObjectDetector initialized: {model}/{backbone} on {self.device}")
    
    def _create_model(self) -> nn.Module:
        """Create detection model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        if self.model_name == "faster_rcnn":
            model = fasterrcnn_resnet50_fpn(
                pretrained=True,
                pretrained_backbone=True
            )
            
            # Replace classifier for custom classes
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            
        elif self.model_name == "retinanet":
            model = retinanet_resnet50_fpn(
                pretrained=True,
                pretrained_backbone=True
            )
            # Retinanet has different head structure
            
        elif self.model_name == "ssd":
            model = ssd300_vgg16(
                pretrained=True,
                pretrained_backbone=True
            )
            
        else:
            self.logger.warning(f"Unknown model: {self.model_name}, using Faster R-CNN")
            model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        return model.to(self.device)
    
    def load_pretrained(self, task: str = "buildings") -> bool:
        """
        Load pretrained weights for a specific task.
        
        Args:
            task: Detection task (buildings, vehicles, trees, etc.).
            
        Returns:
            True if loading successful.
        """
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available")
            return False
        
        try:
            self.class_mapping = self.CLASS_MAPPINGS.get(task, self.CLASS_MAPPINGS["general"])
            self.num_classes = len(self.class_mapping) + 1  # +1 for background
            
            self.model = self._create_model()
            
            # Try to load weights from models directory
            weights_path = Path(self.config.paths.models_dir if hasattr(self.config, 'paths') else "./models")
            weights_file = weights_path / f"{task}_weights.pth"
            
            if weights_file.exists():
                checkpoint = torch.load(weights_file, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"Loaded pretrained weights from {weights_file}")
            else:
                self.logger.info(f"No pretrained weights found for {task}, using COCO weights")
            
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
            
            # Load class mapping if available
            if 'class_mapping' in checkpoint:
                self.class_mapping = checkpoint['class_mapping']
            
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
                'class_mapping': self.class_mapping,
                'config': {
                    'model_name': self.model_name,
                    'backbone': self.backbone,
                    'num_classes': self.num_classes,
                    'confidence_threshold': self.confidence_threshold
                }
            }
            
            torch.save(checkpoint, path)
            self.logger.info(f"Saved weights to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving weights: {e}")
            return False
    
    def detect(
        self,
        image: Union[str, np.ndarray, Image.Image],
        return_all: bool = False
    ) -> DetectionResult:
        """
        Detect objects in an image.
        
        Args:
            image: Image path, numpy array, or PIL Image.
            return_all: Return all detections regardless of confidence.
            
        Returns:
            DetectionResult with detected objects.
        """
        import time
        start_time = time.time()
        
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_pretrained() or load_weights() first.")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for detection")
        
        # Load image
        image = self._load_image(image)
        original_size = image.size
        
        # Preprocess
        image_tensor = self._preprocess(image)
        
        # Detect
        self.model.eval()
        with torch.no_grad():
            predictions = self.model([image_tensor])
        
        # Post-process
        detections = self._process_predictions(
            predictions[0],
            original_size,
            return_all
        )
        
        processing_time = time.time() - start_time
        
        result = DetectionResult(
            detections=detections,
            image_size=original_size,
            processing_time=processing_time,
            model_name=self.model_name
        )
        
        self.logger.info(f"Detected {len(detections)} objects in {processing_time:.3f}s")
        
        return result
    
    def detect_batch(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        batch_size: int = 8
    ) -> List[DetectionResult]:
        """
        Detect objects in multiple images.
        
        Args:
            images: List of images.
            batch_size: Batch size for processing.
            
        Returns:
            List of DetectionResult objects.
        """
        self.progress.init(total=len(images), desc="Batch detection")
        
        results = []
        for i, image in enumerate(images):
            result = self.detect(image)
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
        """Preprocess image for detection."""
        # Resize maintaining aspect ratio
        width, height = image.size
        target_size = min(self.input_size)
        
        scale = target_size / min(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        image_tensor = F.to_tensor(image).to(self.device)
        
        return image_tensor
    
    def _process_predictions(
        self,
        prediction: Dict[str, torch.Tensor],
        original_size: Tuple[int, int],
        return_all: bool = False
    ) -> List[Detection]:
        """Process model predictions to Detection objects."""
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        
        detections = []
        
        # Scale factor for resizing
        width_scale = original_size[0] / self.input_size[0]
        height_scale = original_size[1] / self.input_size[1]
        
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            if score < self.confidence_threshold and not return_all:
                continue
            
            # Scale box back to original size
            x_min, y_min, x_max, y_max = box
            scaled_box = (
                x_min * width_scale,
                y_min * height_scale,
                x_max * width_scale,
                y_max * height_scale
            )
            
            # Get class label
            class_label = self.class_mapping.get(int(label), f"class_{label}")
            
            detection = Detection(
                bbox=scaled_box,
                label=class_label,
                confidence=float(score),
                class_id=int(label)
            )
            detections.append(detection)
        
        # Apply NMS if needed
        if self.nms_threshold < 1.0:
            detections = self._apply_nms(detections, self.nms_threshold)
        
        return detections
    
    def _apply_nms(
        self,
        detections: List[Detection],
        threshold: float
    ) -> List[Detection]:
        """Apply non-maximum suppression."""
        if len(detections) == 0:
            return detections
        
        # Group by class
        by_class = {}
        for det in detections:
            if det.class_id not in by_class:
                by_class[det.class_id] = []
            by_class[det.class_id].append(det)
        
        # Apply NMS per class
        result = []
        for class_dets in by_class.values():
            # Sort by confidence
            class_dets.sort(key=lambda x: x.confidence, reverse=True)
            
            keep = []
            while class_dets:
                best = class_dets.pop(0)
                keep.append(best)
                
                # Remove overlapping detections
                class_dets = [
                    d for d in class_dets
                    if self._iou(best.bbox, d.bbox) < threshold
                ]
            
            result.extend(keep)
        
        return result
    
    def _iou(self, box1: Tuple[float, ...], box2: Tuple[float, ...]) -> float:
        """Calculate Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def train(
        self,
        train_dataset: Any,
        val_dataset: Any,
        epochs: int = 50,
        learning_rate: float = 0.001,
        batch_size: int = 8
    ) -> Dict[str, List[float]]:
        """
        Train the detector on custom dataset.
        
        Args:
            train_dataset: Training dataset (torch.utils.data.Dataset).
            val_dataset: Validation dataset.
            epochs: Number of training epochs.
            learning_rate: Learning rate.
            batch_size: Batch size.
            
        Returns:
            Dictionary with training history.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        if self.model is None:
            self.model = self._create_model()
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, collate_fn=lambda x: tuple(zip(*x))
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, collate_fn=lambda x: tuple(zip(*x))
        )
        
        # Optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        history = {'train_loss': [], 'val_loss': []}
        
        self.progress.init(total=epochs, desc="Training detector")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for images, targets in train_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                train_loss += losses.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()
            
            val_loss /= len(val_loader)
            
            lr_scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            self.logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            self.progress.update(1)
        
        self.is_loaded = True
        return history


# Building footprint extraction convenience class
class BuildingFootprintDetector(ObjectDetector):
    """Specialized detector for building footprint extraction."""
    
    def __init__(self, **kwargs):
        super().__init__(
            model="faster_rcnn",
            num_classes=2,  # background + building
            **kwargs
        )
        self.class_mapping = {0: "building"}
    
    def extract_footprints(
        self,
        image: Union[str, np.ndarray],
        min_area: float = 20.0,
        output_format: str = "geojson"
    ) -> Union[DetectionResult, str]:
        """
        Extract building footprints from imagery.
        
        Args:
            image: Input image.
            min_area: Minimum building area in square meters.
            output_format: Output format (geojson, detections).
            
        Returns:
            DetectionResult or GeoJSON string.
        """
        result = self.detect(image)
        
        # Filter by area
        result.detections = [d for d in result.detections if d.area >= min_area]
        
        if output_format == "geojson":
            import json
            return json.dumps(result.to_geojson(), indent=2)
        
        return result


# Vehicle detection convenience class
class VehicleDetector(ObjectDetector):
    """Specialized detector for vehicle detection."""
    
    def __init__(self, **kwargs):
        super().__init__(
            model="retinanet",
            num_classes=4,  # background + car + truck + bus
            **kwargs
        )
        self.class_mapping = {1: "car", 2: "truck", 3: "bus"}
    
    def count_vehicles(self, image: Union[str, np.ndarray]) -> Dict[str, int]:
        """Count vehicles by type."""
        result = self.detect(image)
        return result.by_class
