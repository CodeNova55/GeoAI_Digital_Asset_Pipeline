"""
Spatial Classifier Module
=========================

Provides machine learning-based spatial classification for land cover,
land use, and other geospatial categorization tasks.

Supports multiple algorithms: Random Forest, XGBoost, SVM, Neural Networks.
Compatible with DeepForest plugin for tree detection.

Example:
    >>> classifier = SpatialClassifier(algorithm="random_forest")
    >>> classifier.train(X_train, y_train)
    >>> predictions = classifier.predict(X_test)
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import logging
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Define stub types for when PyTorch is not available
    class _MockNN:
        class Module: pass
        class Sequential: pass
        class Linear: 
            def __init__(self, *args, **kwargs): pass
        class BatchNorm1d: 
            def __init__(self, *args, **kwargs): pass
        class ReLU: 
            def __init__(self, *args, **kwargs): pass
        class Dropout: 
            def __init__(self, *args, **kwargs): pass
        class CrossEntropyLoss: 
            def __init__(self, *args, **kwargs): pass
    class _MockTorch:
        class device:
            def __init__(self, *args): pass
            def __str__(self): return 'cpu'
        @staticmethod
        def cuda(): return False
    nn = _MockNN()
    torch = _MockTorch()
    optim = type('optim', (), {'Adam': type('Adam', (), {})})()

from src.utils.config import Config
from src.workflow.logger import PipelineLogger
from src.workflow.progress_tracker import ProgressTracker


@dataclass
class ClassificationResult:
    """Container for classification results."""
    
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    class_names: List[str] = field(default_factory=list)
    feature_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "confusion_matrix": self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            "class_names": self.class_names,
            "feature_importance": self.feature_importance
        }


@dataclass
class TrainingResult:
    """Container for training results."""
    
    model_trained: bool = False
    training_accuracy: float = 0.0
    validation_accuracy: float = 0.0
    cross_val_scores: List[float] = field(default_factory=list)
    training_time: float = 0.0
    model_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class NeuralClassifier(nn.Module):
    """Simple neural network classifier for spatial data."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class SpatialClassifier:
    """
    Machine learning classifier for spatial/geospatial data.
    
    This class provides multiple classification algorithms optimized
    for spatial data, including feature importance analysis and
    comprehensive evaluation metrics.
    
    Attributes:
        algorithm: Classification algorithm to use.
        config: Configuration object with ML parameters.
        logger: Logger instance.
        model: Trained model.
        scaler: Feature scaler.
        label_encoder: Label encoder for classes.
        
    Example:
        >>> classifier = SpatialClassifier(
        ...     algorithm="random_forest",
        ...     n_estimators=100,
        ...     max_depth=20
        ... )
        >>> result = classifier.train(X, y, feature_names=["ndvi", "ndwi", "elevation"])
        >>> predictions = classifier.predict(X_new)
    """
    
    def __init__(
        self,
        algorithm: str = "random_forest",
        config: Optional[Config] = None,
        logger: Optional[PipelineLogger] = None,
        **kwargs
    ):
        """
        Initialize the spatial classifier.
        
        Args:
            algorithm: Algorithm name (random_forest, xgboost, svm, neural_network).
            config: Configuration object.
            logger: Logger instance.
            **kwargs: Algorithm-specific parameters.
        """
        self.algorithm = algorithm
        self.config = config or Config.default()
        self.logger = logger or PipelineLogger.get_logger("SpatialClassifier")
        self.params = kwargs
        
        # Load ML config
        ml_config = self.config.ml if hasattr(self.config, 'ml') else {}
        class_config = ml_config.get('classification', {})
        
        # Set default parameters
        self._set_default_params(class_config)
        
        # Initialize model components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.class_names: List[str] = []
        self.is_trained = False
        
        self.progress = ProgressTracker()
        
        self.logger.info(f"SpatialClassifier initialized with algorithm: {algorithm}")
    
    def _set_default_params(self, class_config: Dict[str, Any]) -> None:
        """Set default parameters from config."""
        defaults = {
            "random_forest": {
                "n_estimators": class_config.get("n_estimators", 100),
                "max_depth": class_config.get("max_depth", 20),
                "min_samples_split": class_config.get("min_samples_split", 5),
                "min_samples_leaf": class_config.get("min_samples_leaf", 2),
                "class_weight": class_config.get("class_weights", "balanced"),
                "n_jobs": -1,
                "random_state": 42
            },
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "objective": "multi:softmax",
                "eval_metric": "mlogloss",
                "random_state": 42
            },
            "svm": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "probability": True,
                "random_state": 42
            },
            "neural_network": {
                "hidden_layer_sizes": (128, 64, 32),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.0001,
                "batch_size": 32,
                "learning_rate": "adaptive",
                "max_iter": 200,
                "random_state": 42
            }
        }
        
        # Update with config values
        config_dict = class_config.to_dict() if hasattr(class_config, 'to_dict') else dict(class_config)
        for key, value in config_dict.items():
            if key in defaults and isinstance(defaults[key], dict):
                if isinstance(value, dict):
                    defaults[key].update(value)

        self.default_params = defaults
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        validation_size: float = 0.2,
        cross_validation: bool = True,
        n_folds: int = 5
    ) -> TrainingResult:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target labels.
            feature_names: Names of features.
            class_names: Names of classes.
            validation_size: Fraction of data for validation.
            cross_validation: Whether to perform cross-validation.
            n_folds: Number of CV folds.
            
        Returns:
            TrainingResult with training metrics.
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Training {self.algorithm} classifier on {X.shape[0]} samples")
        self.progress.init(total=100, description="Training classifier")
        
        # Store feature and class names
        if feature_names:
            self.feature_names = feature_names
        if class_names:
            self.class_names = class_names
            self.label_encoder.fit(class_names)
            y_encoded = self.label_encoder.transform(y)
        else:
            y_encoded = y
            self.class_names = list(np.unique(y))
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=validation_size, random_state=42, stratify=y_encoded
        )
        
        self.progress.update(10)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        self.progress.update(20)
        
        # Initialize and train model
        self.model = self._create_model()
        
        if self.algorithm == "neural_network" and TORCH_AVAILABLE:
            # PyTorch training
            self._train_pytorch(X_train_scaled, y_train, X_val_scaled, y_val)
        else:
            # Scikit-learn training
            self.model.fit(X_train_scaled, y_train)
        
        self.progress.update(60)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        self.progress.update(80)
        
        # Cross-validation
        cv_scores = []
        if cross_validation:
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, cv=n_folds, scoring='accuracy'
            ).tolist()
            self.logger.info(f"Cross-validation scores: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
        
        self.progress.update(95)
        
        training_time = time.time() - start_time
        
        result = TrainingResult(
            model_trained=True,
            training_accuracy=train_accuracy,
            validation_accuracy=val_accuracy,
            cross_val_scores=cv_scores,
            training_time=training_time,
            metrics={
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "cv_mean": np.mean(cv_scores) if cv_scores else 0,
                "cv_std": np.std(cv_scores) if cv_scores else 0
            }
        )
        
        self.is_trained = True
        self.progress.update(100)
        
        self.logger.info(
            f"Training complete in {training_time:.2f}s - "
            f"Train: {train_accuracy:.3f}, Val: {val_accuracy:.3f}"
        )
        
        return result
    
    def _create_model(self):
        """Create model based on algorithm."""
        params = self.default_params.get(self.algorithm, {})
        params.update(self.params)
        
        if self.algorithm == "random_forest":
            return RandomForestClassifier(**params)
        
        elif self.algorithm == "xgboost":
            if not XGBOOST_AVAILABLE:
                self.logger.warning("XGBoost not available, falling back to Random Forest")
                return RandomForestClassifier(**self.default_params["random_forest"])
            return xgb.XGBClassifier(**params)
        
        elif self.algorithm == "svm":
            return SVC(**params)
        
        elif self.algorithm == "neural_network":
            if TORCH_AVAILABLE:
                return None  # Will be created in _train_pytorch
            return MLPClassifier(**params)
        
        elif self.algorithm == "gradient_boosting":
            return GradientBoostingClassifier(**params)
        
        else:
            self.logger.warning(f"Unknown algorithm: {self.algorithm}, using Random Forest")
            return RandomForestClassifier(**self.default_params["random_forest"])
    
    def _train_pytorch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> None:
        """Train PyTorch neural network."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        params = self.default_params.get("neural_network", {})
        params.update(self.params)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)
        
        # Create datasets and loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=params.get("batch_size", 32), shuffle=True)
        
        # Create model
        input_size = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        hidden_sizes = params.get("hidden_layer_sizes", (128, 64, 32))
        
        self.model = NeuralClassifier(input_size, hidden_sizes, num_classes)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=params.get("learning_rate", 0.001))
        
        # Training loop
        epochs = params.get("max_iter", 200)
        best_val_acc = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_pred = torch.argmax(val_outputs, dim=1)
                val_acc = (val_pred == y_val_t).float().mean().item()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if (epoch + 1) % 20 == 0:
                self.logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        self.logger.info(f"PyTorch training complete. Best validation accuracy: {best_val_acc:.4f}")
    
    def predict(
        self,
        X: np.ndarray,
        return_probabilities: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions.
        
        Args:
            X: Feature matrix.
            return_probabilities: Whether to return class probabilities.
            
        Returns:
            Predictions or (predictions, probabilities).
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.algorithm == "neural_network" and TORCH_AVAILABLE and isinstance(self.model, NeuralClassifier):
            # PyTorch prediction
            self.model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X_scaled)
                outputs = self.model(X_t)
                probabilities = torch.softmax(outputs, dim=1).numpy()
                predictions = np.argmax(probabilities, axis=1)
        else:
            # Scikit-learn prediction
            predictions = self.model.predict(X_scaled)
            
            if return_probabilities and hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(X_scaled)
            else:
                probabilities = None
        
        if return_probabilities:
            return predictions, probabilities
        return predictions
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> ClassificationResult:
        """
        Evaluate model on test data.
        
        Args:
            X: Feature matrix.
            y: True labels.
            
        Returns:
            ClassificationResult with metrics.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Encode labels if needed
        if len(self.class_names) > 0:
            y_encoded = self.label_encoder.transform(y)
        else:
            y_encoded = y
        
        # Predict
        predictions, probabilities = self.predict(X, return_probabilities=True)
        
        # Calculate metrics
        accuracy = accuracy_score(y_encoded, predictions)
        precision = precision_score(y_encoded, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_encoded, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_encoded, predictions, average='weighted', zero_division=0)
        cm = confusion_matrix(y_encoded, predictions)
        
        # Feature importance
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = {
                name: float(importance)
                for name, importance in zip(self.feature_names, self.model.feature_importances_)
            }
        elif hasattr(self.model, 'coef_'):
            # Linear models
            feature_importance = {
                name: float(abs(self.model.coef_[0][i]))
                for i, name in enumerate(self.feature_names)
            }
        
        return ClassificationResult(
            predictions=predictions,
            probabilities=probabilities,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm,
            class_names=self.class_names,
            feature_importance=feature_importance
        )
    
    def save_model(self, path: str) -> bool:
        """
        Save trained model to file.
        
        Args:
            path: Output path for model file.
            
        Returns:
            True if save successful.
        """
        if not self.is_trained:
            self.logger.warning("No trained model to save")
            return False
        
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            if TORCH_AVAILABLE and isinstance(self.model, nn.Module):
                torch.save(self.model.state_dict(), path)
            else:
                import joblib
                joblib.dump(self.model, path)
            
            # Save metadata
            metadata_path = str(Path(path).with_suffix('.json'))
            metadata = {
                "algorithm": self.algorithm,
                "feature_names": self.feature_names,
                "class_names": self.class_names,
                "is_trained": self.is_trained,
                "scaler_mean": self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                "scaler_scale": self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model saved to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Load trained model from file.
        
        Args:
            path: Path to model file.
            
        Returns:
            True if load successful.
        """
        try:
            # Load metadata
            metadata_path = str(Path(path).with_suffix('.json'))
            if Path(metadata_path).exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.algorithm = metadata.get("algorithm", "random_forest")
                self.feature_names = metadata.get("feature_names", [])
                self.class_names = metadata.get("class_names", [])
                self.is_trained = metadata.get("is_trained", False)
            
            # Load model
            if TORCH_AVAILABLE and self.algorithm == "neural_network":
                self.model = NeuralClassifier(
                    len(self.feature_names),
                    (128, 64, 32),
                    len(self.class_names)
                )
                self.model.load_state_dict(torch.load(path))
            else:
                import joblib
                self.model = joblib.load(path)
            
            # Restore scaler
            if metadata and 'scaler_mean' in metadata:
                self.scaler.mean_ = np.array(metadata['scaler_mean'])
                self.scaler.scale_ = np.array(metadata['scaler_scale'])
                self.scaler.n_features_in_ = len(metadata['scaler_mean'])
            
            self.logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from trained model."""
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return {
                name: float(importance)
                for name, importance in zip(self.feature_names, self.model.feature_importances_)
            }
        return None
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Any:
        """Plot confusion matrix."""
        import matplotlib.pyplot as plt
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.class_names,
               yticklabels=self.class_names,
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        return fig


# Land cover classification convenience class
class LandCoverClassifier(SpatialClassifier):
    """Specialized classifier for land cover classification."""
    
    LAND_COVER_CLASSES = [
        "water", "forest", "grassland", "cropland",
        "urban", "bare_soil", "wetland", "snow"
    ]
    
    def __init__(self, **kwargs):
        super().__init__(algorithm="random_forest", **kwargs)
        self.class_names = self.LAND_COVER_CLASSES
    
    def train_with_sentinel2(
        self,
        sentinel2_data: np.ndarray,
        labels: np.ndarray,
        include_indices: bool = True
    ) -> TrainingResult:
        """
        Train with Sentinel-2 data.
        
        Args:
            sentinel2_data: Sentinel-2 bands (B02, B03, B04, B08, B11, B12).
            labels: Land cover labels.
            include_indices: Whether to calculate spectral indices.
            
        Returns:
            TrainingResult.
        """
        # Calculate spectral indices if requested
        if include_indices:
            features = self._calculate_spectral_indices(sentinel2_data)
        else:
            features = sentinel2_data
        
        feature_names = [
            "B02", "B03", "B04", "B08", "B11", "B12",
            "NDVI", "NDWI", "NDBI", "EVI"
        ] if include_indices else [
            "B02", "B03", "B04", "B08", "B11", "B12"
        ]
        
        return self.train(features, labels, feature_names=feature_names)
    
    def _calculate_spectral_indices(self, data: np.ndarray) -> np.ndarray:
        """Calculate common spectral indices from Sentinel-2 bands."""
        # Assume bands are in order: B02, B03, B04, B08, B11, B12
        B02 = data[:, 0].astype(float)  # Blue
        B03 = data[:, 1].astype(float)  # Green
        B04 = data[:, 2].astype(float)  # Red
        B08 = data[:, 3].astype(float)  # NIR
        B11 = data[:, 4].astype(float)  # SWIR1
        B12 = data[:, 5].astype(float)  # SWIR2
        
        # NDVI
        ndvi = (B08 - B04) / (B08 + B04 + 1e-10)
        
        # NDWI
        ndwi = (B03 - B08) / (B03 + B08 + 1e-10)
        
        # NDBI
        ndbi = (B11 - B08) / (B11 + B08 + 1e-10)
        
        # EVI
        evi = 2.5 * (B08 - B04) / (B08 + 6 * B04 - 7.5 * B02 + 10001 + 1e-10)
        
        return np.column_stack([data, ndvi, ndwi, ndbi, evi])


# DeepForest compatibility wrapper
class DeepForestWrapper:
    """Wrapper for DeepForest plugin compatibility."""
    
    def __init__(self, classifier: SpatialClassifier):
        self.classifier = classifier
    
    def predict_trees(self, image_data: np.ndarray) -> np.ndarray:
        """Predict tree locations from image data."""
        return self.classifier.predict(image_data)
    
    def save_predictions(self, predictions: np.ndarray, output_path: str) -> None:
        """Save predictions in DeepForest format."""
        import geopandas as gpd
        from shapely.geometry import Point
        
        # Convert predictions to points
        points = [Point(i, j) for i, j in np.argwhere(predictions > 0.5)]
        gdf = gpd.GeoDataFrame({"geometry": points, "confidence": predictions[predictions > 0.5]})
        gdf.set_crs("EPSG:4326", inplace=True)
        gdf.to_file(output_path)
