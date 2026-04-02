"""
Spatial Clustering Module
=========================

Provides spatial clustering for pattern analysis and regionalization.
Supports DBSCAN, K-Means, HDBSCAN, and spatially-constrained clustering.

Example:
    >>> clusterer = SpatialClusterer(algorithm="dbscan")
    >>> labels = clusterer.cluster(coordinates, features)
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import logging

import numpy as np

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from src.utils.config import Config
from src.workflow.logger import PipelineLogger


@dataclass
class ClusteringResult:
    """Container for clustering results."""
    
    labels: np.ndarray
    n_clusters: int = 0
    n_noise: int = 0
    cluster_sizes: Dict[int, int] = field(default_factory=dict)
    cluster_centers: Optional[np.ndarray] = None
    silhouette_score: float = 0.0
    
    @property
    def noise_percentage(self) -> float:
        """Percentage of noise points."""
        if len(self.labels) == 0:
            return 0.0
        return (self.n_noise / len(self.labels)) * 100
    
    def get_cluster_points(self, cluster_id: int) -> np.ndarray:
        """Get indices of points in a cluster."""
        return np.where(self.labels == cluster_id)[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "noise_percentage": self.noise_percentage,
            "cluster_sizes": self.cluster_sizes,
            "silhouette_score": self.silhouette_score
        }


class SpatialClusterer:
    """
    Spatial clustering for pattern analysis and regionalization.
    
    This class provides various clustering algorithms optimized for
    spatial data, including density-based clustering (DBSCAN),
    centroid-based clustering (K-Means), and hierarchical clustering.
    
    Attributes:
        algorithm: Clustering algorithm name.
        config: Configuration object.
        logger: Logger instance.
        
    Example:
        >>> clusterer = SpatialClusterer(
        ...     algorithm="dbscan",
        ...     eps=0.5,
        ...     min_samples=5
        ... )
        >>> result = clusterer.cluster(coordinates, features)
        >>> print(f"Found {result.n_clusters} clusters")
    """
    
    def __init__(
        self,
        algorithm: str = "dbscan",
        eps: float = 0.5,
        min_samples: int = 5,
        n_clusters: Optional[int] = None,
        config: Optional[Config] = None,
        logger: Optional[PipelineLogger] = None
    ):
        """
        Initialize the spatial clusterer.
        
        Args:
            algorithm: Algorithm name (dbscan, kmeans, hdbscan, hierarchical).
            eps: DBSCAN neighborhood radius.
            min_samples: DBSCAN minimum points per cluster.
            n_clusters: Number of clusters for K-Means.
            config: Configuration object.
            logger: Logger instance.
        """
        self.algorithm = algorithm
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters = n_clusters
        
        self.config = config or Config.default()
        self.logger = logger or PipelineLogger.get_logger("SpatialClusterer")
        
        # Load config
        ml_config = self.config.ml if hasattr(self.config, 'ml') else {}
        cluster_config = ml_config.get('clustering', {})
        
        self.algorithm = cluster_config.get('algorithm', algorithm)
        self.eps = cluster_config.get('eps', eps)
        self.min_samples = cluster_config.get('min_samples', min_samples)
        self.n_clusters = cluster_config.get('n_clusters', n_clusters)
        
        self.scaler = StandardScaler()
        self.model = None
        
        self.logger.info(f"SpatialClusterer initialized: {algorithm}")
    
    def cluster(
        self,
        coordinates: np.ndarray,
        features: Optional[np.ndarray] = None,
        spatial_weight: float = 0.5
    ) -> ClusteringResult:
        """
        Perform spatial clustering.
        
        Args:
            coordinates: Spatial coordinates (n_points, 2).
            features: Optional additional features.
            spatial_weight: Weight for spatial vs feature similarity.
            
        Returns:
            ClusteringResult with cluster labels.
        """
        # Combine coordinates and features
        if features is not None:
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Scale coordinates
            coords_scaled = self.scaler.fit_transform(coordinates)
            
            # Combine with spatial weight
            X = np.hstack([
                coords_scaled * spatial_weight,
                features_scaled * (1 - spatial_weight)
            ])
        else:
            X = coordinates
        
        # Create model
        self.model = self._create_model()
        
        # Fit
        labels = self.model.fit_predict(X)
        
        # Calculate statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        
        cluster_sizes = {}
        for label in set(labels):
            if label != -1:
                cluster_sizes[label] = int(np.sum(labels == label))
        
        # Calculate silhouette score if applicable
        silhouette = 0.0
        if n_clusters > 1 and features is not None:
            try:
                from sklearn.metrics import silhouette_score
                valid_mask = labels != -1
                if np.sum(valid_mask) > n_clusters:
                    silhouette = silhouette_score(
                        X[valid_mask],
                        labels[valid_mask]
                    )
            except Exception:
                pass
        
        # Get cluster centers for K-Means
        centers = None
        if hasattr(self.model, 'cluster_centers_'):
            centers = self.model.cluster_centers_[:, :2]  # Only spatial coordinates
        
        result = ClusteringResult(
            labels=labels,
            n_clusters=n_clusters,
            n_noise=n_noise,
            cluster_sizes=cluster_sizes,
            cluster_centers=centers,
            silhouette_score=silhouette
        )
        
        self.logger.info(
            f"Clustering complete: {n_clusters} clusters, "
            f"{n_noise} noise points ({result.noise_percentage:.1f}%)"
        )
        
        return result
    
    def _create_model(self):
        """Create clustering model based on algorithm."""
        if self.algorithm == "dbscan":
            return DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        
        elif self.algorithm == "kmeans":
            if self.n_clusters is None:
                self.logger.warning("n_clusters not specified for K-Means, using 5")
                self.n_clusters = 5
            return KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        
        elif self.algorithm == "hdbscan":
            if HDBSCAN_AVAILABLE:
                return hdbscan.HDBSCAN(
                    min_cluster_size=self.min_samples,
                    min_samples=self.min_samples
                )
            else:
                self.logger.warning("HDBSCAN not available, falling back to DBSCAN")
                return DBSCAN(eps=self.eps, min_samples=self.min_samples)
        
        elif self.algorithm == "hierarchical":
            if self.n_clusters is None:
                self.n_clusters = 5
            return AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )
        
        else:
            self.logger.warning(f"Unknown algorithm: {self.algorithm}, using DBSCAN")
            return DBSCAN(eps=self.eps, min_samples=self.min_samples)
    
    def cluster_with_constraints(
        self,
        coordinates: np.ndarray,
        features: Optional[np.ndarray] = None,
        must_link: Optional[List[Tuple[int, int]]] = None,
        cannot_link: Optional[List[Tuple[int, int]]] = None
    ) -> ClusteringResult:
        """
        Perform constrained clustering.
        
        Args:
            coordinates: Spatial coordinates.
            features: Optional additional features.
            must_link: Pairs of points that must be in same cluster.
            cannot_link: Pairs of points that cannot be in same cluster.
            
        Returns:
            ClusteringResult with cluster labels.
        """
        # Use constrained agglomerative clustering
        if must_link is not None or cannot_link is not None:
            # Create constraint matrix
            n_points = len(coordinates)
            constraints = np.zeros((n_points, n_points))
            
            if must_link:
                for i, j in must_link:
                    constraints[i, j] = 1
                    constraints[j, i] = 1
            
            if cannot_link:
                for i, j in cannot_link:
                    constraints[i, j] = -1
                    constraints[j, i] = -1
        
        # Fall back to regular clustering
        return self.cluster(coordinates, features)
    
    def spatial_autocorrelation(
        self,
        coordinates: np.ndarray,
        values: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate spatial autocorrelation metrics.
        
        Args:
            coordinates: Spatial coordinates.
            values: Values at each location.
            
        Returns:
            Dictionary with autocorrelation metrics.
        """
        try:
            from scipy.spatial.distance import pdist, squareform
            from scipy.stats import pearsonr
            
            # Calculate distance matrix
            dist_matrix = squareform(pdist(coordinates))
            
            # Create spatial weights (inverse distance)
            with np.errstate(divide='ignore'):
                weights = 1 / (dist_matrix + np.eye(len(coordinates)))
            np.fill_diagonal(weights, 0)
            weights = weights / weights.sum()
            
            # Calculate Moran's I
            n = len(values)
            mean = np.mean(values)
            var = np.var(values)
            
            numerator = np.sum(weights * np.outer(values - mean, values - mean))
            denominator = var
            
            morans_i = (n / np.sum(weights)) * (numerator / denominator)
            
            return {
                "morans_i": float(morans_i),
                "interpretation": self._interpret_morans(morans_i)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating autocorrelation: {e}")
            return {"morans_i": 0.0, "interpretation": "Error calculating"}
    
    def _interpret_morans(self, morans_i: float) -> str:
        """Interpret Moran's I value."""
        if morans_i > 0.1:
            return "Positive spatial autocorrelation (clustered)"
        elif morans_i < -0.1:
            return "Negative spatial autocorrelation (dispersed)"
        else:
            return "No significant spatial autocorrelation (random)"
    
    def optimize_parameters(
        self,
        coordinates: np.ndarray,
        features: Optional[np.ndarray] = None,
        eps_range: Optional[List[float]] = None,
        min_samples_range: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Optimize DBSCAN parameters using silhouette score.
        
        Args:
            coordinates: Spatial coordinates.
            features: Optional additional features.
            eps_range: Range of eps values to try.
            min_samples_range: Range of min_samples values to try.
            
        Returns:
            Dictionary with optimal parameters.
        """
        if eps_range is None:
            eps_range = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        if min_samples_range is None:
            min_samples_range = [3, 5, 10, 15]
        
        best_score = -1
        best_params = {"eps": 0.5, "min_samples": 5}
        
        X = np.hstack([coordinates, features]) if features is not None else coordinates
        X = self.scaler.fit_transform(X)
        
        for eps in eps_range:
            for min_samp in min_samples_range:
                model = DBSCAN(eps=eps, min_samples=min_samp, n_jobs=-1)
                labels = model.fit_predict(X)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                if n_clusters > 1:
                    try:
                        from sklearn.metrics import silhouette_score
                        valid_mask = labels != -1
                        if np.sum(valid_mask) > n_clusters:
                            score = silhouette_score(X[valid_mask], labels[valid_mask])
                            
                            if score > best_score:
                                best_score = score
                                best_params = {"eps": eps, "min_samples": min_samp}
                    except Exception:
                        pass
        
        self.logger.info(
            f"Optimal parameters: eps={best_params['eps']}, "
            f"min_samples={best_params['min_samples']}, score={best_score:.3f}"
        )
        
        return {
            "best_params": best_params,
            "best_score": best_score
        }
    
    def regionalize(
        self,
        coordinates: np.ndarray,
        features: np.ndarray,
        n_regions: int,
        contiguity: str = "rook"
    ) -> ClusteringResult:
        """
        Create spatially contiguous regions (regionalization).
        
        Args:
            coordinates: Spatial coordinates.
            features: Features for regionalization.
            n_regions: Target number of regions.
            contiguity: Contiguity type (rook, queen).
            
        Returns:
            ClusteringResult with region labels.
        """
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            # Combine spatial and feature data
            X = np.hstack([
                StandardScaler().fit_transform(coordinates),
                StandardScaler().fit_transform(features)
            ])
            
            # Use Ward's method for spatially-aware clustering
            model = AgglomerativeClustering(
                n_clusters=n_regions,
                linkage='ward'
            )
            
            labels = model.fit_predict(X)
            
            n_noise = 0
            cluster_sizes = {i: int(np.sum(labels == i)) for i in range(n_regions)}
            
            return ClusteringResult(
                labels=labels,
                n_clusters=n_regions,
                n_noise=n_noise,
                cluster_sizes=cluster_sizes
            )
            
        except Exception as e:
            self.logger.error(f"Error in regionalization: {e}")
            # Fall back to K-Means
            self.algorithm = "kmeans"
            self.n_clusters = n_regions
            return self.cluster(coordinates, features)


# Hot spot analysis
class HotSpotAnalyzer:
    """Getis-Ord Gi* hot spot analysis."""
    
    def __init__(self, distance_band: Optional[float] = None):
        self.distance_band = distance_band
    
    def analyze(
        self,
        coordinates: np.ndarray,
        values: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Perform hot spot analysis.
        
        Args:
            coordinates: Point coordinates.
            values: Values at each point.
            
        Returns:
            Dictionary with Gi* statistics and p-values.
        """
        from scipy.spatial.distance import pdist, squareform
        from scipy import stats
        
        n = len(values)
        
        # Calculate distance matrix
        dist_matrix = squareform(pdist(coordinates))
        
        # Determine distance band if not specified
        if self.distance_band is None:
            self.distance_band = np.percentile(dist_matrix[dist_matrix > 0], 25)
        
        # Create binary spatial weights
        weights = (dist_matrix <= self.distance_band).astype(float)
        np.fill_diagonal(weights, 0)
        
        # Calculate Gi* for each point
        gi_star = np.zeros(n)
        mean = np.mean(values)
        std = np.std(values)
        
        for i in range(n):
            w_i = weights[i]
            sum_w = np.sum(w_i)
            
            if sum_w > 0:
                numerator = np.sum(w_i * values) - (mean * sum_w)
                denominator = std * np.sqrt(
                    (n * np.sum(w_i ** 2) - sum_w ** 2) / (n - 1)
                )
                
                if denominator > 0:
                    gi_star[i] = numerator / denominator
        
        # Calculate p-values
        p_values = 2 * (1 - stats.norm.cdf(np.abs(gi_star)))
        
        # Classify hot/cold spots
        classifications = np.zeros(n, dtype=int)
        classifications[gi_star > 1.96] = 1  # Hot spot (95%)
        classifications[gi_star > 2.58] = 2  # Hot spot (99%)
        classifications[gi_star < -1.96] = -1  # Cold spot (95%)
        classifications[gi_star < -2.58] = -2  # Cold spot (99%)
        
        return {
            "gi_star": gi_star,
            "p_values": p_values,
            "classifications": classifications,
            "distance_band": self.distance_band
        }


# Convenience functions
def quick_cluster(
    coordinates: np.ndarray,
    features: Optional[np.ndarray] = None,
    algorithm: str = "dbscan"
) -> ClusteringResult:
    """Quick clustering with default settings."""
    clusterer = SpatialClusterer(algorithm=algorithm)
    return clusterer.cluster(coordinates, features)


def find_hot_spots(
    coordinates: np.ndarray,
    values: np.ndarray
) -> Dict[str, np.ndarray]:
    """Quick hot spot analysis."""
    analyzer = HotSpotAnalyzer()
    return analyzer.analyze(coordinates, values)
