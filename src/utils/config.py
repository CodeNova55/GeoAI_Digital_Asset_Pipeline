"""
Configuration Module
====================

Provides configuration management for the GeoAI pipeline.
Supports YAML configuration files with environment variable overrides.

Example:
    >>> config = Config.load("config.yaml")
    >>> print(config.ml.classification.n_estimators)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from copy import deepcopy


class ConfigError(Exception):
    """Configuration error."""
    pass


class ConfigValue:
    """Container for a configuration value."""
    
    def __init__(self, value: Any):
        self._value = value
    
    def __getattr__(self, name: str) -> Any:
        if isinstance(self._value, dict):
            if name in self._value:
                return self._wrap(self._value[name])
            raise AttributeError(f"Configuration key '{name}' not found")
        raise AttributeError(f"Cannot access '{name}' on non-dict configuration")
    
    def __getitem__(self, key: str) -> Any:
        if isinstance(self._value, dict):
            return self._wrap(self._value[key])
        raise TypeError(f"Configuration is not subscriptable")
    
    def __contains__(self, key: str) -> bool:
        if isinstance(self._value, dict):
            return key in self._value
        return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        if isinstance(self._value, dict):
            value = self._value.get(key, default)
            return self._wrap(value)
        return default
    
    def _wrap(self, value: Any) -> Any:
        """Wrap value in ConfigValue if it's a dict."""
        if isinstance(value, dict):
            return ConfigValue(value)
        return value
    
    def to_dict(self) -> Any:
        """Convert to dictionary."""
        return self._value
    
    def __repr__(self) -> str:
        return f"ConfigValue({self._value!r})"


@dataclass
class Config:
    """
    Configuration manager for the GeoAI pipeline.
    
    This class provides a flexible configuration system with support for
    YAML files, environment variable overrides, and nested configuration
    access.
    
    Attributes:
        data: Raw configuration data.
        path: Path to configuration file.
        
    Example:
        >>> config = Config.load("config.yaml")
        >>> n_estimators = config.ml.classification.n_estimators
        >>> paths = config.paths
        >>> data_dir = config.paths.data_dir
    """
    
    data: Dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None
    
    @classmethod
    def load(
        cls,
        path: str,
        env_prefix: str = "GEOAI",
        validate: bool = True
    ) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file.
            env_prefix: Prefix for environment variable overrides.
            validate: Whether to validate configuration.
            
        Returns:
            Config instance.
        """
        path = Path(path)
        
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {path}")
        
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML: {e}")
        
        if data is None:
            data = {}
        
        # Apply environment variable overrides
        data = cls._apply_env_overrides(data, env_prefix)
        
        config = cls(data=data, path=str(path))
        
        if validate:
            config.validate()
        
        return config
    
    @classmethod
    def default(cls) -> 'Config':
        """
        Create default configuration.
        
        Returns:
            Config instance with default values.
        """
        data = {
            "project": {
                "name": "GeoAI Digital Asset Pipeline",
                "version": "1.0.0",
                "crs_default": "EPSG:4326"
            },
            "paths": {
                "data_dir": "./data",
                "raw_data": "./data/raw",
                "processed_data": "./data/processed",
                "models_dir": "./models",
                "outputs_dir": "./outputs",
                "logs_dir": "./logs"
            },
            "qgis": {
                "version": "3.28",
                "max_features": 100000,
                "batch_size": 100
            },
            "ml": {
                "random_seed": 42,
                "test_size": 0.2,
                "classification": {
                    "algorithm": "random_forest",
                    "n_estimators": 100,
                    "max_depth": 20
                },
                "object_detection": {
                    "model": "faster_rcnn",
                    "confidence_threshold": 0.7
                },
                "segmentation": {
                    "model": "unet",
                    "input_size": [512, 512]
                }
            },
            "pipeline": {
                "feature_extraction": {
                    "enabled_features": [
                        "spectral_indices",
                        "texture_features",
                        "geometric_features"
                    ]
                },
                "quality_assurance": {
                    "enabled": True,
                    "auto_repair": True
                }
            },
            "workflow": {
                "logging": {
                    "level": "INFO",
                    "console_output": True
                },
                "progress": {
                    "enabled": True,
                    "show_eta": True
                },
                "parallel": {
                    "enabled": True,
                    "n_workers": -1
                }
            }
        }
        
        return cls(data=data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """
        Create configuration from dictionary.
        
        Args:
            data: Configuration dictionary.
            
        Returns:
            Config instance.
        """
        return cls(data=data)
    
    @staticmethod
    def _apply_env_overrides(
        data: Dict[str, Any],
        prefix: str
    ) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        result = deepcopy(data)
        
        for key, value in os.environ.items():
            if key.startswith(prefix + "_"):
                # Convert env var to config path
                config_path = key[len(prefix) + 1:].lower().split("__")
                
                # Navigate to the right place in the config
                current = result
                for path_part in config_path[:-1]:
                    if path_part not in current:
                        current[path_part] = {}
                    current = current[path_part]
                
                # Set the value
                final_key = config_path[-1]
                current[final_key] = Config._parse_env_value(value)
        
        return result
    
    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # List (comma-separated)
        if ',' in value:
            return [v.strip() for v in value.split(',')]
        
        # String
        return value
    
    def __getattr__(self, name: str) -> ConfigValue:
        if name in self.data:
            return ConfigValue(self.data[name])
        raise AttributeError(f"Configuration section '{name}' not found")
    
    def __getitem__(self, key: str) -> ConfigValue:
        if key in self.data:
            return ConfigValue(self.data[key])
        raise KeyError(f"Configuration section '{key}' not found")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        parts = key.split('.')
        current = self.data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        parts = key.split('.')
        current = self.data
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    def validate(self) -> List[str]:
        """
        Validate configuration.
        
        Returns:
            List of validation errors (empty if valid).
        """
        errors = []
        
        # Check required sections
        required_sections = ['project', 'paths']
        for section in required_sections:
            if section not in self.data:
                errors.append(f"Missing required section: {section}")
        
        # Check paths exist (create if needed)
        if 'paths' in self.data:
            for path_key, path_value in self.data['paths'].items():
                if path_key.endswith('_dir'):
                    path = Path(path_value)
                    if not path.exists():
                        try:
                            path.mkdir(parents=True, exist_ok=True)
                        except Exception as e:
                            errors.append(f"Cannot create directory {path}: {e}")
        
        # Check ML settings
        if 'ml' in self.data:
            ml = self.data['ml']
            if 'random_seed' in ml and not isinstance(ml['random_seed'], int):
                errors.append("ml.random_seed must be an integer")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return deepcopy(self.data)
    
    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.data, default_flow_style=False, sort_keys=False)
    
    def save(self, path: Optional[str] = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            path: Output path. Uses original path if None.
            
        Returns:
            True if save successful.
        """
        output_path = Path(path or self.path)
        
        if not output_path:
            raise ConfigError("No output path specified")
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)
            
            return True
            
        except Exception as e:
            raise ConfigError(f"Failed to save configuration: {e}")
    
    def merge(self, other: 'Config') -> 'Config':
        """
        Merge with another configuration.
        
        Args:
            other: Configuration to merge.
            
        Returns:
            New merged Config instance.
        """
        merged = deepcopy(self.data)
        merged = self._deep_merge(merged, other.data)
        
        return Config(data=merged)
    
    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def __repr__(self) -> str:
        return f"Config(path={self.path!r}, keys={list(self.data.keys())})"


# Convenience functions
def load_config(path: str) -> Config:
    """Load configuration from file."""
    return Config.load(path)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config.default()


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value using dot notation."""
    config = Config.default()
    return config.get(key, default)
