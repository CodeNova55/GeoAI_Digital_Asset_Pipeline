"""
Pipeline Logger Module
======================

Provides comprehensive logging for the GeoAI pipeline.
Supports multiple log levels, file rotation, and structured logging.

Example:
    >>> logger = PipelineLogger.get_logger("MyModule")
    >>> logger.info("Processing started")
    >>> logger.error("An error occurred", exc_info=True)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json

try:
    from loguru import logger as loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """Colored console formatter."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
        
        return json.dumps(log_data)


class PipelineLogger:
    """
    Comprehensive logging system for the GeoAI pipeline.
    
    This class provides a flexible logging system with support for
    multiple handlers, log levels, file rotation, and structured logging.
    
    Attributes:
        name: Logger name.
        level: Logging level.
        handlers: List of configured handlers.
        
    Example:
        >>> logger = PipelineLogger.get_logger("ProcessingModule")
        >>> logger.set_log_file("./logs/pipeline.log")
        >>> logger.info("Starting processing", extra={"files": 100})
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    _default_level = logging.INFO
    _log_dir: Optional[Path] = None
    
    def __init__(
        self,
        name: str,
        level: int = None,
        use_loguru: bool = False
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name.
            level: Logging level.
            use_loguru: Whether to use loguru instead of standard logging.
        """
        self.name = name
        self.level = level or self._default_level
        self.use_loguru = use_loguru and LOGURU_AVAILABLE
        
        if self.use_loguru:
            self._setup_loguru()
        else:
            self._setup_standard_logging()
    
    def _setup_standard_logging(self) -> None:
        """Set up standard Python logging."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        
        # Avoid duplicate handlers
        if self.logger.handlers:
            return
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(console_handler)
    
    def _setup_loguru(self) -> None:
        """Set up loguru logging."""
        # Remove default handler
        loguru_logger.remove()
        
        # Add console handler
        loguru_logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=self.level,
            colorize=True
        )
    
    def set_log_file(
        self,
        file_path: str,
        level: int = logging.DEBUG,
        rotation: str = "10 MB",
        backup_count: int = 5,
        json_format: bool = False
    ) -> None:
        """
        Add file handler.
        
        Args:
            file_path: Path to log file.
            level: Log level for file handler.
            rotation: Rotation policy (size or time).
            backup_count: Number of backup files to keep.
            json_format: Whether to use JSON format.
        """
        if self.use_loguru:
            loguru_logger.add(
                file_path,
                level=level,
                rotation=rotation,
                retention=backup_count,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
            )
        else:
            # Ensure log directory exists
            log_path = Path(file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create rotating file handler
            if rotation.endswith("MB"):
                max_bytes = int(rotation.replace("MB", "").strip()) * 1024 * 1024
                handler = RotatingFileHandler(
                    file_path,
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
            else:
                handler = TimedRotatingFileHandler(
                    file_path,
                    when='midnight',
                    backupCount=backup_count
                )
            
            handler.setLevel(level)
            
            if json_format:
                handler.setFormatter(JSONFormatter())
            else:
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                ))
            
            self.logger.addHandler(handler)
    
    def debug(self, msg: str, **kwargs) -> None:
        """Log debug message."""
        if self.use_loguru:
            loguru_logger.debug(msg, **kwargs)
        else:
            self.logger.debug(msg, **kwargs)
    
    def info(self, msg: str, **kwargs) -> None:
        """Log info message."""
        if self.use_loguru:
            loguru_logger.info(msg, **kwargs)
        else:
            self.logger.info(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs) -> None:
        """Log warning message."""
        if self.use_loguru:
            loguru_logger.warning(msg, **kwargs)
        else:
            self.logger.warning(msg, **kwargs)
    
    def error(self, msg: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message."""
        if self.use_loguru:
            loguru_logger.error(msg, exc_info=exc_info, **kwargs)
        else:
            self.logger.error(msg, exc_info=exc_info, **kwargs)
    
    def critical(self, msg: str, exc_info: bool = False, **kwargs) -> None:
        """Log critical message."""
        if self.use_loguru:
            loguru_logger.critical(msg, exc_info=exc_info, **kwargs)
        else:
            self.logger.critical(msg, exc_info=exc_info, **kwargs)
    
    def log(self, level: int, msg: str, **kwargs) -> None:
        """Log message at specified level."""
        if self.use_loguru:
            loguru_logger.log(level, msg, **kwargs)
        else:
            self.logger.log(level, msg, **kwargs)
    
    def set_level(self, level: int) -> None:
        """Set logging level."""
        self.level = level
        if self.use_loguru:
            loguru_logger.remove()
            self._setup_loguru()
        else:
            self.logger.setLevel(level)
            for handler in self.logger.handlers:
                handler.setLevel(level)
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        level: int = None,
        use_loguru: bool = False
    ) -> 'PipelineLogger':
        """
        Get or create a logger instance.
        
        Args:
            name: Logger name.
            level: Logging level.
            use_loguru: Whether to use loguru.
            
        Returns:
            PipelineLogger instance.
        """
        if name not in cls._loggers:
            cls._loggers[name] = cls(name, level, use_loguru)
        return cls._loggers[name]
    
    @classmethod
    def set_default_level(cls, level: int) -> None:
        """Set default logging level for new loggers."""
        cls._default_level = level
    
    @classmethod
    def set_log_dir(cls, log_dir: str) -> None:
        """Set default log directory."""
        cls._log_dir = Path(log_dir)
        cls._log_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def configure_all(
        cls,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        json_format: bool = False
    ) -> None:
        """
        Configure all existing loggers.
        
        Args:
            log_file: Optional log file path.
            level: Logging level.
            json_format: Whether to use JSON format.
        """
        cls._default_level = level
        
        for logger_instance in cls._loggers.values():
            logger_instance.set_level(level)
            if log_file:
                logger_instance.set_log_file(log_file, json_format=json_format)


class ProcessingLogger:
    """Specialized logger for processing workflows."""
    
    def __init__(self, base_logger: PipelineLogger):
        self.base_logger = base_logger
        self.processing_context: Dict[str, Any] = {}
    
    def start_processing(self, task_name: str, **context) -> None:
        """Log start of processing task."""
        self.processing_context = {
            "task": task_name,
            "start_time": datetime.now().isoformat(),
            **context
        }
        self.base_logger.info(f"Starting task: {task_name}", extra={"extra_data": self.processing_context})
    
    def end_processing(self, status: str = "success", **results) -> None:
        """Log end of processing task."""
        self.processing_context.update({
            "end_time": datetime.now().isoformat(),
            "status": status,
            **results
        })
        
        duration = (
            datetime.fromisoformat(self.processing_context["end_time"]) -
            datetime.fromisoformat(self.processing_context["start_time"])
        ).total_seconds()
        
        self.processing_context["duration_seconds"] = duration
        
        self.base_logger.info(
            f"Completed task: {self.processing_context['task']} - {status}",
            extra={"extra_data": self.processing_context}
        )
    
    def log_step(self, step_name: str, **details) -> None:
        """Log processing step."""
        self.base_logger.info(f"Step: {step_name}", extra={"extra_data": {
            "task": self.processing_context.get("task"),
            **details
        }})
    
    def log_error(self, error: Exception, step: Optional[str] = None) -> None:
        """Log processing error."""
        self.base_logger.error(
            f"Error in {step or self.processing_context.get('task')}: {str(error)}",
            exc_info=True,
            extra={"extra_data": {
                "task": self.processing_context.get("task"),
                "error_type": type(error).__name__
            }}
        )


# Convenience functions
def get_logger(name: str) -> PipelineLogger:
    """Get a logger instance."""
    return PipelineLogger.get_logger(name)


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    json_format: bool = False
) -> None:
    """Set up logging configuration."""
    PipelineLogger.configure_all(log_file, level, json_format)
