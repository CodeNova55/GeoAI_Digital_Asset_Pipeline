"""
Progress Tracker Module
=======================

Provides progress tracking for long-running geospatial processing tasks.
Supports progress bars, ETA calculation, and memory monitoring.

Example:
    >>> tracker = ProgressTracker()
    >>> tracker.init(total=100, desc="Processing")
    >>> for item in items:
    ...     process(item)
    ...     tracker.update(1)
"""

import os
import sys
import time
import psutil
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class ProgressState:
    """Container for progress state."""
    
    total: int = 0
    current: int = 0
    description: str = ""
    start_time: float = 0.0
    last_update: float = 0.0
    completed: bool = False
    
    @property
    def percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def rate(self) -> float:
        """Calculate processing rate (items/second)."""
        elapsed = self.elapsed
        if elapsed == 0:
            return 0.0
        return self.current / elapsed
    
    @property
    def eta(self) -> Optional[float]:
        """Calculate estimated time remaining in seconds."""
        if self.rate == 0:
            return None
        remaining = self.total - self.current
        return remaining / self.rate
    
    def format_eta(self) -> str:
        """Format ETA as human-readable string."""
        eta_seconds = self.eta
        if eta_seconds is None:
            return "--:--:--"
        
        hours, remainder = divmod(int(eta_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def format_elapsed(self) -> str:
        """Format elapsed time as human-readable string."""
        elapsed = self.elapsed
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ProgressBar:
    """Simple text-based progress bar."""
    
    def __init__(
        self,
        total: int,
        description: str = "",
        width: int = 40,
        show_eta: bool = True,
        show_memory: bool = False
    ):
        self.total = total
        self.description = description
        self.width = width
        self.show_eta = show_eta
        self.show_memory = show_memory
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1) -> None:
        """Update progress bar."""
        self.current += n
        self.render()
    
    def render(self) -> None:
        """Render progress bar to console."""
        percentage = (self.current / self.total * 100) if self.total > 0 else 0
        filled = int(self.width * self.current / self.total) if self.total > 0 else 0
        bar = '█' * filled + '░' * (self.width - filled)
        
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        
        parts = [f"\r{self.description}: |{bar}| {percentage:5.1f}%"]
        parts.append(f" ({self.current}/{self.total})")
        
        if self.show_eta and self.total > 0:
            remaining = self.total - self.current
            eta = remaining / rate if rate > 0 else 0
            eta_str = f" ETA: {int(eta//3600):02d}:{int((eta%3600)//60):02d}:{int(eta%60):02d}"
            parts.append(eta_str)
        
        if self.show_memory:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            parts.append(f" Mem: {memory_mb:.1f}MB")
        
        sys.stdout.write(''.join(parts))
        sys.stdout.flush()
    
    def close(self) -> None:
        """Finish progress bar."""
        self.current = self.total
        self.render()
        print()  # New line


class ProgressTracker:
    """
    Progress tracking for long-running geospatial processing tasks.
    
    This class provides comprehensive progress tracking with support for
    progress bars, ETA calculation, memory monitoring, and callbacks.
    
    Attributes:
        state: Current progress state.
        show_progress: Whether to display progress.
        show_eta: Whether to show ETA.
        show_memory: Whether to show memory usage.
        update_interval: Minimum time between updates.
        
    Example:
        >>> tracker = ProgressTracker(show_eta=True, show_memory=True)
        >>> tracker.init(total=1000, desc="Processing features")
        >>> for i, feature in enumerate(features):
        ...     process(feature)
        ...     tracker.update()
        >>> tracker.finish()
    """
    
    def __init__(
        self,
        show_progress: bool = True,
        show_eta: bool = True,
        show_memory: bool = False,
        update_interval: float = 0.1,
        use_tqdm: bool = False
    ):
        """
        Initialize the progress tracker.
        
        Args:
            show_progress: Whether to display progress.
            show_eta: Whether to show ETA.
            show_memory: Whether to show memory usage.
            update_interval: Minimum time between updates (seconds).
            use_tqdm: Whether to use tqdm for progress display.
        """
        self.show_progress = show_progress
        self.show_eta = show_eta
        self.show_memory = show_memory
        self.update_interval = update_interval
        self.use_tqdm = use_tqdm
        
        self.state = ProgressState()
        self.progress_bar: Optional[ProgressBar] = None
        self.tqdm_bar = None
        self.callbacks: list = []
        self.last_render = 0.0
        
        # Try to import tqdm
        if self.use_tqdm:
            try:
                from tqdm import tqdm
                self.tqdm_available = True
            except ImportError:
                self.tqdm_available = False
                self.use_tqdm = False
        else:
            self.tqdm_available = False
    
    def init(
        self,
        total: int,
        description: str = "Processing",
        initial: int = 0
    ) -> None:
        """
        Initialize progress tracking.
        
        Args:
            total: Total number of items.
            description: Description of the task.
            initial: Initial count.
        """
        self.state = ProgressState(
            total=total,
            current=initial,
            description=description,
            start_time=time.time(),
            last_update=time.time()
        )
        
        if self.show_progress:
            if self.use_tqdm and self.tqdm_available:
                from tqdm import tqdm
                self.tqdm_bar = tqdm(
                    total=total,
                    desc=description,
                    initial=initial,
                    unit="items"
                )
            else:
                self.progress_bar = ProgressBar(
                    total=total,
                    description=description,
                    show_eta=self.show_eta,
                    show_memory=self.show_memory
                )
    
    def update(self, n: int = 1) -> None:
        """
        Update progress.
        
        Args:
            n: Number of items completed.
        """
        self.state.current += n
        self.state.last_update = time.time()
        
        # Check if we should render
        if time.time() - self.last_render >= self.update_interval:
            self.render()
            self.last_render = time.time()
        
        # Call callbacks
        for callback in self.callbacks:
            try:
                callback(self.state)
            except Exception:
                pass
    
    def render(self) -> None:
        """Render current progress."""
        if self.tqdm_bar:
            self.tqdm_bar.update(0)  # Trigger refresh
        elif self.progress_bar:
            self.progress_bar.render()
    
    def set_description(self, description: str) -> None:
        """Update progress description."""
        self.state.description = description
        if self.tqdm_bar:
            self.tqdm_bar.set_description(description)
        elif self.progress_bar:
            self.progress_bar.description = description
    
    def add_callback(self, callback: Callable[[ProgressState], None]) -> None:
        """
        Add progress callback.
        
        Args:
            callback: Function to call on each update.
        """
        self.callbacks.append(callback)
    
    def finish(self) -> None:
        """Mark progress as complete."""
        self.state.current = self.state.total
        self.state.completed = True
        self.render()
        
        if self.tqdm_bar:
            self.tqdm_bar.close()
            self.tqdm_bar = None
        elif self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None
    
    def __enter__(self) -> 'ProgressTracker':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if not self.state.completed:
            self.finish()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status as dictionary."""
        return {
            "description": self.state.description,
            "current": self.state.current,
            "total": self.state.total,
            "percentage": self.state.percentage,
            "elapsed": self.state.format_elapsed(),
            "eta": self.state.format_eta(),
            "rate": f"{self.state.rate:.2f} items/s",
            "completed": self.state.completed,
            "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024 if self.show_memory else None
        }
    
    def print_status(self) -> None:
        """Print current status to console."""
        status = self.get_status()
        print(
            f"{status['description']}: {status['current']}/{status['total']} "
            f"({status['percentage']:.1f}%) - "
            f"Elapsed: {status['elapsed']} - "
            f"ETA: {status['eta']} - "
            f"Rate: {status['rate']}"
        )


class MultiProgressTracker:
    """Track progress for multiple concurrent tasks."""
    
    def __init__(self, task_names: list):
        """
        Initialize multi-progress tracker.
        
        Args:
            task_names: List of task names.
        """
        self.tasks = {name: ProgressTracker() for name in task_names}
        self.task_order = task_names
    
    def init_task(
        self,
        task_name: str,
        total: int,
        description: Optional[str] = None
    ) -> None:
        """Initialize a specific task."""
        if task_name in self.tasks:
            self.tasks[task_name].init(
                total=total,
                description=description or task_name
            )
    
    def update_task(self, task_name: str, n: int = 1) -> None:
        """Update a specific task."""
        if task_name in self.tasks:
            self.tasks[task_name].update(n)
    
    def finish_task(self, task_name: str) -> None:
        """Finish a specific task."""
        if task_name in self.tasks:
            self.tasks[task_name].finish()
    
    def print_all_status(self) -> None:
        """Print status of all tasks."""
        print("\n" + "=" * 50)
        for task_name in self.task_order:
            tracker = self.tasks[task_name]
            status = tracker.get_status()
            bar = self._make_mini_bar(status['percentage'])
            print(f"{task_name}: {bar} {status['percentage']:.1f}%")
        print("=" * 50 + "\n")
    
    def _make_mini_bar(self, percentage: float, width: int = 20) -> str:
        """Create a mini progress bar."""
        filled = int(width * percentage / 100)
        return '[' + '█' * filled + '░' * (width - filled) + ']'


class MemoryMonitor:
    """Monitor memory usage during processing."""
    
    def __init__(self, warning_threshold_mb: float = 1024, critical_threshold_mb: float = 2048):
        """
        Initialize memory monitor.
        
        Args:
            warning_threshold_mb: Warning threshold in MB.
            critical_threshold_mb: Critical threshold in MB.
        """
        self.warning_threshold = warning_threshold_mb
        self.critical_threshold = critical_threshold_mb
        self.baseline_memory = None
        self.peak_memory = 0
    
    def start(self) -> float:
        """Start monitoring and return baseline memory."""
        self.baseline_memory = self.get_memory_mb()
        self.peak_memory = self.baseline_memory
        return self.baseline_memory
    
    def check(self) -> Dict[str, Any]:
        """Check current memory status."""
        current = self.get_memory_mb()
        self.peak_memory = max(self.peak_memory, current)
        
        delta = current - (self.baseline_memory or 0)
        
        status = "normal"
        if current > self.critical_threshold:
            status = "critical"
        elif current > self.warning_threshold:
            status = "warning"
        
        return {
            "current_mb": current,
            "baseline_mb": self.baseline_memory,
            "delta_mb": delta,
            "peak_mb": self.peak_memory,
            "status": status
        }
    
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_system_memory(self) -> Dict[str, Any]:
        """Get system memory information."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / 1024 / 1024 / 1024,
            "available_gb": memory.available / 1024 / 1024 / 1024,
            "used_percent": memory.percent
        }


# Convenience functions
def track_progress(iterable, total: Optional[int] = None, description: str = "Processing"):
    """Convenience function to track progress over an iterable."""
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = 0
    
    tracker = ProgressTracker()
    tracker.init(total=total, description=description)
    
    for item in iterable:
        yield item
        tracker.update()
    
    tracker.finish()


def with_progress(func: Callable, total: int, description: str = "Processing"):
    """Decorator to add progress tracking to a function."""
    def wrapper(*args, **kwargs):
        tracker = ProgressTracker()
        tracker.init(total=total, description=description)
        
        # Add tracker to kwargs
        kwargs['progress_tracker'] = tracker
        
        try:
            result = func(*args, **kwargs)
            tracker.finish()
            return result
        except Exception as e:
            tracker.finish()
            raise
    
    return wrapper
