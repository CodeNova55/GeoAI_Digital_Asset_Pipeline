"""
Processing Graph Module
=======================

Provides a graph-based processing workflow system for complex
geospatial data pipelines with dependency management and error handling.

Example:
    >>> graph = ProcessingGraph()
    >>> graph.add_node("load", load_data)
    >>> graph.add_node("process", process_data, depends_on=["load"])
    >>> graph.add_node("export", export_data, depends_on=["process"])
    >>> results = graph.execute()
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import json
import traceback

from .logger import PipelineLogger
from .progress_tracker import ProgressTracker


class NodeStatus(Enum):
    """Status of a processing node."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeResult:
    """Result of a node execution."""
    
    node_id: str
    status: NodeStatus
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "output_type": type(self.output).__name__ if self.output is not None else None,
            "error": self.error,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp
        }


@dataclass
class ProcessingNode:
    """A node in the processing graph."""
    
    id: str
    func: Callable
    depends_on: List[str] = field(default_factory=list)
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    
    # Execution state
    status: NodeStatus = NodeStatus.PENDING
    result: Optional[NodeResult] = None
    cached_output: Any = None
    use_cache: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "depends_on": self.depends_on,
            "parameters": self.parameters,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }


class ProcessingGraph:
    """
    Graph-based processing workflow system.
    
    This class provides a flexible system for defining and executing
    complex geospatial data processing workflows with dependency
    management, error handling, and caching.
    
    Attributes:
        nodes: Dictionary of processing nodes.
        logger: Logger instance.
        results: Dictionary of node execution results.
        
    Example:
        >>> graph = ProcessingGraph()
        >>> graph.add_node(
        ...     "load_data",
        ...     load_function,
        ...     parameters={"path": "data.shp"}
        ... )
        >>> graph.add_node(
        ...     "process",
        ...     process_function,
        ...     depends_on=["load_data"]
        ... )
        >>> results = graph.execute()
    """
    
    def __init__(
        self,
        logger: Optional[PipelineLogger] = None,
        cache_dir: Optional[str] = None,
        max_parallel: int = 1
    ):
        """
        Initialize the processing graph.
        
        Args:
            logger: Logger instance.
            cache_dir: Directory for caching results.
            max_parallel: Maximum parallel executions.
        """
        self.nodes: Dict[str, ProcessingNode] = {}
        self.results: Dict[str, NodeResult] = {}
        self.logger = logger or PipelineLogger.get_logger("ProcessingGraph")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_parallel = max_parallel
        
        self.progress = ProgressTracker()
        self.execution_order: List[str] = []
        
        self.logger.info("ProcessingGraph initialized")
    
    def add_node(
        self,
        node_id: str,
        func: Callable,
        depends_on: Optional[List[str]] = None,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        use_cache: bool = True
    ) -> 'ProcessingGraph':
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique node identifier.
            func: Function to execute.
            depends_on: List of node IDs this node depends on.
            description: Node description.
            parameters: Parameters to pass to the function.
            max_retries: Maximum retry attempts on failure.
            use_cache: Whether to cache results.
            
        Returns:
            Self for method chaining.
        """
        if node_id in self.nodes:
            raise ValueError(f"Node '{node_id}' already exists")
        
        # Validate dependencies
        if depends_on:
            for dep in depends_on:
                if dep not in self.nodes:
                    raise ValueError(f"Dependency '{dep}' does not exist")
        
        self.nodes[node_id] = ProcessingNode(
            id=node_id,
            func=func,
            depends_on=depends_on or [],
            description=description,
            parameters=parameters or {},
            max_retries=max_retries,
            use_cache=use_cache
        )
        
        self.logger.debug(f"Added node: {node_id}")
        
        return self
    
    def add_edge(self, from_node: str, to_node: str) -> 'ProcessingGraph':
        """
        Add an edge (dependency) between nodes.
        
        Args:
            from_node: Source node ID.
            to_node: Target node ID.
            
        Returns:
            Self for method chaining.
        """
        if from_node not in self.nodes:
            raise ValueError(f"Node '{from_node}' does not exist")
        if to_node not in self.nodes:
            raise ValueError(f"Node '{to_node}' does not exist")
        
        if from_node not in self.nodes[to_node].depends_on:
            self.nodes[to_node].depends_on.append(from_node)
        
        return self
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the graph.
        
        Args:
            node_id: Node to remove.
            
        Returns:
            True if removed successfully.
        """
        if node_id not in self.nodes:
            return False
        
        # Check if other nodes depend on this one
        for node in self.nodes.values():
            if node_id in node.depends_on:
                raise ValueError(f"Cannot remove node '{node_id}': other nodes depend on it")
        
        del self.nodes[node_id]
        return True
    
    def get_execution_order(self) -> List[str]:
        """
        Get topologically sorted execution order.
        
        Returns:
            List of node IDs in execution order.
        """
        # Kahn's algorithm for topological sorting
        in_degree = {node_id: len(node.depends_on) for node_id, node in self.nodes.items()}
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            node_id = queue.pop(0)
            order.append(node_id)
            
            for other_id, node in self.nodes.items():
                if node_id in node.depends_on:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)
        
        if len(order) != len(self.nodes):
            raise ValueError("Graph contains a cycle - cannot determine execution order")
        
        self.execution_order = order
        return order
    
    def execute(
        self,
        node_ids: Optional[List[str]] = None,
        stop_on_error: bool = False,
        return_all_outputs: bool = False
    ) -> Dict[str, NodeResult]:
        """
        Execute the processing graph.
        
        Args:
            node_ids: Specific nodes to execute. None executes all.
            stop_on_error: Whether to stop on first error.
            return_all_outputs: Whether to return all outputs.
            
        Returns:
            Dictionary of node results.
        """
        import time
        
        # Determine nodes to execute
        if node_ids:
            nodes_to_execute = node_ids
        else:
            nodes_to_execute = self.get_execution_order()
        
        self.logger.info(f"Executing {len(nodes_to_execute)} nodes")
        self.progress.init(total=len(nodes_to_execute), description="Processing graph")
        
        executed = set()
        skipped = set()
        
        for node_id in nodes_to_execute:
            node = self.nodes[node_id]
            
            # Check dependencies
            deps_satisfied = all(
                dep in executed or dep in skipped
                for dep in node.depends_on
            )
            
            if not deps_satisfied:
                # Check if any dependency failed
                deps_failed = any(
                    dep in self.results and self.results[dep].status == NodeStatus.FAILED
                    for dep in node.depends_on
                )
                
                if deps_failed:
                    node.status = NodeStatus.SKIPPED
                    node.result = NodeResult(
                        node_id=node_id,
                        status=NodeStatus.SKIPPED,
                        error="Dependency failed"
                    )
                    self.results[node_id] = node.result
                    skipped.add(node_id)
                    self.progress.update()
                    continue
            
            # Execute node
            result = self._execute_node(node)
            self.results[node_id] = result
            
            if result.status == NodeStatus.COMPLETED:
                executed.add(node_id)
            elif result.status == NodeStatus.FAILED:
                if stop_on_error:
                    self.logger.error(f"Stopping execution due to error in node '{node_id}'")
                    break
            
            self.progress.update()
        
        self.progress.finish()
        
        # Summary
        completed = sum(1 for r in self.results.values() if r.status == NodeStatus.COMPLETED)
        failed = sum(1 for r in self.results.values() if r.status == NodeStatus.FAILED)
        skipped_count = sum(1 for r in self.results.values() if r.status == NodeStatus.SKIPPED)
        
        self.logger.info(
            f"Execution complete: {completed} completed, "
            f"{failed} failed, {skipped_count} skipped"
        )
        
        if return_all_outputs:
            return {
                node_id: result.output
                for node_id, result in self.results.items()
                if result.output is not None
            }
        
        return self.results
    
    def _execute_node(self, node: ProcessingNode) -> NodeResult:
        """Execute a single node."""
        import time
        start_time = time.time()
        
        self.logger.info(f"Executing node: {node.id}")
        node.status = NodeStatus.RUNNING
        
        # Check cache
        if node.use_cache and self.cache_dir:
            cached = self._load_from_cache(node.id)
            if cached is not None:
                self.logger.info(f"Using cached result for node: {node.id}")
                node.cached_output = cached
                node.status = NodeStatus.COMPLETED
                return NodeResult(
                    node_id=node.id,
                    status=NodeStatus.COMPLETED,
                    output=cached,
                    execution_time=0.0
                )
        
        # Gather inputs from dependencies
        inputs = {}
        for dep_id in node.depends_on:
            if dep_id in self.results and self.results[dep_id].output is not None:
                inputs[dep_id] = self.results[dep_id].output
        
        # Execute with retries
        last_error = None
        for attempt in range(node.max_retries):
            node.retry_count = attempt
            
            try:
                # Call function with parameters and dependency outputs
                all_params = {**node.parameters, **inputs}
                output = node.func(**all_params)
                
                # Cache result
                if node.use_cache and self.cache_dir and output is not None:
                    self._save_to_cache(node.id, output)
                
                execution_time = time.time() - start_time
                
                node.status = NodeStatus.COMPLETED
                result = NodeResult(
                    node_id=node.id,
                    status=NodeStatus.COMPLETED,
                    output=output,
                    execution_time=execution_time
                )
                
                self.logger.info(
                    f"Node '{node.id}' completed in {execution_time:.2f}s"
                )
                
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Node '{node.id}' attempt {attempt + 1} failed: {str(e)}"
                )
                
                if attempt < node.max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        # All retries failed
        execution_time = time.time() - start_time
        error_msg = f"{str(last_error)}\n{traceback.format_exc()}"
        
        node.status = NodeStatus.FAILED
        result = NodeResult(
            node_id=node.id,
            status=NodeStatus.FAILED,
            error=error_msg,
            execution_time=execution_time
        )
        
        self.logger.error(f"Node '{node.id}' failed after {node.max_retries} attempts")
        
        return result
    
    def _save_to_cache(self, node_id: str, output: Any) -> None:
        """Save node output to cache."""
        if not self.cache_dir:
            return
        
        try:
            cache_file = self.cache_dir / f"{node_id}.json"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Try to serialize
            with open(cache_file, 'w') as f:
                json.dump(self._serialize_output(output), f, indent=2, default=str)
                
        except Exception as e:
            self.logger.warning(f"Failed to cache node '{node_id}': {e}")
    
    def _load_from_cache(self, node_id: str) -> Optional[Any]:
        """Load node output from cache."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{node_id}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache for node '{node_id}': {e}")
        
        return None
    
    def _serialize_output(self, output: Any) -> Any:
        """Serialize output for caching."""
        if isinstance(output, dict):
            return {k: self._serialize_output(v) for k, v in output.items()}
        elif isinstance(output, list):
            return [self._serialize_output(item) for item in output]
        elif hasattr(output, 'to_dict'):
            return output.to_dict()
        elif hasattr(output, '__dict__'):
            return output.__dict__
        else:
            return output
    
    def get_status(self) -> Dict[str, Any]:
        """Get graph execution status."""
        return {
            "total_nodes": len(self.nodes),
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "results": {node_id: result.to_dict() for node_id, result in self.results.items()},
            "execution_order": self.execution_order
        }
    
    def visualize(self, output_path: str) -> bool:
        """
        Generate graph visualization (DOT format).
        
        Args:
            output_path: Path for output DOT file.
            
        Returns:
            True if generation successful.
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            lines = ["digraph ProcessingGraph {"]
            lines.append("  rankdir=LR;")
            lines.append("  node [shape=box];")
            
            for node_id, node in self.nodes.items():
                color = {
                    NodeStatus.PENDING: "gray",
                    NodeStatus.RUNNING: "yellow",
                    NodeStatus.COMPLETED: "green",
                    NodeStatus.FAILED: "red",
                    NodeStatus.SKIPPED: "orange"
                }.get(node.status, "white")
                
                label = f"{node_id}\\n{node.description}"
                lines.append(f'  "{node_id}" [label="{label}", style=filled, fillcolor={color}];')
                
                for dep in node.depends_on:
                    lines.append(f'  "{dep}" -> "{node_id}";')
            
            lines.append("}")
            
            with open(output_path, 'w') as f:
                f.write('\n'.join(lines))
            
            self.logger.info(f"Graph visualization saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to visualize graph: {e}")
            return False
    
    def save(self, path: str) -> bool:
        """Save graph configuration to file."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            config = {
                "nodes": {
                    node_id: {
                        "description": node.description,
                        "depends_on": node.depends_on,
                        "parameters": node.parameters,
                        "max_retries": node.max_retries
                    }
                    for node_id, node in self.nodes.items()
                }
            }
            
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save graph: {e}")
            return False
    
    @classmethod
    def load(cls, path: str, functions: Dict[str, Callable]) -> 'ProcessingGraph':
        """
        Load graph configuration from file.
        
        Args:
            path: Path to configuration file.
            functions: Dictionary of available functions.
            
        Returns:
            ProcessingGraph instance.
        """
        with open(path, 'r') as f:
            config = json.load(f)
        
        graph = cls()
        
        for node_id, node_config in config["nodes"].items():
            func_name = node_config.get("function", node_id)
            func = functions.get(func_name)
            
            if func is None:
                raise ValueError(f"Function '{func_name}' not found for node '{node_id}'")
            
            graph.add_node(
                node_id=node_id,
                func=func,
                depends_on=node_config.get("depends_on", []),
                description=node_config.get("description", ""),
                parameters=node_config.get("parameters", {}),
                max_retries=node_config.get("max_retries", 3)
            )
        
        return graph


# Convenience decorators for defining graph nodes
def node(
    node_id: str,
    depends_on: Optional[List[str]] = None,
    **parameters
):
    """Decorator to mark a function as a graph node."""
    def decorator(func):
        func._node_config = {
            "id": node_id,
            "depends_on": depends_on or [],
            "parameters": parameters
        }
        return func
    return decorator


def auto_graph(*functions):
    """
    Automatically create a graph from decorated functions.
    
    Args:
        *functions: Functions decorated with @node.
        
    Returns:
        ProcessingGraph instance.
    """
    graph = ProcessingGraph()
    
    for func in functions:
        if hasattr(func, '_node_config'):
            config = func._node_config
            graph.add_node(
                node_id=config["id"],
                func=func,
                depends_on=config["depends_on"],
                **config["parameters"]
            )
    
    return graph
