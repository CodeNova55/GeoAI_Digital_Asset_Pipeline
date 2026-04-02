#!/usr/bin/env python3
"""
GeoAI Digital Asset Pipeline - Main Entry Point
================================================

This is the main entry point for the GeoAI pipeline.
It provides a unified interface for running all pipeline components.

Usage:
    python -m src.main --help
    python -m src.main run --config config.yaml
    python -m src.main demo
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import Config
from workflow.logger import PipelineLogger, setup_logging
from workflow.progress_tracker import ProgressTracker


def run_demo():
    """Run a demonstration of the pipeline capabilities."""
    print("\n" + "=" * 60)
    print("GeoAI Digital Asset Pipeline - Demo")
    print("=" * 60 + "\n")
    
    logger = PipelineLogger.get_logger("Demo")
    config = Config.default()
    progress = ProgressTracker()
    
    # Demo 1: Configuration
    print("1. Configuration System")
    print("-" * 40)
    print(f"   Project: {config.project.name}")
    print(f"   Version: {config.project.version}")
    print(f"   Data directory: {config.paths.data_dir}")
    print(f"   Output directory: {config.paths.outputs_dir}")
    print()
    
    # Demo 2: Feature Extraction
    print("2. Feature Extraction")
    print("-" * 40)
    from pipeline.feature_extractor import FeatureExtractor
    
    extractor = FeatureExtractor(config=config, logger=logger)
    
    # Create sample data for demonstration
    import numpy as np
    bands = {
        "red": np.ones((10, 10)) * 100,
        "nir": np.ones((10, 10)) * 200,
        "green": np.ones((10, 10)) * 150,
        "blue": np.ones((10, 10)) * 50,
        "swir1": np.ones((10, 10)) * 80,
    }
    
    indices = extractor.calculate_spectral_indices(bands)
    print(f"   Calculated indices: {list(indices.keys())}")
    print(f"   NDVI sample value: {indices['NDVI'][0, 0]:.4f}")
    print()
    
    # Demo 3: Machine Learning
    print("3. Machine Learning Classification")
    print("-" * 40)
    from ml.classifier import SpatialClassifier
    
    classifier = SpatialClassifier(algorithm="random_forest", n_estimators=10)
    
    # Create sample training data
    X = np.random.rand(50, 4)
    y = np.random.randint(0, 3, 50)
    
    result = classifier.train(X, y, validation_size=0.2)
    print(f"   Algorithm: {classifier.algorithm}")
    print(f"   Training accuracy: {result.training_accuracy:.3f}")
    print(f"   Validation accuracy: {result.validation_accuracy:.3f}")
    print()
    
    # Demo 4: Quality Assurance
    print("4. Quality Assurance")
    print("-" * 40)
    from pipeline.quality_assurance import QualityAssurance
    
    qa = QualityAssurance(logger=logger)
    print(f"   Enabled checks: {qa.enabled_checks}")
    print(f"   Auto-repair: {qa.auto_repair}")
    print()
    
    # Demo 5: Processing Graph
    print("5. Processing Workflow")
    print("-" * 40)
    from workflow.processing_graph import ProcessingGraph
    
    graph = ProcessingGraph(logger=logger)
    
    def step1():
        return {"data": "processed"}
    
    def step2(data):
        return {"result": f"completed with {data}"}
    
    graph.add_node("load", step1)
    graph.add_node("process", step2, depends_on=["load"])
    
    print(f"   Nodes in graph: {list(graph.nodes.keys())}")
    print(f"   Execution order: {graph.get_execution_order()}")
    print()
    
    print("=" * 60)
    print("Demo complete! All systems operational.")
    print("=" * 60 + "\n")


def run_pipeline(config_path: str):
    """Run the full pipeline with configuration."""
    logger = PipelineLogger.get_logger("Pipeline")
    
    # Load configuration
    if Path(config_path).exists():
        config = Config.load(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        config = Config.default()
        logger.info("Using default configuration")
    
    # Initialize progress tracker
    progress = ProgressTracker()
    
    logger.info("Starting GeoAI pipeline...")
    
    # Pipeline execution would go here
    # This is a placeholder for the actual pipeline logic
    
    logger.info("Pipeline execution complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GeoAI Digital Asset Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.main demo
    python -m src.main run --config config.yaml
    python -m src.main info
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demonstration")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run pipeline")
    run_parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    
    if args.command == "demo":
        run_demo()
    elif args.command == "run":
        run_pipeline(args.config)
    elif args.command == "info":
        from scripts.cli import info
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(info, [])
        print(result.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
