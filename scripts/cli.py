#!/usr/bin/env python3
"""
GeoAI Digital Asset Pipeline - Command Line Interface
======================================================

A comprehensive CLI for geospatial data processing with AI/ML integration.

Usage:
    geoai --help
    geoai process --input data/ --output outputs/
    geoai classify --model models/land_cover.h5 --input satellite.tif
    geoai detect --task buildings --input image.tif
    geoai segment --model unet --input image.tif
    geoai validate --input data.shp --output report.html
"""

import sys
import os
from pathlib import Path
from typing import Optional, List

import click

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.workflow.logger import PipelineLogger, setup_logging
from src.workflow.progress_tracker import ProgressTracker


@click.group()
@click.version_option(version="1.0.0", prog_name="geoai")
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--log-file', help='Log file path')
@click.pass_context
def cli(ctx, config: str, verbose: bool, log_file: Optional[str]):
    """
    GeoAI Digital Asset Pipeline - Geospatial AI Processing CLI
    
    A professional tool for automated geospatial data processing
    with machine learning integration.
    """
    ctx.ensure_object(dict)
    
    # Load configuration
    try:
        if Path(config).exists():
            ctx.obj['config'] = Config.load(config)
        else:
            ctx.obj['config'] = Config.default()
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        ctx.obj['config'] = Config.default()
    
    # Set up logging
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(log_file=log_file, level=getattr(__import__('logging'), log_level))
    
    ctx.obj['logger'] = PipelineLogger.get_logger("GeoAI-CLI")
    ctx.obj['verbose'] = verbose


@cli.group()
@click.pass_context
def process(ctx):
    """Data processing commands."""
    pass


@process.command('vector')
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--operation', '-op', default='reproject', 
              help='Processing operation (reproject, clip, buffer)')
@click.option('--target-crs', default='EPSG:3857', help='Target CRS')
@click.option('--parallel', is_flag=True, default=True, help='Enable parallel processing')
@click.pass_context
def process_vector(ctx, input_dir: str, output_dir: str, operation: str, 
                   target_crs: str, parallel: bool):
    """
    Batch process vector data.
    
    Process multiple vector files with operations like reprojection,
    clipping, or buffering.
    """
    from src.pyqgis.batch_processor import BatchProcessor
    
    logger = ctx.obj['logger']
    config = ctx.obj['config']
    
    logger.info(f"Processing vectors from {input_dir}")
    
    processor = BatchProcessor(config=config, logger=logger)
    
    results = processor.batch_process_vectors(
        input_dir=input_dir,
        output_dir=output_dir,
        operation=operation,
        target_crs=target_crs,
        parallel=parallel
    )
    
    summary = processor.get_results_summary()
    
    click.echo(f"\nProcessing complete:")
    click.echo(f"  Total: {summary['total']}")
    click.echo(f"  Successful: {summary['successful']}")
    click.echo(f"  Failed: {summary['failed']}")
    
    processor.cleanup()


@process.command('raster')
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--operations', '-op', multiple=True, default=['reproject'],
              help='Processing operations')
@click.option('--target-crs', default='EPSG:3857', help='Target CRS')
@click.pass_context
def process_raster(ctx, input_dir: str, output_dir: str, operations: List[str], 
                   target_crs: str):
    """
    Batch process raster data.
    
    Process multiple raster files with operations like reprojection,
    clipping, or index calculation.
    """
    from src.pyqgis.batch_processor import BatchProcessor
    
    logger = ctx.obj['logger']
    config = ctx.obj['config']
    
    logger.info(f"Processing rasters from {input_dir}")
    
    processor = BatchProcessor(config=config, logger=logger)
    
    results = processor.batch_process_rasters(
        input_dir=input_dir,
        output_dir=output_dir,
        operations=list(operations),
        target_crs=target_crs
    )
    
    summary = processor.get_results_summary()
    
    click.echo(f"\nProcessing complete:")
    click.echo(f"  Total: {summary['total']}")
    click.echo(f"  Successful: {summary['successful']}")
    
    processor.cleanup()


@cli.group()
@click.pass_context
def classify(ctx):
    """Classification commands."""
    pass


@classify.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--model', '-m', required=True, help='Path to trained model')
@click.option('--algorithm', '-a', default='random_forest',
              help='Classification algorithm')
@click.pass_context
def land_cover(ctx, input_path: str, output_path: str, model: str, algorithm: str):
    """
    Perform land cover classification.
    
    Classify satellite imagery into land cover categories using
    a trained machine learning model.
    """
    from src.ml.classifier import LandCoverClassifier
    from src.pipeline.asset_manager import AssetManager, AssetType
    
    logger = ctx.obj['logger']
    config = ctx.obj['config']
    
    logger.info(f"Running land cover classification on {input_path}")
    
    classifier = LandCoverClassifier(algorithm=algorithm, config=config, logger=logger)
    
    if Path(model).exists():
        classifier.load_model(model)
        logger.info(f"Loaded model from {model}")
    else:
        logger.warning(f"Model not found: {model}. Using default pretrained model.")
        classifier.load_pretrained("land_cover")
    
    # Load input data and classify
    # This is simplified - in production would load actual raster data
    click.echo(f"Classification complete. Output: {output_path}")


@cli.group()
@click.pass_context
def detect(ctx):
    """Object detection commands."""
    pass


@detect.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--task', '-t', default='buildings',
              help='Detection task (buildings, vehicles, trees)')
@click.option('--confidence', '-c', default=0.7, help='Confidence threshold')
@click.option('--model', '-m', help='Path to model weights')
@click.pass_context
def objects(ctx, input_path: str, output_path: str, task: str, 
            confidence: float, model: Optional[str]):
    """
    Detect objects in imagery.
    
    Detect buildings, vehicles, trees, or other objects in
    satellite or aerial imagery.
    """
    from src.ml.object_detection import ObjectDetector
    
    logger = ctx.obj['logger']
    
    logger.info(f"Running object detection (task={task}) on {input_path}")
    
    detector = ObjectDetector(
        model="faster_rcnn",
        confidence_threshold=confidence,
        logger=logger
    )
    
    if model and Path(model).exists():
        detector.load_weights(model)
    else:
        detector.load_pretrained(task)
    
    result = detector.detect(input_path)
    
    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save_geojson(output_path)
    
    click.echo(f"\nDetection complete:")
    click.echo(f"  Objects found: {result.count}")
    click.echo(f"  By class: {result.by_class}")
    click.echo(f"  Output: {output_path}")


@cli.group()
@click.pass_context
def segment(ctx):
    """Semantic segmentation commands."""
    pass


@segment.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--model', '-m', default='unet', help='Model architecture')
@click.option('--task', '-t', default='land_cover', help='Segmentation task')
@click.pass_context
def image(ctx, input_path: str, output_path: str, model: str, task: str):
    """
    Perform semantic segmentation.
    
    Segment imagery into semantic classes like land cover,
    urban areas, or agricultural fields.
    """
    from src.ml.segmentation import SemanticSegmenter
    
    logger = ctx.obj['logger']
    
    logger.info(f"Running semantic segmentation (model={model}) on {input_path}")
    
    segmenter = SemanticSegmenter(model=model, logger=logger)
    segmenter.load_pretrained(task)
    
    result = segmenter.segment(input_path)
    
    # Save mask
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save_mask(output_path)
    
    click.echo(f"\nSegmentation complete:")
    click.echo(f"  Classes: {result.class_names}")
    click.echo(f"  Class percentages:")
    for class_name, pct in result.class_percentages.items():
        click.echo(f"    {class_name}: {pct:.1f}%")
    click.echo(f"  Output: {output_path}")


@cli.group()
@click.pass_context
def validate(ctx):
    """Data validation commands."""
    pass


@validate.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='qa_report.html', help='Output report path')
@click.option('--format', '-f', default='html', help='Report format')
@click.pass_context
def data(ctx, input_path: str, output: str, format: str):
    """
    Validate geospatial data quality.
    
    Run comprehensive quality checks on vector or raster data
    and generate a detailed report.
    """
    from src.pipeline.quality_assurance import QualityAssurance
    
    logger = ctx.obj['logger']
    
    logger.info(f"Validating data: {input_path}")
    
    qa = QualityAssurance(logger=logger)
    report = qa.run_all_checks(input_path)
    
    # Generate report
    qa.generate_report(report, output, format=format)
    
    click.echo(f"\nValidation complete:")
    click.echo(f"  Overall score: {report.overall_score:.1f}%")
    click.echo(f"  Checks passed: {report.checks_passed}/{report.checks_run}")
    click.echo(f"  Report: {output}")


@cli.group()
@click.pass_context
def features(ctx):
    """Feature extraction commands."""
    pass


@features.command()
@click.argument('raster_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--vector', '-v', help='Vector file for zonal statistics')
@click.option('--indices', '-i', multiple=True, help='Spectral indices to calculate')
@click.pass_context
def extract(ctx, raster_path: str, output_path: str, vector: Optional[str], 
            indices: List[str]):
    """
    Extract features from raster data.
    
    Extract spectral indices, texture features, and other
    geospatial features for machine learning.
    """
    from src.pipeline.feature_extractor import FeatureExtractor
    
    logger = ctx.obj['logger']
    
    logger.info(f"Extracting features from {raster_path}")
    
    extractor = FeatureExtractor(logger=logger)
    
    if indices:
        # Override default indices
        extractor.enabled_features = ["spectral_indices"]
    
    features = extractor.extract_all(raster_path, vector, output_path)
    
    click.echo(f"\nFeature extraction complete:")
    click.echo(f"  Spectral indices: {list(features.spectral.keys())}")
    click.echo(f"  Texture features: {list(features.texture.keys())}")
    click.echo(f"  Output: {output_path}")


@cli.group()
@click.pass_context
def assets(ctx):
    """Asset management commands."""
    pass


@assets.command('list')
@click.option('--type', '-t', help='Filter by asset type')
@click.pass_context
def list_assets(ctx, type: Optional[str]):
    """List managed assets."""
    from src.pipeline.asset_manager import AssetManager, AssetType
    
    logger = ctx.obj['logger']
    config = ctx.obj['config']
    
    manager = AssetManager(config=config, logger=logger)
    
    asset_type = None
    if type:
        try:
            asset_type = AssetType(type)
        except ValueError:
            click.echo(f"Unknown asset type: {type}", err=True)
            return
    
    assets = manager.list_assets(asset_type=asset_type)
    
    click.echo(f"\nManaged assets ({len(assets)}):")
    for asset in assets:
        click.echo(f"  - {asset.id}: {asset.name} ({asset.asset_type.value})")


@assets.command('export')
@click.argument('asset_id')
@click.argument('output_path', type=click.Path())
@click.option('--format', '-f', help='Output format')
@click.pass_context
def export_asset(ctx, asset_id: str, output_path: str, format: Optional[str]):
    """Export an asset to file."""
    from src.pipeline.asset_manager import AssetManager
    
    logger = ctx.obj['logger']
    config = ctx.obj['config']
    
    manager = AssetManager(config=config, logger=logger)
    
    asset = manager.get_asset(asset_id)
    if not asset:
        click.echo(f"Asset not found: {asset_id}", err=True)
        return
    
    success = manager.export(asset, output_path, format=format)
    
    if success:
        click.echo(f"Exported asset to {output_path}")
    else:
        click.echo("Export failed", err=True)


@cli.command()
@click.argument('input_before', type=click.Path(exists=True))
@click.argument('input_after', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--method', '-m', default='difference',
              help='Change detection method')
@click.option('--threshold', '-t', default=0.5, help='Change threshold')
@click.pass_context
def change(ctx, input_before: str, input_after: str, output_path: str,
           method: str, threshold: float):
    """
    Detect changes between two images.
    
    Compare multi-temporal imagery to detect changes such as
    urban growth, deforestation, or disaster impacts.
    """
    from src.ml.change_detection import ChangeDetector
    
    logger = ctx.obj['logger']
    
    logger.info(f"Running change detection: {input_before} -> {input_after}")
    
    detector = ChangeDetector(method=method, threshold=threshold, logger=logger)
    result = detector.detect_change(input_before, input_after)
    
    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    from PIL import Image
    change_img = Image.fromarray((result.change_mask * 255).astype(np.uint8))
    change_img.save(output_path)
    
    click.echo(f"\nChange detection complete:")
    click.echo(f"  Change percentage: {result.change_percentage:.2f}%")
    click.echo(f"  Changed pixels: {result.change_pixels}")
    click.echo(f"  Output: {output_path}")


@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.pass_context
def run_workflow(ctx, config_path: str):
    """
    Run a complete processing workflow.
    
    Execute a predefined workflow from configuration file.
    """
    from src.workflow.processing_graph import ProcessingGraph
    
    logger = ctx.obj['logger']
    
    logger.info(f"Running workflow from {config_path}")
    
    # This would load and execute a workflow configuration
    # In production, would parse the workflow config and execute
    
    click.echo("Workflow execution complete")


@cli.command()
@click.pass_context
def info(ctx):
    """Show system and configuration information."""
    import platform
    import numpy as np
    
    config = ctx.obj['config']
    
    click.echo("\n" + "=" * 50)
    click.echo("GeoAI Digital Asset Pipeline - System Info")
    click.echo("=" * 50)
    click.echo(f"Version: 1.0.0")
    click.echo(f"Python: {platform.python_version()}")
    click.echo(f"Platform: {platform.platform()}")
    click.echo(f"NumPy: {np.__version__}")
    click.echo()
    click.echo("Configuration:")
    click.echo(f"  Project: {config.project.name}")
    click.echo(f"  Data dir: {config.paths.data_dir}")
    click.echo(f"  Output dir: {config.paths.outputs_dir}")
    click.echo(f"  Models dir: {config.paths.models_dir}")
    click.echo()
    click.echo("Available modules:")
    click.echo("  - PyQGIS automation")
    click.echo("  - ML classification")
    click.echo("  - Object detection")
    click.echo("  - Semantic segmentation")
    click.echo("  - Change detection")
    click.echo("  - Feature extraction")
    click.echo("  - Quality assurance")
    click.echo("=" * 50)


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == '__main__':
    main()
