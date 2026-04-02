"""
Layer Styler Module
===================

Provides automated layer styling and export functionality for QGIS.
Supports rule-based styling, categorized styling, and export to multiple formats.

Example:
    >>> styler = LayerStyler()
    >>> styler.apply_land_cover_style(layer, style_path)
    >>> styler.export_layer(layer, output_path, format="geojson")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import logging

try:
    from qgis.core import (
        QgsVectorLayer,
        QgsRasterLayer,
        QgsSymbol,
        QgsFillSymbol,
        QgsLineSymbol,
        QgsMarkerSymbol,
        QgsSimpleFillSymbolLayer,
        QgsSimpleLineSymbolLayer,
        QgsSimpleMarkerSymbolLayer,
        QgsCategorizedSymbolRenderer,
        QgsGraduatedSymbolRenderer,
        QgsSingleSymbolRenderer,
        QgsRuleBasedRenderer,
        QgsRendererCategory,
        QgsRendererRange,
        QgsSymbolLayer,
        QgsPalLayerSettings,
        QgsVectorLayerSimpleLabeling,
        QgsProject,
        QgsMapSettings,
        QgsMapRendererJob,
        QgsMapLayerStyle,
        QgsRasterShader,
        QgsColorRampShader,
        QgsSingleBandPseudoColorRenderer,
    )
    from qgis.PyQt import QtGui
    from qgis.PyQt.QtCore import Qt
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False
    # Define stub types for when QGIS is not available
    QgsVectorLayer = type('QgsVectorLayer', (), {})
    QgsRasterLayer = type('QgsRasterLayer', (), {})
    QgsSymbol = type('QgsSymbol', (), {})
    QgsFillSymbol = type('QgsFillSymbol', (), {})
    QgsLineSymbol = type('QgsLineSymbol', (), {})
    QgsMarkerSymbol = type('QgsMarkerSymbol', (), {})
    QgsSimpleFillSymbolLayer = type('QgsSimpleFillSymbolLayer', (), {})
    QgsSimpleLineSymbolLayer = type('QgsSimpleLineSymbolLayer', (), {})
    QgsSimpleMarkerSymbolLayer = type('QgsSimpleMarkerSymbolLayer', (), {})
    QgsCategorizedSymbolRenderer = type('QgsCategorizedSymbolRenderer', (), {})
    QgsGraduatedSymbolRenderer = type('QgsGraduatedSymbolRenderer', (), {})
    QgsSingleSymbolRenderer = type('QgsSingleSymbolRenderer', (), {})
    QgsRuleBasedRenderer = type('QgsRuleBasedRenderer', (), {})
    QgsRendererCategory = type('QgsRendererCategory', (), {})
    QgsRendererRange = type('QgsRendererRange', (), {})
    QgsSymbolLayer = type('QgsSymbolLayer', (), {})
    QgsPalLayerSettings = type('QgsPalLayerSettings', (), {})
    QgsVectorLayerSimpleLabeling = type('QgsVectorLayerSimpleLabeling', (), {})
    QgsProject = type('QgsProject', (), {})
    QgsMapSettings = type('QgsMapSettings', (), {})
    QgsMapRendererJob = type('QgsMapRendererJob', (), {})
    QgsMapLayerStyle = type('QgsMapLayerStyle', (), {})
    QgsRasterShader = type('QgsRasterShader', (), {})
    QgsColorRampShader = type('QgsColorRampShader', (), {})
    QgsSingleBandPseudoColorRenderer = type('QgsSingleBandPseudoColorRenderer', (), {})
    
    class _MockQtGui:
        class QColor:
            def __init__(self, *args): pass
        class QTextFormat:
            def setFontPointSize(self, *args): pass
            def setForeground(self, *args): pass
    QtGui = _MockQtGui()
    
    class _MockQt:
        pass
    Qt = _MockQt()

from src.utils.config import Config
from src.workflow.logger import PipelineLogger


@dataclass
class StyleDefinition:
    """Container for style definition."""
    
    name: str
    fill_color: Optional[str] = None
    stroke_color: Optional[str] = None
    stroke_width: float = 1.0
    opacity: float = 1.0
    symbol_type: str = "fill"  # fill, line, marker
    symbol_size: float = 4.0
    label_field: Optional[str] = None
    rules: Optional[List[Dict[str, Any]]] = None


class LayerStyler:
    """
    Automated layer styling and export for QGIS layers.
    
    This class provides methods for applying styles to vector and raster
    layers, creating rule-based renderers, and exporting styled layers
    to various formats.
    
    Attributes:
        config: Configuration object with style parameters.
        logger: Logger instance for recording styling events.
        style_library: Dictionary of predefined styles.
        
    Example:
        >>> styler = LayerStyler()
        >>> styler.apply_categorized_style(
        ...     layer, 
        ...     field="land_use",
        ...     categories={"forest": "#228B22", "urban": "#808080"}
        ... )
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        logger: Optional[PipelineLogger] = None
    ):
        """
        Initialize the layer styler.
        
        Args:
            config: Configuration object. Uses default if None.
            logger: Logger instance. Creates new if None.
        """
        self.config = config or Config.default()
        self.logger = logger or PipelineLogger.get_logger("LayerStyler")
        self.style_library = self._load_style_library()
        
        self.logger.info("LayerStyler initialized")
    
    def _load_style_library(self) -> Dict[str, StyleDefinition]:
        """Load predefined style library from config."""
        styles = {}
        
        # Default land cover styles
        land_cover_styles = {
            "water": StyleDefinition(
                name="water", fill_color="#339AF0", stroke_color="#1C7ED6",
                opacity=0.8, symbol_type="fill"
            ),
            "vegetation": StyleDefinition(
                name="vegetation", fill_color="#51CF66", stroke_color="#2F9E44",
                opacity=0.6, symbol_type="fill"
            ),
            "buildings": StyleDefinition(
                name="buildings", fill_color="#FF6B6B", stroke_color="#C92A2A",
                opacity=0.7, symbol_type="fill"
            ),
            "roads": StyleDefinition(
                name="roads", fill_color=None, stroke_color="#495057",
                stroke_width=2.0, opacity=1.0, symbol_type="line"
            ),
            "bare_soil": StyleDefinition(
                name="bare_soil", fill_color="#E4C48E", stroke_color="#BFA76F",
                opacity=0.7, symbol_type="fill"
            ),
            "urban": StyleDefinition(
                name="urban", fill_color="#868E96", stroke_color="#495057",
                opacity=0.8, symbol_type="fill"
            ),
            "forest": StyleDefinition(
                name="forest", fill_color="#2F9E44", stroke_color="#1B7A3B",
                opacity=0.7, symbol_type="fill"
            ),
            "agriculture": StyleDefinition(
                name="agriculture", fill_color="#94D82D", stroke_color="#66A80F",
                opacity=0.6, symbol_type="fill"
            ),
        }
        
        styles.update(land_cover_styles)
        
        # Load additional styles from config
        if self.config and hasattr(self.config, "visualization"):
            vis_config = self.config.visualization
            # Could load custom styles from config here
            
        return styles
    
    def apply_style(
        self,
        layer: Union[QgsVectorLayer, QgsRasterLayer],
        style_def: StyleDefinition
    ) -> bool:
        """
        Apply a style definition to a layer.
        
        Args:
            layer: QGIS layer to style.
            style_def: Style definition to apply.
            
        Returns:
            True if styling successful.
        """
        if not QGIS_AVAILABLE:
            self.logger.warning("QGIS not available for styling")
            return False
        
        try:
            if isinstance(layer, QgsVectorLayer):
                return self._apply_vector_style(layer, style_def)
            elif isinstance(layer, QgsRasterLayer):
                return self._apply_raster_style(layer, style_def)
            else:
                self.logger.error(f"Unsupported layer type: {type(layer)}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error applying style: {e}")
            return False
    
    def _apply_vector_style(
        self,
        layer: QgsVectorLayer,
        style_def: StyleDefinition
    ) -> bool:
        """Apply style to vector layer."""
        try:
            if style_def.symbol_type == "fill":
                symbol = self._create_fill_symbol(style_def)
            elif style_def.symbol_type == "line":
                symbol = self._create_line_symbol(style_def)
            elif style_def.symbol_type == "marker":
                symbol = self._create_marker_symbol(style_def)
            else:
                self.logger.error(f"Unknown symbol type: {style_def.symbol_type}")
                return False
            
            renderer = QgsSingleSymbolRenderer(symbol)
            layer.setRenderer(renderer)
            
            # Apply labeling if specified
            if style_def.label_field:
                self._apply_labeling(layer, style_def.label_field)
            
            layer.triggerRepaint()
            self.logger.info(f"Applied style '{style_def.name}' to layer '{layer.name()}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying vector style: {e}")
            return False
    
    def _create_fill_symbol(self, style_def: StyleDefinition) -> QgsFillSymbol:
        """Create fill symbol from style definition."""
        symbol = QgsFillSymbol.createSimple({})
        
        if style_def.fill_color:
            color = self._parse_color(style_def.fill_color)
            symbol.symbolLayer(0).setColor(color)
        
        if style_def.stroke_color:
            stroke_color = self._parse_color(style_def.stroke_color)
            symbol.symbolLayer(0).setStrokeColor(stroke_color)
        
        symbol.symbolLayer(0).setStrokeWidth(style_def.stroke_width)
        symbol.setOpacity(style_def.opacity)
        
        return symbol
    
    def _create_line_symbol(self, style_def: StyleDefinition) -> QgsLineSymbol:
        """Create line symbol from style definition."""
        symbol = QgsLineSymbol.createSimple({})
        
        if style_def.stroke_color:
            color = self._parse_color(style_def.stroke_color)
            symbol.symbolLayer(0).setColor(color)
        elif style_def.fill_color:
            color = self._parse_color(style_def.fill_color)
            symbol.symbolLayer(0).setColor(color)
        
        symbol.symbolLayer(0).setWidth(style_def.stroke_width)
        symbol.setOpacity(style_def.opacity)
        
        return symbol
    
    def _create_marker_symbol(self, style_def: StyleDefinition) -> QgsMarkerSymbol:
        """Create marker symbol from style definition."""
        symbol = QgsMarkerSymbol.createSimple({})
        
        if style_def.fill_color:
            color = self._parse_color(style_def.fill_color)
            symbol.symbolLayer(0).setColor(color)
        
        if style_def.stroke_color:
            stroke_color = self._parse_color(style_def.stroke_color)
            symbol.symbolLayer(0).setStrokeColor(stroke_color)
        
        symbol.symbolLayer(0).setSize(style_def.symbol_size)
        symbol.setOpacity(style_def.opacity)
        
        return symbol
    
    def _apply_raster_style(
        self,
        layer: QgsRasterLayer,
        style_def: StyleDefinition
    ) -> bool:
        """Apply style to raster layer."""
        try:
            # Create color ramp shader
            shader = QgsColorRampShader()
            shader.setColorRampType(QgsColorRampShader.Interpolated)
            
            # Default color ramp for single band
            color_entries = [
                QgsColorRampShader.ColorRampItem(0, QtGui.QColor(0, 0, 255), "Low"),
                QgsColorRampShader.ColorRampItem(127, QtGui.QColor(0, 255, 0), "Medium"),
                QgsColorRampShader.ColorRampItem(255, QtGui.QColor(255, 0, 0), "High"),
            ]
            shader.setColorRampItemList(color_entries)
            
            # Create raster shader
            raster_shader = QgsRasterShader()
            raster_shader.setRasterShaderFunction(shader)
            
            # Apply renderer
            renderer = QgsSingleBandPseudoColorRenderer(
                layer.dataProvider(), 1, raster_shader
            )
            layer.setRenderer(renderer)
            layer.triggerRepaint()
            
            self.logger.info(f"Applied raster style to layer '{layer.name()}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying raster style: {e}")
            return False
    
    def apply_categorized_style(
        self,
        layer: QgsVectorLayer,
        field: str,
        categories: Dict[str, str],
        symbol_type: str = "fill"
    ) -> bool:
        """
        Apply categorized style based on attribute values.
        
        Args:
            layer: Vector layer to style.
            field: Attribute field for categorization.
            categories: Dictionary mapping category values to colors.
            symbol_type: Type of symbol (fill, line, marker).
            
        Returns:
            True if styling successful.
            
        Example:
            >>> styler.apply_categorized_style(
            ...     layer,
            ...     field="land_use",
            ...     categories={
            ...         "forest": "#228B22",
            ...         "water": "#0000FF",
            ...         "urban": "#808080"
            ...     }
            ... )
        """
        if not QGIS_AVAILABLE:
            return False
        
        try:
            renderer_categories = []
            
            for value, color in categories.items():
                if symbol_type == "fill":
                    symbol = self._create_fill_symbol(
                        StyleDefinition(name=value, fill_color=color)
                    )
                elif symbol_type == "line":
                    symbol = self._create_line_symbol(
                        StyleDefinition(name=value, stroke_color=color)
                    )
                else:
                    symbol = self._create_marker_symbol(
                        StyleDefinition(name=value, fill_color=color)
                    )
                
                category = QgsRendererCategory(value, symbol, str(value))
                renderer_categories.append(category)
            
            renderer = QgsCategorizedSymbolRenderer(field, renderer_categories)
            layer.setRenderer(renderer)
            layer.triggerRepaint()
            
            self.logger.info(f"Applied categorized style on field '{field}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying categorized style: {e}")
            return False
    
    def apply_graduated_style(
        self,
        layer: QgsVectorLayer,
        field: str,
        ranges: List[Tuple[float, float, str]],
        symbol_type: str = "fill"
    ) -> bool:
        """
        Apply graduated style based on numeric attribute ranges.
        
        Args:
            layer: Vector layer to style.
            field: Numeric attribute field for graduation.
            ranges: List of (min, max, color) tuples.
            symbol_type: Type of symbol (fill, line, marker).
            
        Returns:
            True if styling successful.
            
        Example:
            >>> styler.apply_graduated_style(
            ...     layer,
            ...     field="population",
            ...     ranges=[
            ...         (0, 1000, "#FFFFCC"),
            ...         (1000, 5000, "#FD8D3C"),
            ...         (5000, float('inf'), "#BD0026")
            ...     ]
            ... )
        """
        if not QGIS_AVAILABLE:
            return False
        
        try:
            renderer_ranges = []
            
            for min_val, max_val, color in ranges:
                if symbol_type == "fill":
                    symbol = self._create_fill_symbol(
                        StyleDefinition(name=f"{min_val}-{max_val}", fill_color=color)
                    )
                elif symbol_type == "line":
                    symbol = self._create_line_symbol(
                        StyleDefinition(name=f"{min_val}-{max_val}", stroke_color=color)
                    )
                else:
                    symbol = self._create_marker_symbol(
                        StyleDefinition(name=f"{min_val}-{max_val}", fill_color=color)
                    )
                
                renderer_range = QgsRendererRange(min_val, max_val, symbol, f"{min_val} - {max_val}")
                renderer_ranges.append(renderer_range)
            
            renderer = QgsGraduatedSymbolRenderer(field, renderer_ranges)
            layer.setRenderer(renderer)
            layer.triggerRepaint()
            
            self.logger.info(f"Applied graduated style on field '{field}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying graduated style: {e}")
            return False
    
    def apply_rule_based_style(
        self,
        layer: QgsVectorLayer,
        rules: List[Dict[str, Any]]
    ) -> bool:
        """
        Apply rule-based style with complex conditions.
        
        Args:
            layer: Vector layer to style.
            rules: List of rule definitions with filter and symbol properties.
            
        Returns:
            True if styling successful.
            
        Example:
            >>> styler.apply_rule_based_style(layer, [
            ...     {
            ...         "filter": "area > 1000",
            ...         "symbol": {"fill_color": "#FF0000"},
            ...         "label": "Large"
            ...     },
            ...     {
            ...         "filter": "area <= 1000",
            ...         "symbol": {"fill_color": "#00FF00"},
            ...         "label": "Small"
            ...     }
            ... ])
        """
        if not QGIS_AVAILABLE:
            return False
        
        try:
            rule_list = []
            
            for rule_def in rules:
                filter_expr = rule_def.get("filter", "")
                symbol_def = rule_def.get("symbol", {})
                label = rule_def.get("label", "")
                
                symbol = self._create_fill_symbol(
                    StyleDefinition(
                        name=label,
                        fill_color=symbol_def.get("fill_color"),
                        stroke_color=symbol_def.get("stroke_color"),
                        opacity=symbol_def.get("opacity", 1.0)
                    )
                )
                
                rule = QgsRuleBasedRenderer.Rule(symbol, filter_expr, label=label)
                rule_list.append(rule)
            
            root_rule = QgsRuleBasedRenderer.Rule(None)
            for rule in rule_list:
                root_rule.appendChild(rule)
            
            renderer = QgsRuleBasedRenderer(root_rule)
            layer.setRenderer(renderer)
            layer.triggerRepaint()
            
            self.logger.info(f"Applied rule-based style with {len(rules)} rules")
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying rule-based style: {e}")
            return False
    
    def apply_labeling(
        self,
        layer: QgsVectorLayer,
        field: str,
        font_size: int = 10,
        font_color: str = "#000000",
        placement: str = "above"
    ) -> bool:
        """
        Apply labeling to a vector layer.
        
        Args:
            layer: Vector layer to label.
            field: Attribute field for label text.
            font_size: Font size in points.
            font_color: Font color as hex string.
            placement: Label placement (above, line, centroid).
            
        Returns:
            True if labeling successful.
        """
        if not QGIS_AVAILABLE:
            return False
        
        try:
            settings = QgsPalLayerSettings()
            settings.fieldName = field
            settings.placement = getattr(QgsPalLayerSettings, placement.upper(), QgsPalLayerSettings.OverPoint)
            settings.enabled = True
            
            # Text format
            text_format = QtGui.QTextFormat()
            text_format.setFontPointSize(font_size)
            text_format.setForeground(QtGui.QColor(self._parse_color(font_color)))
            settings.setFormat(text_format)
            
            labeling = QgsVectorLayerSimpleLabeling(settings)
            layer.setLabelsEnabled(True)
            layer.setLabeling(labeling)
            layer.triggerRepaint()
            
            self.logger.info(f"Applied labeling on field '{field}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying labeling: {e}")
            return False
    
    def export_layer(
        self,
        layer: Union[QgsVectorLayer, QgsRasterLayer],
        output_path: str,
        format: str = "geojson",
        include_style: bool = False,
        crs: Optional[str] = None
    ) -> bool:
        """
        Export layer to file with optional styling.
        
        Args:
            layer: Layer to export.
            output_path: Output file path.
            format: Export format (geojson, shapefile, gpkg, kml, geotiff).
            include_style: Whether to include style in export.
            crs: Target CRS for export.
            
        Returns:
            True if export successful.
        """
        if not QGIS_AVAILABLE:
            self.logger.warning("QGIS not available for export")
            return False
        
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(layer, QgsVectorLayer):
                return self._export_vector(layer, output_path, format, include_style, crs)
            elif isinstance(layer, QgsRasterLayer):
                return self._export_raster(layer, output_path, format, crs)
            else:
                self.logger.error(f"Unsupported layer type: {type(layer)}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error exporting layer: {e}")
            return False
    
    def _export_vector(
        self,
        layer: QgsVectorLayer,
        output_path: str,
        format: str,
        include_style: bool,
        crs: Optional[str]
    ) -> bool:
        """Export vector layer."""
        try:
            # Determine driver
            drivers = {
                "geojson": "GeoJSON",
                "shapefile": "ESRI Shapefile",
                "gpkg": "GPKG",
                "kml": "KML",
                "gml": "GML",
            }
            driver = drivers.get(format.lower(), "GPKG")
            
            # Handle shapefile extension
            if format.lower() == "shapefile":
                output_path = str(Path(output_path).with_suffix(".shp"))
            
            # Create transform context if CRS specified
            transform_context = None
            if crs:
                from qgis.core import QgsCoordinateTransform, QgsCoordinateReferenceSystem
                target_crs = QgsCoordinateReferenceSystem(crs)
                transform_context = QgsProject.instance().transformContext()
            
            # Export options
            options = {
                "encoding": "UTF-8",
                "driverName": driver,
                "onlySelected": False,
            }
            
            if include_style:
                # Save style to separate file
                style_path = str(Path(output_path).with_suffix(".qml"))
                layer.saveStyleToQMLFile(style_path)
                self.logger.info(f"Style saved to {style_path}")
            
            error = QgsVectorLayer.writeAsVectorFormat(
                layer, output_path, **options
            )
            
            if error == QgsVectorLayer.NoError:
                self.logger.info(f"Exported vector layer to {output_path}")
                return True
            else:
                self.logger.error(f"Export failed with error code: {error}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error exporting vector: {e}")
            return False
    
    def _export_raster(
        self,
        layer: QgsRasterLayer,
        output_path: str,
        format: str,
        crs: Optional[str]
    ) -> bool:
        """Export raster layer."""
        try:
            # Use GDAL for raster export
            from osgeo import gdal
            
            drivers = {
                "geotiff": "GTiff",
                "tiff": "GTiff",
                "jp2": "JP2OpenJPEG",
                "img": "HFA",
                "asc": "AAIGrid",
            }
            driver_name = drivers.get(format.lower(), "GTiff")
            
            # Open source dataset
            source_path = layer.dataProvider().dataSourceUri()
            source = gdal.Open(source_path)
            
            if not source:
                self.logger.error(f"Cannot open raster source: {source_path}")
                return False
            
            # Get driver and create copy
            driver = gdal.GetDriverByName(driver_name)
            dataset = driver.CreateCopy(output_path, source, 0)
            
            if dataset:
                self.logger.info(f"Exported raster layer to {output_path}")
                return True
            else:
                self.logger.error("Failed to create raster output")
                return False
                
        except Exception as e:
            self.logger.error(f"Error exporting raster: {e}")
            return False
    
    def save_layer_style(self, layer: QgsVectorLayer, style_path: str) -> bool:
        """
        Save layer style to QML file.
        
        Args:
            layer: Layer with style to save.
            style_path: Path for QML style file.
            
        Returns:
            True if save successful.
        """
        if not QGIS_AVAILABLE:
            return False
        
        try:
            Path(style_path).parent.mkdir(parents=True, exist_ok=True)
            
            if layer.saveStyleToQMLFile(style_path):
                self.logger.info(f"Style saved to {style_path}")
                return True
            else:
                self.logger.error(f"Failed to save style to {style_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving style: {e}")
            return False
    
    def load_layer_style(self, layer: QgsVectorLayer, style_path: str) -> bool:
        """
        Load layer style from QML file.
        
        Args:
            layer: Layer to apply style to.
            style_path: Path to QML style file.
            
        Returns:
            True if load successful.
        """
        if not QGIS_AVAILABLE:
            return False
        
        try:
            if not Path(style_path).exists():
                self.logger.error(f"Style file not found: {style_path}")
                return False
            
            error_msg = ""
            if layer.loadStyleFromQMLFile(style_path, error_msg):
                layer.triggerRepaint()
                self.logger.info(f"Style loaded from {style_path}")
                return True
            else:
                self.logger.error(f"Failed to load style: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading style: {e}")
            return False
    
    def _parse_color(self, color_str: str) -> QtGui.QColor:
        """Parse color string to QColor."""
        if color_str.startswith("#"):
            return QtGui.QColor(color_str)
        elif hasattr(QtGui.QColor, color_str.upper()):
            return getattr(QtGui.QColor, color_str.upper())()
        else:
            return QtGui.QColor("#000000")
    
    def get_style_names(self) -> List[str]:
        """Get list of available style names."""
        return list(self.style_library.keys())
    
    def get_style(self, name: str) -> Optional[StyleDefinition]:
        """Get style definition by name."""
        return self.style_library.get(name)


# Convenience functions for common styling tasks
def style_land_cover(layer: QgsVectorLayer, field: str = "class") -> bool:
    """Apply standard land cover styling."""
    styler = LayerStyler()
    categories = {
        "water": "#339AF0",
        "vegetation": "#51CF66",
        "buildings": "#FF6B6B",
        "roads": "#495057",
        "bare_soil": "#E4C48E",
        "urban": "#868E96",
        "forest": "#2F9E44",
        "agriculture": "#94D82D",
    }
    return styler.apply_categorized_style(layer, field, categories)


def style_buildings(layer: QgsVectorLayer) -> bool:
    """Apply standard building footprint styling."""
    styler = LayerStyler()
    style_def = StyleDefinition(
        name="buildings",
        fill_color="#FF6B6B",
        stroke_color="#C92A2A",
        stroke_width=0.5,
        opacity=0.8
    )
    return styler.apply_style(layer, style_def)


def style_roads(layer: QgsVectorLayer) -> bool:
    """Apply standard road network styling."""
    styler = LayerStyler()
    style_def = StyleDefinition(
        name="roads",
        stroke_color="#495057",
        stroke_width=2.0,
        opacity=1.0,
        symbol_type="line"
    )
    return styler.apply_style(layer, style_def)
