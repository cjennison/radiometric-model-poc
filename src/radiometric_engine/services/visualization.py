"""
Visualization service for displaying radiometric data as heatmaps.

This module provides real-time visualization of thermographic data using matplotlib
with customizable color maps and interactive features.
"""

import logging
import threading
from datetime import datetime
from typing import Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import matplotlib.patches as patches

from ..models import RadiometricFrame, ThermalAnomaly
from ..config import settings

logger = logging.getLogger(__name__)


class HeatmapVisualizer:
    """
    Real-time heatmap visualization for radiometric data.
    
    This visualizer creates and updates a matplotlib heatmap display that shows
    temperature distributions with customizable color schemes and anomaly highlighting.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 10),
        colormap: str = "hot",
        temperature_range: Optional[Tuple[float, float]] = None,
        update_interval_ms: int = 100,
    ) -> None:
        """
        Initialize the heatmap visualizer.
        
        Args:
            figsize: Figure size (width, height) in inches
            colormap: Matplotlib colormap name for temperature visualization
            temperature_range: Optional fixed temperature range (min, max) in Kelvin
            update_interval_ms: Update interval in milliseconds
        """
        self.figsize = figsize
        self.colormap = colormap
        self.temperature_range = temperature_range
        self.update_interval_ms = update_interval_ms
        
        # Visualization state
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.im: Optional[plt.matplotlib.image.AxesImage] = None
        self.colorbar: Optional[plt.matplotlib.colorbar.Colorbar] = None
        self.title_text: Optional[plt.Text] = None
        self.stats_text: Optional[plt.Text] = None
        
        # Data state
        self.current_frame: Optional[RadiometricFrame] = None
        self.anomalies: list[ThermalAnomaly] = []
        self.anomaly_patches: list[plt.matplotlib.patches.Patch] = []
        
        # Threading for non-blocking updates
        self._update_lock = threading.Lock()
        
        logger.info(
            f"Heatmap visualizer initialized: {figsize}, colormap={colormap}, "
            f"update_interval={update_interval_ms}ms"
        )
    
    def setup_display(self) -> None:
        """Initialize the matplotlib display."""
        plt.ion()  # Enable interactive mode
        
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.fig.patch.set_facecolor('black')
        
        # Set up axes
        self.ax.set_facecolor('black')
        self.ax.set_aspect('equal')
        
        # Add title and stats text
        self.title_text = self.fig.suptitle(
            'Radiometric Data - Sun Simulation',
            fontsize=16,
            color='white',
            fontweight='bold'
        )
        
        self.stats_text = self.fig.text(
            0.02, 0.02,
            '',
            fontsize=10,
            color='white',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )
        
        # Initialize with dummy data
        dummy_data = np.zeros((settings.grid_height, settings.grid_width))
        self._create_heatmap(dummy_data)
        
        plt.tight_layout()
        logger.info("Display setup complete")
    
    def _create_heatmap(self, data: np.ndarray) -> None:
        """
        Create the initial heatmap display.
        
        Args:
            data: 2D temperature data array
        """
        # Determine temperature range
        if self.temperature_range:
            vmin, vmax = self.temperature_range
        else:
            vmin, vmax = np.min(data), np.max(data)
            
        # Create heatmap
        self.im = self.ax.imshow(
            data,
            cmap=self.colormap,
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
            interpolation='bilinear'
        )
        
        # Add colorbar
        if self.colorbar is None:
            self.colorbar = plt.colorbar(
                self.im,
                ax=self.ax,
                fraction=0.046,
                pad=0.04
            )
            self.colorbar.set_label('Temperature (K)', color='white', fontsize=12)
            self.colorbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(self.colorbar.ax.axes, 'yticklabels'), color='white')
        
        # Configure axes
        self.ax.set_xlabel('X Position (pixels)', color='white', fontsize=12)
        self.ax.set_ylabel('Y Position (pixels)', color='white', fontsize=12)
        self.ax.tick_params(colors='white')
        
        # Add grid
        self.ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    def update_frame(self, frame: RadiometricFrame) -> None:
        """
        Update the visualization with a new frame.
        
        Args:
            frame: New radiometric frame to display
        """
        with self._update_lock:
            self.current_frame = frame
            
            if self.im is None:
                self._create_heatmap(frame.data)
            else:
                # Update existing heatmap
                self.im.set_array(frame.data)
                
                # Update color scale if not fixed
                if not self.temperature_range:
                    vmin, vmax = np.min(frame.data), np.max(frame.data)
                    self.im.set_clim(vmin, vmax)
            
            # Update title with timestamp
            if self.title_text:
                self.title_text.set_text(
                    f'Radiometric Data - Sun Simulation ({frame.timestamp.strftime("%H:%M:%S")})'
                )
            
            # Update statistics
            self._update_stats_display(frame)
            
            # Clear old anomaly highlights
            self._clear_anomaly_highlights()
    
    def highlight_anomalies(self, anomalies: list[ThermalAnomaly]) -> None:
        """
        Highlight detected anomalies on the heatmap.
        
        Args:
            anomalies: List of thermal anomalies to highlight
        """
        with self._update_lock:
            self.anomalies = anomalies.copy()
            self._clear_anomaly_highlights()
            
            for anomaly in anomalies:
                # Create highlight patch based on severity
                color = self._get_anomaly_color(anomaly.severity)
                alpha = 0.3 + 0.4 * anomaly.severity  # More severe = more opaque
                
                # Create circle around anomaly location
                circle = patches.Circle(
                    (anomaly.location[0], anomaly.location[1]),
                    radius=np.sqrt(anomaly.area_pixels / np.pi),
                    linewidth=2,
                    edgecolor=color,
                    facecolor='none',
                    alpha=alpha,
                    linestyle='--'
                )
                
                self.ax.add_patch(circle)
                self.anomaly_patches.append(circle)
                
                # Add text label
                text = self.ax.text(
                    anomaly.location[0] + 10,
                    anomaly.location[1] - 10,
                    f'{anomaly.description}\n{anomaly.severity:.2f}',
                    color=color,
                    fontsize=8,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
                )
                self.anomaly_patches.append(text)
    
    def _get_anomaly_color(self, severity: float) -> str:
        """
        Get color for anomaly highlighting based on severity.
        
        Args:
            severity: Anomaly severity (0.0 to 1.0)
            
        Returns:
            Color string for matplotlib
        """
        if severity < 0.3:
            return 'yellow'
        elif severity < 0.7:
            return 'orange'
        else:
            return 'red'
    
    def _clear_anomaly_highlights(self) -> None:
        """Clear all anomaly highlight patches."""
        for patch in self.anomaly_patches:
            if hasattr(patch, 'remove'):
                patch.remove()
        self.anomaly_patches.clear()
    
    def _update_stats_display(self, frame: RadiometricFrame) -> None:
        """
        Update the statistics display.
        
        Args:
            frame: Current radiometric frame
        """
        if not self.stats_text or not frame.metadata:
            return
        
        stats_text = (
            f"Time Factor: {frame.metadata.get('time_factor', 0):.2f}\n"
            f"Min Temp: {np.min(frame.data):.1f}K\n"
            f"Max Temp: {np.max(frame.data):.1f}K\n"
            f"Mean Temp: {np.mean(frame.data):.1f}K\n"
            f"Grid: {frame.width}x{frame.height}\n"
            f"Anomalies: {len(self.anomalies)}"
        )
        
        self.stats_text.set_text(stats_text)
    
    def save_frame(self, filename: str, dpi: int = 300) -> None:
        """
        Save the current frame to a file.
        
        Args:
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        if self.fig:
            self.fig.savefig(
                filename,
                dpi=dpi,
                bbox_inches='tight',
                facecolor='black',
                edgecolor='none'
            )
            logger.info(f"Saved frame to {filename}")
    
    def show(self, block: bool = True) -> None:
        """
        Display the visualization.
        
        Args:
            block: Whether to block execution until window is closed
        """
        if self.fig:
            plt.show(block=block)
    
    def close(self) -> None:
        """Close the visualization display."""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.im = None
            self.colorbar = None
            logger.info("Visualization closed")


class AnimatedHeatmapVisualizer(HeatmapVisualizer):
    """
    Animated version of the heatmap visualizer that automatically updates.
    
    This extends the base visualizer with automatic frame updates for smooth
    real-time visualization.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.animation: Optional[animation.FuncAnimation] = None
        self._frame_queue = []
        self._max_queue_size = 5
    
    def start_animation(self) -> None:
        """Start the automatic animation loop."""
        if self.fig is None:
            self.setup_display()
        
        self.animation = animation.FuncAnimation(
            self.fig,
            self._animate_frame,
            interval=self.update_interval_ms,
            blit=False,
            cache_frame_data=False
        )
        
        logger.info("Started animated visualization")
    
    def stop_animation(self) -> None:
        """Stop the automatic animation."""
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
            logger.info("Stopped animated visualization")
    
    def queue_frame(self, frame: RadiometricFrame) -> None:
        """
        Queue a frame for display in the animation.
        
        Args:
            frame: RadiometricFrame to queue for display
        """
        with self._update_lock:
            self._frame_queue.append(frame)
            
            # Keep queue size manageable
            if len(self._frame_queue) > self._max_queue_size:
                self._frame_queue.pop(0)
    
    def _animate_frame(self, frame_num: int) -> list:
        """
        Animation callback function.
        
        Args:
            frame_num: Animation frame number (unused)
            
        Returns:
            List of artists that were modified
        """
        with self._update_lock:
            if self._frame_queue:
                latest_frame = self._frame_queue.pop(0)
                self.update_frame(latest_frame)
        
        return [self.im] if self.im else []