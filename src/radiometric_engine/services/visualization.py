"""
Visualization service for displaying radiometric data as heatmaps.

This module provides real-time visualization of thermographic data using matplotlib
with customizable color maps and interactive features.
"""

import logging
import threading
import os
import io
import base64
import glob
from datetime import datetime
from typing import Optional, Tuple, Any, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import matplotlib.patches as patches

from ..models import RadiometricFrame, ThermalAnomaly
from ..services.anomaly_detection import DetectedAnomaly
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


class WebVisualizationManager(HeatmapVisualizer):
    """
    Web-optimized visualization manager for radiometric data.
    
    This class extends HeatmapVisualizer to provide web-specific functionality
    including image generation for web browsers and update throttling.
    """
    
    def __init__(self, figsize: Tuple[float, float] = (8, 8), **kwargs):
        """
        Initialize the web visualization manager.
        
        Args:
            figsize: Figure size in inches (width, height)
            **kwargs: Additional arguments passed to HeatmapVisualizer
        """
        super().__init__(figsize=figsize, **kwargs)
        
        # Web-specific attributes
        self._frame_count = 0
        self._update_frequency = 1  # Update every N frames
        
        # Ensure proper visualization setup
        if not hasattr(self, 'fig') or self.fig is None:
            self.setup_display()  # Use parent's setup method for proper aspect ratio
        
        # Configure for web usage
        plt.style.use('dark_background')
        
    def set_update_frequency(self, every_n_frames: int) -> None:
        """
        Set how often to update the visualization.
        
        Args:
            every_n_frames: Update every N frames to reduce CPU usage
        """
        self._update_frequency = max(1, every_n_frames)
        logger.info(f"Web visualization update frequency set to every {self._update_frequency} frames")
    
    def create_image(self, frame: RadiometricFrame) -> Optional[str]:
        """
        Create a base64-encoded image from a radiometric frame.
        
        This method is optimized for web use with throttling to prevent
        excessive CPU usage on rapid frame updates.
        
        Args:
            frame: Radiometric frame to visualize
            
        Returns:
            Base64-encoded image string, or None if throttled
        """
        import io
        import base64
        
        # Increment frame counter
        self._frame_count += 1
        
        # Check if we should skip this frame for throttling
        if self._frame_count % self._update_frequency != 0:
            return None
        
        try:
            # Update the visualization
            self.update_frame(frame)
            
            # Generate image for web
            if self.fig and self.ax:
                # Save to memory buffer as PNG
                img_buffer = io.BytesIO()
                self.fig.savefig(
                    img_buffer, 
                    format='png', 
                    bbox_inches='tight',
                    facecolor='black',
                    edgecolor='none',
                    dpi=100,  # Reasonable web DPI
                    pad_inches=0.1
                )
                img_buffer.seek(0)
                
                # Encode as base64 for web transmission
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                img_buffer.close()
                
                return f"data:image/png;base64,{img_base64}"
                
        except Exception as e:
            logger.error(f"Error creating web image: {e}")
            return None
        
        return None
    
    def save_anomaly_image(self, frame: RadiometricFrame, detected_anomalies: List, save_dir: str = "anomaly_captures", max_images: int = 50) -> Optional[str]:
        """
        Save a complete heatmap image with anomaly overlays when anomalies are detected.
        Automatically manages folder size by deleting oldest images when limit is exceeded.
        
        Args:
            frame: Radiometric frame containing the data
            detected_anomalies: List of DetectedAnomaly objects
            save_dir: Directory to save anomaly images
            max_images: Maximum number of images to keep (default: 50)
            
        Returns:
            Path to saved image file, or None if failed
        """
        if not detected_anomalies:
            return None
            
        try:
            # Create save directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Clean up old images if we're at the limit
            self._cleanup_old_images(save_dir, max_images)
            
            # Create a copy of the current figure for saving
            save_fig, save_ax = plt.subplots(figsize=self.figsize, facecolor='black')
            
            # Generate the heatmap
            if self.temperature_range:
                vmin, vmax = self.temperature_range
            else:
                vmin, vmax = np.min(frame.data), np.max(frame.data)
            
            # Create the base heatmap
            im = save_ax.imshow(
                frame.data,
                cmap=self.colormap,
                vmin=vmin,
                vmax=vmax,
                aspect='equal',
                interpolation='bilinear'
            )
            
            # Add colorbar
            cbar = save_fig.colorbar(im, ax=save_ax, label='Temperature (K)')
            cbar.ax.tick_params(colors='white')
            cbar.set_label('Temperature (K)', color='white')
            
            # Style the plot
            save_ax.set_title(
                f'ANOMALY DETECTED - {frame.timestamp.strftime("%Y-%m-%d %H:%M:%S")}',
                color='red', 
                fontsize=14, 
                fontweight='bold'
            )
            save_ax.set_xlabel('X Position', color='white')
            save_ax.set_ylabel('Y Position', color='white')
            save_ax.tick_params(colors='white')
            
            # Add grid
            save_ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
            
            # Overlay anomaly markers
            for anomaly in detected_anomalies:
                center_x, center_y = anomaly.region_center
                
                # Determine color based on severity
                severity_colors = {
                    'LOW': '#ffd700',     # Gold
                    'MEDIUM': '#ff8c00',  # Dark orange
                    'HIGH': '#ff4500',    # Orange red
                    'CRITICAL': '#ff0000' # Red
                }
                color = severity_colors.get(anomaly.severity.name, '#ffffff')
                
                # Calculate opacity based on confidence
                alpha = 0.4 + 0.4 * anomaly.confidence_score
                
                # Draw affected area if we have pixel data
                if anomaly.affected_pixels:
                    for pixel_x, pixel_y in anomaly.affected_pixels:
                        save_ax.add_patch(
                            plt.Rectangle(
                                (pixel_x - 0.5, pixel_y - 0.5), 1, 1,
                                fill=True,
                                facecolor=color,
                                alpha=0.8,  # Even more visible
                                edgecolor='white',  # Add white border for contrast
                                linewidth=1.0,  # Thicker border
                                zorder=10  # Ensure rectangles appear above heatmap
                            )
                        )
                
                # Draw center marker with enhanced visibility
                # First draw a larger white 'x' for contrast
                save_ax.scatter(
                    center_x, center_y,
                    s=300, 
                    c='white', 
                    marker='x', 
                    linewidths=4,
                    alpha=1.0,
                    zorder=15  # Ensure center marker appears above rectangles
                )
                # Then draw the colored 'x' on top
                save_ax.scatter(
                    center_x, center_y,
                    s=200, 
                    c=color, 
                    marker='x', 
                    linewidths=3,
                    alpha=1.0,
                    zorder=16  # Ensure colored marker is on top
                )
                
                # Add text annotation
                save_ax.annotate(
                    f'{anomaly.severity.name}\n{anomaly.anomaly_type.name}',
                    (center_x, center_y),
                    xytext=(10, 10), 
                    textcoords='offset points',
                    color=color,
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor=color),
                    zorder=20  # Ensure text appears above everything
                )
            
            # Generate filename with timestamp and anomaly info
            timestamp_str = frame.timestamp.strftime("%Y%m%d_%H%M%S")
            severity_levels = [a.severity.name for a in detected_anomalies]
            max_severity = max(severity_levels, key=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].index)
            filename = f"anomaly_{timestamp_str}_{max_severity}_{len(detected_anomalies)}anomalies.png"
            filepath = os.path.join(save_dir, filename)
            
            # Save the image
            save_fig.savefig(
                filepath,
                format='png',
                bbox_inches='tight',
                facecolor='black',
                edgecolor='none',
                dpi=150,  # Higher DPI for saved images
                pad_inches=0.2
            )
            
            # Close the figure to free memory
            plt.close(save_fig)
            
            logger.info(f"Saved anomaly image: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving anomaly image: {e}")
            if 'save_fig' in locals():
                plt.close(save_fig)
            return None
    
    def _cleanup_old_images(self, save_dir: str, max_images: int) -> None:
        """
        Remove oldest anomaly images if the folder exceeds max_images limit.
        
        Args:
            save_dir: Directory containing anomaly images
            max_images: Maximum number of images to keep
        """
        try:
            # Get all PNG files in the directory
            image_pattern = os.path.join(save_dir, "anomaly_*.png")
            image_files = glob.glob(image_pattern)
            
            # If we're at or above the limit, remove oldest files
            if len(image_files) >= max_images:
                # Sort by modification time (oldest first)
                image_files.sort(key=lambda x: os.path.getmtime(x))
                
                # Calculate how many to remove (keep one slot for the new image)
                files_to_remove = len(image_files) - max_images + 1
                
                for i in range(files_to_remove):
                    if i < len(image_files):
                        file_to_remove = image_files[i]
                        try:
                            os.remove(file_to_remove)
                            logger.info(f"Removed old anomaly image: {os.path.basename(file_to_remove)}")
                        except OSError as e:
                            logger.warning(f"Failed to remove old image {file_to_remove}: {e}")
                            
        except Exception as e:
            logger.error(f"Error during anomaly image cleanup: {e}")