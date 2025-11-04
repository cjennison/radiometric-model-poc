"""
Web-optimized visualization manager for smooth heatmap updates.
"""

import io
import base64
import threading
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..models import RadiometricFrame


class WebVisualizationManager:
    """
    Optimized visualization manager for web interface that reduces flashing
    by reusing matplotlib figures and throttling updates.
    """
    
    def __init__(self, figsize=(8, 8)):
        """Initialize the web visualization manager."""
        self.figsize = figsize
        self.fig: Optional[Figure] = None
        self.ax = None
        self.im = None
        self.colorbar = None
        self._lock = threading.Lock()
        self._frame_count = 0
        self._update_every_n_frames = 3  # Only update every 3rd frame (reduces to ~3Hz)
        
        # Initialize the persistent figure
        self._setup_figure()
    
    def _setup_figure(self):
        """Set up the persistent matplotlib figure."""
        with self._lock:
            # Create figure with dark theme
            self.fig = Figure(figsize=self.figsize, facecolor='black')
            self.ax = self.fig.add_subplot(111, facecolor='black')
            
            # Configure axes styling
            self.ax.set_xlabel('X Position (pixels)', color='white', fontsize=10)
            self.ax.set_ylabel('Y Position (pixels)', color='white', fontsize=10)
            self.ax.tick_params(colors='white', labelsize=8)
            self.ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
            
            # Initialize with dummy data to set up the image and colorbar
            dummy_data = np.zeros((150, 150))
            self.im = self.ax.imshow(
                dummy_data,
                cmap='hot',
                aspect='equal',
                interpolation='bilinear'
            )
            
            # Add colorbar once
            self.colorbar = self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
            self.colorbar.set_label('Temperature (K)', color='white', fontsize=10)
            self.colorbar.ax.yaxis.set_tick_params(color='white', labelsize=8)
            plt.setp(plt.getp(self.colorbar.ax.axes, 'yticklabels'), color='white')
            
            # Tight layout to prevent clipping
            self.fig.tight_layout()
    
    def should_update_frame(self) -> bool:
        """
        Check if this frame should be sent to web clients (throttling).
        
        Returns:
            True if frame should be updated, False otherwise
        """
        self._frame_count += 1
        return self._frame_count % self._update_every_n_frames == 0
    
    def create_image(self, frame: RadiometricFrame) -> Optional[str]:
        """
        Create optimized image from radiometric frame by reusing the figure.
        
        Args:
            frame: Radiometric frame to visualize
            
        Returns:
            Base64-encoded PNG image data or None if throttled
        """
        if not self.should_update_frame():
            return None
            
        with self._lock:
            try:
                # Update existing image data instead of creating new figure
                self.im.set_array(frame.data)
                
                # Update color scale
                vmin, vmax = np.min(frame.data), np.max(frame.data)
                self.im.set_clim(vmin, vmax)
                
                # Update title
                title = f'Radiometric Data ({frame.timestamp.strftime("%H:%M:%S")})'
                self.ax.set_title(title, color='white', fontsize=12, fontweight='bold')
                
                # Save to bytes buffer efficiently
                buffer = io.BytesIO()
                self.fig.savefig(
                    buffer,
                    format='png',
                    bbox_inches='tight',
                    facecolor='black',
                    edgecolor='none',
                    dpi=100,
                    pad_inches=0.1
                )
                buffer.seek(0)
                
                # Encode as base64
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                buffer.close()
                
                return f"data:image/png;base64,{image_data}"
                
            except Exception as e:
                print(f"Error creating web image: {e}")
                return None
    
    def set_update_frequency(self, every_n_frames: int):
        """
        Set how often to update the web interface.
        
        Args:
            every_n_frames: Update every N frames (higher = less frequent updates)
        """
        if every_n_frames > 0:
            self._update_every_n_frames = every_n_frames
    
    def cleanup(self):
        """Clean up matplotlib resources."""
        with self._lock:
            if self.fig:
                plt.close(self.fig)
                self.fig = None
                self.ax = None
                self.im = None
                self.colorbar = None