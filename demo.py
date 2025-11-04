#!/usr/bin/env python3
"""
Quick demonstration script for the radiometric engine.

This script shows how to use the engine components to generate and visualize
radiometric data with a simple matplotlib display.
"""

import sys
import os
import time
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from radiometric_engine.services import SunSimulator, HeatmapVisualizer
from radiometric_engine.config import settings

def main():
    """Run a simple demonstration of the radiometric engine."""
    print("🌞 Radiometric Engine Demo")
    print("=" * 50)
    print(f"Grid size: {settings.grid_width}x{settings.grid_height}")
    print(f"Sun temperature: {settings.sun_base_temperature}K")
    print("Generating radiometric data...")
    
    # Create simulator and visualizer
    simulator = SunSimulator()
    visualizer = HeatmapVisualizer(figsize=(10, 8))
    
    # Setup display
    visualizer.setup_display()
    
    print("Starting live simulation (press Ctrl+C to stop)...")
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Generate new frame
            frame = simulator.generate_frame(anomaly_chance=0.05)  # 5% chance of anomaly
            
            # Update visualization
            visualizer.update_frame(frame)
            
            # Refresh display
            plt.pause(0.1)  # 10 FPS
            
            frame_count += 1
            
            # Print stats every 50 frames
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                temp_range = f"{frame.data.min():.1f}K - {frame.data.max():.1f}K"
                time_factor = frame.metadata.get('time_factor', 1.0)
                
                print(f"Frame {frame_count}: {temp_range}, "
                      f"Time factor: {time_factor:.2f}, "
                      f"FPS: {fps:.1f}")
    
    except KeyboardInterrupt:
        print(f"\nDemo stopped after {frame_count} frames")
        
        # Final statistics
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"Average FPS: {avg_fps:.1f}")
        print("Closing visualization...")
        
    finally:
        visualizer.close()
        print("Demo complete!")

if __name__ == "__main__":
    main()