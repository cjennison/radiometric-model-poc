"""
Main application entry point for the radiometric engine.

This module provides both standalone visualization and web-based interfaces
for real-time radiometric data simulation and display.
"""

import logging
import sys
import signal
import time
from datetime import datetime
from typing import Optional

from .services import SunSimulator, DataStreamEngine, AnimatedHeatmapVisualizer
from .config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('radiometric_engine.log')
    ]
)

logger = logging.getLogger(__name__)


class RadiometricEngine:
    """
    Main radiometric engine that coordinates simulation, streaming, and visualization.
    """
    
    def __init__(self) -> None:
        """Initialize the radiometric engine."""
        logger.info("Initializing Radiometric Engine")
        
        # Initialize components
        self.simulator = SunSimulator()
        self.data_stream = DataStreamEngine(simulator=self.simulator)
        self.visualizer: Optional[AnimatedHeatmapVisualizer] = None
        
        # Engine state
        self._running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Radiometric Engine initialized successfully")
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def start_visualization(self, web_mode: bool = False) -> None:
        """
        Start the visualization system.
        
        Args:
            web_mode: If True, start web interface; if False, start standalone GUI
        """
        if web_mode:
            self._start_web_interface()
        else:
            self._start_standalone_visualization()
    
    def _start_standalone_visualization(self) -> None:
        """Start standalone matplotlib visualization."""
        logger.info("Starting standalone visualization")
        
        # Create and setup visualizer
        self.visualizer = AnimatedHeatmapVisualizer(
            figsize=(14, 10),
            colormap='hot',
            update_interval_ms=100
        )
        
        # Setup the display
        self.visualizer.setup_display()
        
        # Connect visualizer to data stream
        self.data_stream.add_consumer(self.visualizer.queue_frame)
        
        # Start streaming and animation
        self.data_stream.start_streaming()
        self.visualizer.start_animation()
        
        self._running = True
        
        try:
            # Show the visualization (blocks until window closed)
            self.visualizer.show(block=True)
        except KeyboardInterrupt:
            logger.info("Visualization interrupted by user")
        finally:
            self.stop()
    
    def _start_web_interface(self) -> None:
        """Start Flask web interface."""
        from .web import create_app
        
        logger.info("Starting web interface")
        
        # Create Flask app
        app = create_app(self.data_stream)
        
        # Start data streaming
        self.data_stream.start_streaming()
        self._running = True
        
        try:
            # Start Flask development server
            app.run(
                host=settings.flask_host,
                port=settings.flask_port,
                debug=settings.flask_debug,
                threaded=True
            )
        except KeyboardInterrupt:
            logger.info("Web interface interrupted by user")
        finally:
            self.stop()
    
    def force_anomaly(self) -> None:
        """Force the next frame to contain a solar anomaly."""
        self.data_stream.force_anomaly()
        logger.info("Forced anomaly triggered")
    
    def set_anomaly_probability(self, probability: float) -> None:
        """
        Set the probability of anomalies appearing.
        
        Args:
            probability: Probability between 0.0 and 1.0
        """
        self.data_stream.set_anomaly_probability(probability)
    
    def get_stats(self) -> dict:
        """Get engine statistics."""
        stats = {
            "engine_running": self._running,
            "current_time": datetime.now().isoformat(),
        }
        
        # Add streaming stats
        if self.data_stream:
            stats.update(self.data_stream.get_stats())
        
        return stats
    
    def stop(self) -> None:
        """Stop the radiometric engine."""
        if not self._running:
            return
        
        logger.info("Stopping Radiometric Engine")
        self._running = False
        
        # Stop data streaming
        if self.data_stream:
            self.data_stream.stop_streaming()
        
        # Stop visualization
        if self.visualizer:
            self.visualizer.stop_animation()
            self.visualizer.close()
        
        logger.info("Radiometric Engine stopped")
    
    def run_demo(self, duration_seconds: int = 60) -> None:
        """
        Run a demonstration of the engine for a specified duration.
        
        Args:
            duration_seconds: How long to run the demo
        """
        logger.info(f"Starting {duration_seconds}s demonstration")
        
        # Start data streaming
        self.data_stream.start_streaming()
        self._running = True
        
        start_time = time.time()
        next_anomaly = start_time + 10  # First anomaly after 10 seconds
        
        try:
            while time.time() - start_time < duration_seconds and self._running:
                # Force periodic anomalies for demonstration
                if time.time() >= next_anomaly:
                    self.force_anomaly()
                    next_anomaly = time.time() + 15  # Next anomaly in 15 seconds
                
                # Get and print latest frame stats
                frame = self.data_stream.get_latest_frame(timeout=1.0)
                if frame:
                    temp_stats = f"Temp range: {frame.data.min():.1f}K - {frame.data.max():.1f}K"
                    time_factor = frame.metadata.get('time_factor', 1.0) if frame.metadata else 1.0
                    logger.info(f"{temp_stats}, Time factor: {time_factor:.2f}")
                
                time.sleep(2)
                
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        finally:
            self.stop()
        
        # Print final statistics
        stats = self.get_stats()
        logger.info(f"Demo completed. Final stats: {stats}")


def main() -> None:
    """Main entry point for the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Radiometric Engine - Thermographic Simulation")
    parser.add_argument(
        "--mode",
        choices=["gui", "web", "demo"],
        default="gui",
        help="Execution mode: gui (standalone), web (Flask server), or demo (headless)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration for demo mode in seconds"
    )
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.01,
        help="Anomaly probability (0.0 to 1.0)"
    )
    
    args = parser.parse_args()
    
    # Create and configure engine
    engine = RadiometricEngine()
    engine.set_anomaly_probability(args.anomaly_rate)
    
    try:
        if args.mode == "gui":
            engine.start_visualization(web_mode=False)
        elif args.mode == "web":
            engine.start_visualization(web_mode=True)
        elif args.mode == "demo":
            engine.run_demo(duration_seconds=args.duration)
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()