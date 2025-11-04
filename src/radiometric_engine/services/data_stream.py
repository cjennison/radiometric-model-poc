"""
Real-time data streaming engine for radiometric data.

This module provides a streaming engine that continuously generates and broadcasts
radiometric data frames in real-time for visualization and analysis.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Any
from queue import Queue, Full

from ..models import RadiometricFrame
from ..config import settings
from .sun_simulator import SunSimulator
from .temperature_tracker import TemperatureTracker

logger = logging.getLogger(__name__)


class DataStreamEngine:
    """
    Real-time streaming engine for radiometric data.
    
    This engine continuously generates radiometric frames and distributes them
    to registered consumers (visualizers, anomaly detectors, etc.).
    """
    
    def __init__(
        self,
        update_frequency_hz: float = None,
        max_queue_size: int = 100,
        simulator: Optional[SunSimulator] = None,
    ) -> None:
        """
        Initialize the data streaming engine.
        
        Args:
            update_frequency_hz: Frames per second to generate
            max_queue_size: Maximum number of frames to queue
            simulator: Optional custom simulator (uses SunSimulator by default)
        """
        self.update_frequency_hz = update_frequency_hz or settings.update_frequency_hz
        self.max_queue_size = max_queue_size
        self.simulator = simulator or SunSimulator()
        
        # Time simulation parameters
        self._time_speed = 144  # Default: 1 day in 10 minutes (24*60/10 = 144x)
        current_time = time.time()
        
        # Start simulation at 6:00 AM of current date
        from datetime import datetime, time as time_obj
        current_date = datetime.fromtimestamp(current_time).date()
        self._simulation_start_time = datetime.combine(current_date, time_obj(6, 0, 0))
        self._real_start_time = current_time
        
        # Temperature tracking for daily pattern visualization
        self.temperature_tracker = TemperatureTracker(sample_interval_minutes=1.0)
        
        # Streaming state
        self._is_streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._frame_consumers: List[Callable[[RadiometricFrame], None]] = []
        self._frame_queue: Queue[RadiometricFrame] = Queue(maxsize=max_queue_size)
        self._stats = {
            "frames_generated": 0,
            "frames_dropped": 0,
            "consumers_count": 0,
            "start_time": None,
        }
        
        # Anomaly simulation controls
        self._anomaly_probability = 0.01
        self._force_next_anomaly = False
        
        logger.info(
            f"Data stream engine initialized: {self.update_frequency_hz}Hz, "
            f"queue size={max_queue_size}"
        )
    
    def add_consumer(self, consumer: Callable[[RadiometricFrame], None]) -> None:
        """
        Add a consumer function that will receive each frame.
        
        Args:
            consumer: Function that takes a RadiometricFrame as input
        """
        self._frame_consumers.append(consumer)
        self._stats["consumers_count"] = len(self._frame_consumers)
        logger.info(f"Added frame consumer. Total consumers: {len(self._frame_consumers)}")
    
    def remove_consumer(self, consumer: Callable[[RadiometricFrame], None]) -> None:
        """
        Remove a frame consumer.
        
        Args:
            consumer: Consumer function to remove
        """
        if consumer in self._frame_consumers:
            self._frame_consumers.remove(consumer)
            self._stats["consumers_count"] = len(self._frame_consumers)
            logger.info(f"Removed frame consumer. Total consumers: {len(self._frame_consumers)}")
    
    def force_anomaly(self) -> None:
        """Force the immediate creation of a solar anomaly."""
        if hasattr(self.simulator, 'force_anomaly_creation'):
            self.simulator.force_anomaly_creation()
            logger.info("Forced creation of solar anomaly")
        else:
            # Fallback for older simulator versions
            self._force_next_anomaly = True
            logger.info("Next frame will contain a forced solar anomaly")
    
    def set_anomaly_probability(self, probability: float) -> None:
        """
        Set the probability of anomalies appearing in frames.
        
        Args:
            probability: Probability between 0.0 and 1.0
        """
        if not (0.0 <= probability <= 1.0):
            raise ValueError("Probability must be between 0.0 and 1.0")
        
        self._anomaly_probability = probability
        logger.info(f"Anomaly probability set to {probability:.3f}")
    
    def start_streaming(self) -> None:
        """Start the real-time data streaming."""
        if self._is_streaming:
            logger.warning("Streaming is already active")
            return
        
        self._is_streaming = True
        self._stats["start_time"] = datetime.now()
        self._stats["frames_generated"] = 0
        self._stats["frames_dropped"] = 0
        
        # Start streaming thread
        self._stream_thread = threading.Thread(
            target=self._streaming_loop,
            name="radiometric-stream",
            daemon=True,
        )
        self._stream_thread.start()
        
        logger.info(f"Started streaming at {self.update_frequency_hz}Hz")
    
    def stop_streaming(self) -> None:
        """Stop the data streaming."""
        if not self._is_streaming:
            logger.warning("Streaming is not active")
            return
        
        self._is_streaming = False
        
        # Wait for streaming thread to finish
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=2.0)
        
        logger.info("Stopped streaming")
    
    def get_stats(self) -> dict:
        """
        Get streaming statistics.
        
        Returns:
            Dictionary containing streaming statistics
        """
        stats = self._stats.copy()
        
        if stats["start_time"]:
            runtime = (datetime.now() - stats["start_time"]).total_seconds()
            stats["runtime_seconds"] = runtime
            
            if runtime > 0:
                stats["average_fps"] = stats["frames_generated"] / runtime
            else:
                stats["average_fps"] = 0.0
        
        stats["queue_size"] = self._frame_queue.qsize()
        stats["is_streaming"] = self._is_streaming
        stats["time_speed"] = self._time_speed
        stats["simulated_time"] = self.get_simulated_time()
        
        return stats
    
    def get_latest_frame(self, timeout: float = 1.0) -> Optional[RadiometricFrame]:
        """
        Get the most recent frame from the queue.
        
        Args:
            timeout: Maximum time to wait for a frame
            
        Returns:
            Latest RadiometricFrame or None if timeout
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except Exception:
            return None
    
    def _streaming_loop(self) -> None:
        """Main streaming loop that generates and distributes frames."""
        frame_interval = 1.0 / self.update_frequency_hz
        
        logger.info(f"Starting streaming loop with {frame_interval:.3f}s interval")
        
        while self._is_streaming:
            loop_start = time.time()
            
            try:
                # Determine anomaly probability for this frame
                anomaly_chance = self._anomaly_probability
                if self._force_next_anomaly:
                    anomaly_chance = 1.0
                    self._force_next_anomaly = False
                
                # Generate frame with simulated time
                simulated_time = self.get_simulated_time()
                frame = self.simulator.generate_frame(
                    anomaly_chance=anomaly_chance,
                    current_time=simulated_time
                )
                self._stats["frames_generated"] += 1
                
                # Update temperature tracker with mean temperature
                import numpy as np
                mean_temp = float(np.mean(frame.data))
                self.temperature_tracker.update(simulated_time, mean_temp)
                
                # Add to queue (non-blocking)
                try:
                    self._frame_queue.put_nowait(frame)
                except Full:
                    # Queue is full, drop oldest frame
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait(frame)
                        self._stats["frames_dropped"] += 1
                    except Exception:
                        self._stats["frames_dropped"] += 1
                
                # Distribute to consumers
                self._distribute_frame(frame)
                
                # Sleep to maintain frame rate
                loop_duration = time.time() - loop_start
                sleep_time = max(0, frame_interval - loop_duration)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif loop_duration > frame_interval * 1.5:
                    # Log if we're significantly behind
                    logger.warning(
                        f"Frame generation taking too long: {loop_duration:.3f}s "
                        f"(target: {frame_interval:.3f}s)"
                    )
                
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}", exc_info=True)
                time.sleep(0.1)  # Brief pause on error
        
        logger.info("Streaming loop finished")
    
    def _distribute_frame(self, frame: RadiometricFrame) -> None:
        """
        Distribute a frame to all registered consumers.
        
        Args:
            frame: RadiometricFrame to distribute
        """
        for consumer in self._frame_consumers:
            try:
                consumer(frame)
            except Exception as e:
                logger.error(f"Error in frame consumer {consumer}: {e}", exc_info=True)
    
    def set_time_speed(self, speed: int) -> None:
        """
        Set the time speed multiplier for simulation.
        
        Args:
            speed: Time speed multiplier (1 = real time, 144 = 1 day in 10 minutes)
        """
        self._time_speed = max(1, speed)
        logger.info(f"Time speed set to {self._time_speed}x")
    
    def reset_time(self) -> None:
        """Reset simulation time to current real time."""
        current_time = time.time()
        self._simulation_start_time = datetime.fromtimestamp(current_time)
        self._real_start_time = current_time
        logger.info("Simulation time reset to current time")
    
    def get_simulated_time(self) -> datetime:
        """
        Get the current simulated time based on speed multiplier.
        
        Returns:
            Current simulated datetime
        """
        current_real_time = time.time()
        elapsed_real_time = current_real_time - self._real_start_time
        elapsed_simulated_time = elapsed_real_time * self._time_speed
        
        simulated_datetime = self._simulation_start_time + timedelta(seconds=elapsed_simulated_time)
        return simulated_datetime

    def get_temperature_graph_data(self) -> dict:
        """
        Get temperature graph data for visualization.
        
        Returns:
            Dictionary containing temperature data and current time marker
        """
        current_time = self.get_simulated_time()
        return {
            'daily_data': self.temperature_tracker.get_daily_data(),
            'current_time_marker': self.temperature_tracker.get_current_time_marker(current_time),
            'temperature_range': self.temperature_tracker.get_temperature_range(),
            'tracker_stats': self.temperature_tracker.get_stats()
        }
    
    def reset_temperature_tracking(self) -> None:
        """Reset the temperature tracking data (useful for testing)."""
        self.temperature_tracker.force_reset()
        logger.info("Temperature tracking data reset")

    def __enter__(self):
        """Context manager entry."""
        self.start_streaming()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_streaming()