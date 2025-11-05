"""
Temperature tracking service for daily temperature pattern visualization.

This module provides functionality to track mean temperatures throughout a simulated day,
creating a "living graph" that builds up the daily temperature pattern in real-time.
"""

import logging
from datetime import datetime, time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TemperaturePoint:
    """A single temperature measurement at a specific time of day."""
    time_of_day: float  # Hours since midnight (0.0 to 23.999...)
    mean_temperature: float  # Mean temperature in Kelvin
    timestamp: datetime  # When this measurement was taken


class TemperatureTracker:
    """
    Tracks mean temperatures throughout a simulated day for visualization.
    
    This class maintains a rolling log of temperature measurements mapped to time-of-day,
    creating a daily temperature pattern that can be visualized as a "living graph".
    """
    
    def __init__(self, sample_interval_minutes: float = 1.0):
        """
        Initialize the temperature tracker.
        
        Args:
            sample_interval_minutes: Minimum interval between samples in simulated minutes
        """
        self.sample_interval_minutes = sample_interval_minutes
        self.sample_interval_hours = sample_interval_minutes / 60.0
        
        # Daily temperature data: time_of_day -> TemperaturePoint
        self._daily_data: Dict[float, TemperaturePoint] = {}
        
        # Track the current day to detect rollovers
        self._current_day: Optional[int] = None
        self._last_sample_time: Optional[float] = None
    
    def update(self, simulated_time: datetime, mean_temperature: float) -> bool:
        """
        Update the temperature tracker with a new measurement.
        
        Args:
            simulated_time: Current simulated datetime
            mean_temperature: Mean temperature across the radiometric grid
            
        Returns:
            True if a new data point was recorded, False if skipped due to sampling interval
        """
        # Convert to time of day (hours since midnight)
        time_of_day = simulated_time.hour + simulated_time.minute / 60.0 + simulated_time.second / 3600.0
        current_day = simulated_time.day
        
        # Check for day rollover
        if self._current_day is not None and current_day != self._current_day:
            logger.info(f"Day rollover detected: {self._current_day} -> {current_day}")
            self._reset_daily_data()
        
        self._current_day = current_day
        
        # Check sampling interval
        if self._last_sample_time is not None:
            time_diff = time_of_day - self._last_sample_time
            # Handle midnight rollover in time difference calculation
            if time_diff < 0:
                time_diff += 24.0
            
            if time_diff < self.sample_interval_hours:
                return False  # Skip this sample, too soon
        
        # Record the temperature point
        temp_point = TemperaturePoint(
            time_of_day=time_of_day,
            mean_temperature=mean_temperature,
            timestamp=simulated_time
        )
        
        self._daily_data[time_of_day] = temp_point
        self._last_sample_time = time_of_day
        
        return True
    
    def _reset_daily_data(self) -> None:
        """Reset the daily temperature data for a new day."""
        self._daily_data.clear()
        self._last_sample_time = None
    
    def get_daily_data(self) -> List[Dict]:
        """
        Get the current daily temperature data for visualization.
        
        Returns:
            List of dictionaries with 'time_of_day' and 'temperature' keys,
            sorted by time of day
        """
        sorted_data = sorted(self._daily_data.items())
        return [
            {
                'time_of_day': time_of_day,
                'temperature': point.mean_temperature,
                'timestamp': point.timestamp.isoformat()
            }
            for time_of_day, point in sorted_data
        ]
    
    def get_current_time_marker(self, simulated_time: datetime) -> float:
        """
        Get the current time marker position for the graph.
        
        Args:
            simulated_time: Current simulated datetime
            
        Returns:
            Time of day as hours since midnight (0.0 to 23.999...)
        """
        return simulated_time.hour + simulated_time.minute / 60.0 + simulated_time.second / 3600.0
    
    def get_temperature_range(self) -> Tuple[float, float]:
        """
        Get the current temperature range for graph scaling.
        
        Returns:
            Tuple of (min_temp, max_temp) in Kelvin, or (0, 100) if no data
        """
        if not self._daily_data:
            return (0.0, 100.0)  # Default range
        
        temperatures = [point.mean_temperature for point in self._daily_data.values()]
        min_temp = min(temperatures)
        max_temp = max(temperatures)
        
        # Add some padding to the range
        temp_range = max_temp - min_temp
        padding = max(temp_range * 0.1, 5.0)  # 10% padding or 5K minimum
        
        return (min_temp - padding, max_temp + padding)
    
    def get_stats(self) -> Dict:
        """
        Get tracker statistics for debugging and monitoring.
        
        Returns:
            Dictionary with tracker statistics
        """
        temperatures = [point.mean_temperature for point in self._daily_data.values()] if self._daily_data else []
        
        return {
            'data_points': len(self._daily_data),
            'current_day': self._current_day,
            'sample_interval_minutes': self.sample_interval_minutes,
            'last_sample_time': self._last_sample_time,
            'temperature_range': self.get_temperature_range(),
            'mean_temperature': np.mean(temperatures) if temperatures else None,
            'temperature_std': np.std(temperatures) if temperatures else None
        }
    
    def force_reset(self) -> None:
        """Force a reset of the daily data (useful for testing/debugging)."""
        self._reset_daily_data()
        self._current_day = None