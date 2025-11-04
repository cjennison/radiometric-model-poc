"""
Sun simulation service for generating realistic radiometric data.

This module simulates a thermographic camera pointed at the sun, generating
realistic temperature distributions with time-based variations and atmospheric effects.
"""

import logging
import time
from datetime import datetime
from typing import Tuple, Optional, List
from dataclasses import dataclass
import numpy as np

from ..models import RadiometricFrame, SimulationParameters
from ..config import settings

@dataclass
class PersistentAnomaly:
    """Represents a solar anomaly that persists across multiple frames."""
    position: Tuple[int, int]  # (row, col) position on grid
    anomaly_type: str  # 'sunspot', 'flare', 'prominence'
    intensity: float  # Relative intensity multiplier (0.0 to 2.0)
    size: float  # Radius in pixels
    total_duration: int  # Total frames this anomaly should persist
    remaining_frames: int  # Frames remaining for this anomaly
    creation_time: float  # Timestamp when anomaly was created
    
    @property
    def age_factor(self) -> float:
        """Calculate age-based intensity factor (0.0 to 1.0)."""
        if self.total_duration <= 0:
            return 0.0
        progress = 1.0 - (self.remaining_frames / self.total_duration)
        # Use a bell curve for realistic lifecycle: weak start, peak middle, fade end
        if progress < 0.5:
            return 2 * progress  # Ramp up
        else:
            return 2 * (1 - progress)  # Fade out
    
    @property
    def current_intensity(self) -> float:
        """Get current intensity considering age factor."""
        return self.intensity * self.age_factor

logger = logging.getLogger(__name__)


class SunSimulator:
    """
    Simulates radiometric data from a thermographic camera pointed at the sun.
    
    This simulator creates realistic temperature distributions that vary based on:
    - Time of day (cooler in morning/evening, hotter at midday)
    - Atmospheric effects (clouds, turbulence)
    - Solar phenomena (sunspots, flares)
    """
    
    def __init__(
        self,
        grid_width: int = None,
        grid_height: int = None,
        sun_radius: int = None,
        base_temperature: float = None,
    ) -> None:
        """
        Initialize the sun simulator.
        
        Args:
            grid_width: Width of the simulation grid in pixels
            grid_height: Height of the simulation grid in pixels  
            sun_radius: Radius of the sun disc in pixels
            base_temperature: Base temperature of the sun in Kelvin
        """
        self.grid_width = grid_width or settings.grid_width
        self.grid_height = grid_height or settings.grid_height
        self.sun_radius = sun_radius or settings.sun_radius_pixels
        self.base_temperature = base_temperature or settings.sun_base_temperature
        
        # Position sun at center of grid
        self.sun_center_x = self.grid_width // 2
        self.sun_center_y = self.grid_height // 2
        
        # Create coordinate grids for efficient computation
        self._create_coordinate_grids()
        
        # Initialize atmospheric noise state
        self._noise_phase = 0.0
        self._atmospheric_turbulence = np.random.random((self.grid_height, self.grid_width))
        
        # Initialize persistent anomaly tracking
        self.active_anomalies: List[PersistentAnomaly] = []
        self.last_anomaly_check = 0.0  # Timestamp of last anomaly creation check
        self.frame_count = 0
        
        logger.info(
            f"Sun simulator initialized: {self.grid_width}x{self.grid_height} grid, "
            f"sun radius={self.sun_radius}, base temp={self.base_temperature}K"
        )
    
    def _create_coordinate_grids(self) -> None:
        """Create coordinate grids for efficient distance calculations."""
        x = np.arange(self.grid_width)
        y = np.arange(self.grid_height)
        self.x_grid, self.y_grid = np.meshgrid(x, y)
        
        # Pre-calculate distance from sun center
        self.distance_from_sun = np.sqrt(
            (self.x_grid - self.sun_center_x) ** 2 + 
            (self.y_grid - self.sun_center_y) ** 2
        )
    
    def _calculate_time_factor(self, current_time: datetime) -> float:
        """
        Calculate temperature variation factor based on time of day.
        
        Simulates the sun being cooler in morning/evening and hotter at midday.
        
        Args:
            current_time: Current datetime
            
        Returns:
            Factor between 0.8 and 1.2 representing temperature scaling
        """
        hour = current_time.hour
        minute = current_time.minute
        
        # Convert to decimal hours (0-24)
        decimal_hour = hour + minute / 60.0
        
        # Create sinusoidal variation with peak at noon (12:00)
        # Factor ranges from 0.8 (dawn/dusk) to 1.2 (noon)
        angle = 2 * np.pi * (decimal_hour - 12) / 24  # Shift so peak is at noon (12:00)
        factor = 1.0 + 0.2 * np.cos(angle)  # cos(0) = 1.0 when decimal_hour = 12
        
        return factor
    
    def _generate_sun_disc(self, time_factor: float) -> np.ndarray:
        """
        Generate the main sun disc with realistic temperature gradients.
        
        Args:
            time_factor: Time-based temperature scaling factor
            
        Returns:
            2D array of temperature values for the sun disc
        """
        # Create base sun disc with radial temperature gradient
        temperatures = np.zeros((self.grid_height, self.grid_width))
        
        # Sun disc - hotter in center, cooler at edges
        sun_mask = self.distance_from_sun <= self.sun_radius
        
        # Radial temperature gradient (hotter in center)
        normalized_distance = self.distance_from_sun / self.sun_radius
        
        # Temperature drops from center to edge with realistic solar limb darkening
        temperature_profile = np.where(
            sun_mask,
            self.base_temperature * time_factor * (1.0 - 0.3 * normalized_distance ** 2),
            293.15  # Ambient space temperature (~20°C)
        )
        
        temperatures = temperature_profile
        
        return temperatures
    
    def _add_atmospheric_effects(self, temperatures: np.ndarray) -> np.ndarray:
        """
        Add atmospheric effects like turbulence and scintillation.
        
        Args:
            temperatures: Base temperature array
            
        Returns:
            Temperature array with atmospheric effects added
        """
        # Update atmospheric turbulence slowly over time
        self._noise_phase += 0.05
        
        # Create dynamic atmospheric noise
        turbulence = (
            np.sin(self._atmospheric_turbulence * 10 + self._noise_phase) * 
            settings.noise_amplitude * 0.5
        )
        
        # Add small-scale scintillation effects
        scintillation = np.random.normal(0, settings.noise_amplitude * 0.3, temperatures.shape)
        
        # Apply atmospheric effects only to the sun disc area
        sun_mask = self.distance_from_sun <= self.sun_radius
        atmospheric_effects = np.where(sun_mask, turbulence + scintillation, 0)
        
        return temperatures + atmospheric_effects
    
    def _simulate_solar_phenomena(self, temperatures: np.ndarray, anomaly_chance: float = 0.01) -> np.ndarray:
        """
        Simulate persistent solar phenomena like sunspots and solar flares.
        
        This method manages a collection of active anomalies that persist across
        multiple frames, creating more realistic solar behavior.
        
        Args:
            temperatures: Base temperature array
            anomaly_chance: Probability of generating a new anomaly this frame
            
        Returns:
            Temperature array with active solar phenomena applied
        """
        current_time = time.time()
        
        # Update existing anomalies - reduce their remaining frames
        self._update_existing_anomalies()
        
        # Check if we should create new anomalies (limit to one new anomaly per few seconds)
        time_since_last_check = current_time - self.last_anomaly_check
        if time_since_last_check >= 2.0 and np.random.random() < anomaly_chance:
            self._create_new_anomaly()
            self.last_anomaly_check = current_time
        
        # Apply all active anomalies to temperature grid
        self._apply_active_anomalies(temperatures)
        
        self.frame_count += 1
        return temperatures
    
    def _update_existing_anomalies(self) -> None:
        """Update existing anomalies and remove expired ones."""
        # Decrement remaining frames for all active anomalies
        for anomaly in self.active_anomalies:
            anomaly.remaining_frames -= 1
        
        # Remove expired anomalies
        initial_count = len(self.active_anomalies)
        self.active_anomalies = [
            anomaly for anomaly in self.active_anomalies 
            if anomaly.remaining_frames > 0
        ]
        
        removed_count = initial_count - len(self.active_anomalies)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} expired solar anomalies")
    
    def _create_new_anomaly(self) -> None:
        """Create a new persistent solar anomaly."""
        # Determine anomaly type based on weighted probabilities
        if np.random.random() < 0.6:  # 60% chance of sunspot
            anomaly_type = "sunspot"
            intensity = np.random.uniform(0.3, 0.7)  # Cooler (reduction factor)
            size = np.random.uniform(8, 20)
        elif np.random.random() < 0.8:  # 20% chance of solar flare
            anomaly_type = "flare"
            intensity = np.random.uniform(1.3, 1.8)  # Hotter (multiplication factor)
            size = np.random.uniform(5, 15)
        else:  # 20% chance of prominence
            anomaly_type = "prominence"
            intensity = np.random.uniform(1.1, 1.4)  # Moderately hotter
            size = np.random.uniform(10, 25)
        
        # Random position within sun disc (80% of radius to avoid edge effects)
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, self.sun_radius * 0.8)
        
        position = (
            int(self.sun_center_y + distance * np.sin(angle)),
            int(self.sun_center_x + distance * np.cos(angle))
        )
        
        # Random duration between 5-50 frames as requested
        duration = np.random.randint(5, 51)
        
        # Create the persistent anomaly
        anomaly = PersistentAnomaly(
            position=position,
            anomaly_type=anomaly_type,
            intensity=intensity,
            size=size,
            total_duration=duration,
            remaining_frames=duration,
            creation_time=time.time()
        )
        
        self.active_anomalies.append(anomaly)
        
        logger.info(
            f"Created new {anomaly_type} at {position} "
            f"(size={size:.1f}, intensity={intensity:.2f}, duration={duration} frames)"
        )
    
    def _apply_active_anomalies(self, temperatures: np.ndarray) -> None:
        """Apply all active anomalies to the temperature grid."""
        for anomaly in self.active_anomalies:
            self._apply_single_anomaly(temperatures, anomaly)
    
    def _apply_single_anomaly(self, temperatures: np.ndarray, anomaly: PersistentAnomaly) -> None:
        """Apply a single anomaly to the temperature grid."""
        row, col = anomaly.position
        current_intensity = anomaly.current_intensity
        
        # Create circular mask for the anomaly
        anomaly_mask = (
            (self.x_grid - col) ** 2 + (self.y_grid - row) ** 2
        ) <= anomaly.size ** 2
        
        # Apply temperature modification based on anomaly type
        if anomaly.anomaly_type == "sunspot":
            # Sunspots are cooler - subtract temperature
            temperature_change = current_intensity * np.random.uniform(800, 1200)
            temperatures[anomaly_mask] -= temperature_change
        
        elif anomaly.anomaly_type == "flare":
            # Solar flares are hotter - add temperature
            temperature_change = (current_intensity - 1.0) * np.random.uniform(1000, 1800)
            temperatures[anomaly_mask] += temperature_change
        
        elif anomaly.anomaly_type == "prominence":
            # Prominences are moderately hotter with some variability
            temperature_change = (current_intensity - 1.0) * np.random.uniform(600, 1200)
            # Add some spatial variation for more realistic appearance
            spatial_variation = np.random.normal(1.0, 0.1, temperatures[anomaly_mask].shape)
            temperatures[anomaly_mask] += temperature_change * spatial_variation
    
    def force_anomaly_creation(self) -> None:
        """Force the creation of a new anomaly immediately."""
        self._create_new_anomaly()
        logger.info("Forced creation of new solar anomaly")
    
    def get_anomaly_statistics(self) -> dict:
        """
        Get statistics about currently active anomalies.
        
        Returns:
            Dictionary containing anomaly counts and details
        """
        if not self.active_anomalies:
            return {
                "total_count": 0,
                "by_type": {},
                "oldest_anomaly_age": 0,
                "anomalies": []
            }
        
        # Count anomalies by type
        type_counts = {}
        anomaly_details = []
        
        for anomaly in self.active_anomalies:
            # Count by type
            if anomaly.anomaly_type not in type_counts:
                type_counts[anomaly.anomaly_type] = 0
            type_counts[anomaly.anomaly_type] += 1
            
            # Collect details for web interface
            anomaly_details.append({
                "type": anomaly.anomaly_type,
                "position": anomaly.position,
                "intensity": round(anomaly.current_intensity, 3),
                "size": round(anomaly.size, 1),
                "remaining_frames": anomaly.remaining_frames,
                "total_duration": anomaly.total_duration,
                "age_progress": round(1.0 - (anomaly.remaining_frames / anomaly.total_duration), 2)
            })
        
        # Find oldest anomaly
        oldest_age = max(
            anomaly.total_duration - anomaly.remaining_frames 
            for anomaly in self.active_anomalies
        )
        
        return {
            "total_count": len(self.active_anomalies),
            "by_type": type_counts,
            "oldest_anomaly_age": oldest_age,
            "anomalies": anomaly_details
        }

    def generate_frame(self, anomaly_chance: float = 0.01, current_time: Optional[datetime] = None) -> RadiometricFrame:
        """
        Generate a single frame of radiometric data.
        
        Args:
            anomaly_chance: Probability of generating solar phenomena
            current_time: Optional custom time for simulation (uses real time if None)
            
        Returns:
            RadiometricFrame containing the simulated temperature data
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Calculate time-based temperature variations
        time_factor = self._calculate_time_factor(current_time)
        
        # Generate base sun disc
        temperatures = self._generate_sun_disc(time_factor)
        
        # Add atmospheric effects
        temperatures = self._add_atmospheric_effects(temperatures)
        
        # Possibly add solar phenomena
        temperatures = self._simulate_solar_phenomena(temperatures, anomaly_chance)
        
        # Ensure temperatures don't go below absolute zero
        temperatures = np.maximum(temperatures, 1.0)
        
        # Create metadata including anomaly information
        anomaly_stats = self.get_anomaly_statistics()
        metadata = {
            "time_factor": time_factor,
            "sun_position": (self.sun_center_x, self.sun_center_y),
            "sun_radius": self.sun_radius,
            "base_temperature": self.base_temperature,
            "grid_size": (self.grid_width, self.grid_height),
            "active_anomalies": anomaly_stats,
            "frame_count": self.frame_count,
        }
        
        return RadiometricFrame(
            timestamp=current_time,
            data=temperatures,
            width=self.grid_width,
            height=self.grid_height,
            metadata=metadata,
        )