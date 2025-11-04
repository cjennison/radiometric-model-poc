"""
Sun simulation service for generating realistic radiometric data.

This module simulates a thermographic camera pointed at the sun, generating
realistic temperature distributions with time-based variations and atmospheric effects.
"""

import logging
import time
from datetime import datetime
from typing import Tuple, Optional
import numpy as np

from ..models import RadiometricFrame, SimulationParameters
from ..config import settings

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
        angle = 2 * np.pi * (decimal_hour - 6) / 24  # Shift so peak is at noon
        factor = 1.0 + 0.2 * np.cos(angle)
        
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
        Simulate solar phenomena like sunspots and solar flares.
        
        Args:
            temperatures: Base temperature array
            anomaly_chance: Probability of generating an anomaly this frame
            
        Returns:
            Temperature array with potential solar phenomena
        """
        if np.random.random() < anomaly_chance:
            # Generate a random solar phenomenon
            if np.random.random() < 0.7:  # 70% chance of sunspot (cooler)
                self._add_sunspot(temperatures)
            else:  # 30% chance of solar flare (hotter)
                self._add_solar_flare(temperatures)
        
        return temperatures
    
    def _add_sunspot(self, temperatures: np.ndarray) -> None:
        """Add a cooler sunspot to the sun disc."""
        # Random position within sun disc
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, self.sun_radius * 0.8)
        
        spot_x = int(self.sun_center_x + distance * np.cos(angle))
        spot_y = int(self.sun_center_y + distance * np.sin(angle))
        
        # Sunspot parameters
        spot_radius = np.random.uniform(5, 15)
        temperature_reduction = np.random.uniform(500, 1000)  # Cooler by 500-1000K
        
        # Apply sunspot
        spot_mask = (
            (self.x_grid - spot_x) ** 2 + (self.y_grid - spot_y) ** 2
        ) <= spot_radius ** 2
        
        temperatures[spot_mask] -= temperature_reduction
        
        logger.info(f"Generated sunspot at ({spot_x}, {spot_y}) with radius {spot_radius:.1f}")
    
    def _add_solar_flare(self, temperatures: np.ndarray) -> None:
        """Add a hotter solar flare to the sun disc."""
        # Random position within sun disc
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, self.sun_radius * 0.9)
        
        flare_x = int(self.sun_center_x + distance * np.cos(angle))
        flare_y = int(self.sun_center_y + distance * np.sin(angle))
        
        # Solar flare parameters
        flare_width = np.random.uniform(8, 20)
        flare_height = np.random.uniform(3, 8)
        temperature_increase = np.random.uniform(800, 1500)  # Hotter by 800-1500K
        
        # Create elongated flare shape
        flare_mask = (
            ((self.x_grid - flare_x) / flare_width) ** 2 + 
            ((self.y_grid - flare_y) / flare_height) ** 2
        ) <= 1
        
        temperatures[flare_mask] += temperature_increase
        
        logger.info(f"Generated solar flare at ({flare_x}, {flare_y}) "
                   f"with size {flare_width:.1f}x{flare_height:.1f}")
    
    def generate_frame(self, anomaly_chance: float = 0.01) -> RadiometricFrame:
        """
        Generate a single frame of radiometric data.
        
        Args:
            anomaly_chance: Probability of generating solar phenomena
            
        Returns:
            RadiometricFrame containing the simulated temperature data
        """
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
        
        # Create metadata
        metadata = {
            "time_factor": time_factor,
            "sun_position": (self.sun_center_x, self.sun_center_y),
            "sun_radius": self.sun_radius,
            "base_temperature": self.base_temperature,
            "grid_size": (self.grid_width, self.grid_height),
        }
        
        return RadiometricFrame(
            timestamp=current_time,
            data=temperatures,
            width=self.grid_width,
            height=self.grid_height,
            metadata=metadata,
        )