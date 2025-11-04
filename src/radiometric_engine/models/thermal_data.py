"""
Data models for radiometric measurements and thermal data.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class RadiometricFrame:
    """
    Represents a single frame of radiometric data from a thermographic camera.
    
    Attributes:
        timestamp: When the frame was captured
        data: 2D numpy array of temperature values in Kelvin
        width: Frame width in pixels
        height: Frame height in pixels
        metadata: Optional metadata about capture conditions
    """
    timestamp: datetime
    data: np.ndarray
    width: int
    height: int
    metadata: Optional[dict] = None
    
    def __post_init__(self) -> None:
        """Validate the radiometric frame data."""
        if self.data.shape != (self.height, self.width):
            raise ValueError(
                f"Data shape {self.data.shape} doesn't match dimensions "
                f"({self.height}, {self.width})"
            )
        if np.any(self.data < 0):
            raise ValueError("Temperature values cannot be negative (Kelvin scale)")


@dataclass
class ThermalAnomaly:
    """
    Represents a detected thermal anomaly in radiometric data.
    
    Attributes:
        timestamp: When the anomaly was detected
        location: (x, y) coordinates of the anomaly center
        severity: Severity score (0.0 to 1.0)
        temperature_deviation: How much the temperature deviates from baseline
        area_pixels: Size of the anomalous area in pixels
        description: Human-readable description of the anomaly
    """
    timestamp: datetime
    location: Tuple[int, int]
    severity: float
    temperature_deviation: float
    area_pixels: int
    description: str
    
    def __post_init__(self) -> None:
        """Validate anomaly data."""
        if not (0.0 <= self.severity <= 1.0):
            raise ValueError("Severity must be between 0.0 and 1.0")
        if self.area_pixels <= 0:
            raise ValueError("Area must be positive")


@dataclass
class BaselineData:
    """
    Represents baseline thermal data for a specific scene and conditions.
    
    Attributes:
        timestamp: When the baseline was captured
        data: 2D numpy array of baseline temperature values
        conditions: Environmental conditions when captured
        time_of_day: Hour of day (0-23) when captured
        description: Description of the baseline conditions
    """
    timestamp: datetime
    data: np.ndarray
    conditions: dict
    time_of_day: int
    description: str
    
    def __post_init__(self) -> None:
        """Validate baseline data."""
        if not (0 <= self.time_of_day <= 23):
            raise ValueError("Time of day must be between 0 and 23")
        if np.any(self.data < 0):
            raise ValueError("Baseline temperatures cannot be negative")


@dataclass
class SimulationParameters:
    """
    Parameters for controlling the radiometric simulation.
    
    Attributes:
        grid_size: (width, height) of the simulation grid
        sun_position: (x, y) coordinates of the sun center
        sun_radius: Radius of the sun in pixels
        base_temperature: Base temperature of the sun in Kelvin
        noise_level: Amount of random noise to add
        time_factor: Time-based temperature variation factor
        anomaly_probability: Probability of generating anomalies per frame
    """
    grid_size: Tuple[int, int]
    sun_position: Tuple[int, int]
    sun_radius: int
    base_temperature: float
    noise_level: float
    time_factor: float
    anomaly_probability: float = 0.01
    
    def __post_init__(self) -> None:
        """Validate simulation parameters."""
        if any(dim <= 0 for dim in self.grid_size):
            raise ValueError("Grid dimensions must be positive")
        if self.sun_radius <= 0:
            raise ValueError("Sun radius must be positive")
        if self.base_temperature <= 0:
            raise ValueError("Base temperature must be positive")
        if not (0.0 <= self.anomaly_probability <= 1.0):
            raise ValueError("Anomaly probability must be between 0.0 and 1.0")