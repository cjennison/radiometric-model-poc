"""
Configuration settings for the radiometric engine.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    """Application settings loaded from environment variables."""
    
    # Simulation parameters
    grid_width: int = int(os.getenv("GRID_WIDTH", "150"))
    grid_height: int = int(os.getenv("GRID_HEIGHT", "150"))
    update_frequency_hz: float = float(os.getenv("UPDATE_FREQUENCY_HZ", "10.0"))
    
    # Sun simulation parameters
    sun_base_temperature: float = float(os.getenv("SUN_BASE_TEMP", "5778.0"))  # Kelvin
    sun_radius_pixels: int = int(os.getenv("SUN_RADIUS_PIXELS", "60"))
    noise_amplitude: float = float(os.getenv("NOISE_AMPLITUDE", "50.0"))
    
    # Time-based variations
    temperature_variation_range: float = float(os.getenv("TEMP_VARIATION_RANGE", "200.0"))
    
    # Flask web application
    flask_host: str = os.getenv("FLASK_HOST", "localhost")
    flask_port: int = int(os.getenv("FLASK_PORT", "5000"))
    flask_debug: bool = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///radiometric.db")
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.grid_width <= 0 or self.grid_height <= 0:
            raise ValueError("Grid dimensions must be positive")
        if self.update_frequency_hz <= 0:
            raise ValueError("Update frequency must be positive")


# Global settings instance
settings = Settings()