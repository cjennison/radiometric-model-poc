"""
Anomaly Detection Engine for Radiometric Data

This module implements real-time anomaly detection by comparing live radiometric data
against established baseline patterns. It identifies regions where temperature patterns
deviate significantly from historical norms.

Key Features:
- Real-time comparison against baseline database
- 5-minute time bucket matching
- Regional anomaly clustering (not just single pixels)
- Configurable statistical thresholds
- Multiple anomaly detection algorithms
"""

import sys
import os
import sqlite3
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.radiometric_engine.services.baseline_manager import BaselineDataManager
from src.radiometric_engine.models.thermal_data import RadiometricFrame, ThermalAnomaly

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    TEMPERATURE_SPIKE = "temperature_spike"
    TEMPERATURE_DROP = "temperature_drop"
    REGIONAL_CLUSTER = "regional_cluster"
    TEMPORAL_DEVIATION = "temporal_deviation"
    STATISTICAL_OUTLIER = "statistical_outlier"


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""
    LOW = "low"          # 2-3 standard deviations
    MEDIUM = "medium"    # 3-4 standard deviations  
    HIGH = "high"        # 4-5 standard deviations
    CRITICAL = "critical" # >5 standard deviations


@dataclass
class DetectedAnomaly:
    """Container for a detected anomaly."""
    anomaly_id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    region_center: Tuple[int, int]
    affected_pixels: List[Tuple[int, int]]
    confidence_score: float
    temperature_deviation: float
    baseline_mean: float
    observed_mean: float
    description: str
    metadata: Dict[str, Any]


class AnomalyDetectionEngine:
    """
    Real-time anomaly detection engine that compares live data against baseline patterns.
    """
    
    def __init__(self, 
                 baseline_db_path: str = "baseline_data.db",
                 detection_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the anomaly detection engine.
        
        Args:
            baseline_db_path: Path to baseline database
            detection_config: Configuration parameters for detection algorithms
        """
        self.baseline_manager = BaselineDataManager(bucket_minutes=5, db_path=baseline_db_path)
        
        # Default detection configuration
        self.config = {
            'std_dev_threshold': 2.5,      # Standard deviations for anomaly threshold
            'cluster_min_size': 5,         # Minimum pixels for regional anomaly
            'cluster_max_distance': 3,     # Maximum distance between pixels in cluster
            'confidence_threshold': 0.7,   # Minimum confidence for reporting
            'temporal_window_minutes': 15, # Window for temporal anomaly detection
            'region_analysis_enabled': True,
            'single_pixel_detection': False  # Usually too noisy
        }
        
        # Update with user config
        if detection_config:
            self.config.update(detection_config)
        
        # Detection state
        self.recent_detections: List[DetectedAnomaly] = []
        self.detection_statistics = {
            'total_frames_analyzed': 0,
            'anomalies_detected': 0,
            'false_positive_rate': 0.0,
            'last_detection_time': None
        }
        
        logger.info(f"Anomaly detection engine initialized with config: {self.config}")
    
    def analyze_frame(self, frame: RadiometricFrame) -> List[DetectedAnomaly]:
        """
        Analyze a single frame for anomalies against baseline data.
        
        Args:
            frame: RadiometricFrame to analyze
            
        Returns:
            List of detected anomalies
        """
        self.detection_statistics['total_frames_analyzed'] += 1
        detected_anomalies = []
        
        try:
            # Get current time bucket
            bucket_id = self.baseline_manager.get_time_bucket_id(frame.timestamp)
            logger.info(f"Analyzing frame for bucket {bucket_id} at time {frame.timestamp}")
            
            # Get baseline data for this time bucket
            baseline_data = self._get_baseline_for_bucket(bucket_id)
            if not baseline_data:
                logger.warning(f"No baseline data available for bucket {bucket_id}")
                return []
            
            logger.info(f"Found baseline data for {len(baseline_data)} grid points")
            
            # Convert frame data to analysis format
            current_temps = frame.data
            height, width = current_temps.shape
            
            # 1. Statistical Outlier Detection
            statistical_anomalies = self._detect_statistical_outliers(
                current_temps, baseline_data, frame.timestamp
            )
            detected_anomalies.extend(statistical_anomalies)
            
            # 2. Regional Cluster Detection
            if self.config['region_analysis_enabled']:
                cluster_anomalies = self._detect_regional_clusters(
                    current_temps, baseline_data, frame.timestamp
                )
                detected_anomalies.extend(cluster_anomalies)
            
            # 3. Temporal Deviation Detection
            temporal_anomalies = self._detect_temporal_deviations(
                current_temps, baseline_data, frame.timestamp
            )
            detected_anomalies.extend(temporal_anomalies)
            
            # Update detection statistics
            if detected_anomalies:
                self.detection_statistics['anomalies_detected'] += len(detected_anomalies)
                self.detection_statistics['last_detection_time'] = frame.timestamp
                
                # Store recent detections (keep last 100)
                self.recent_detections.extend(detected_anomalies)
                if len(self.recent_detections) > 100:
                    self.recent_detections = self.recent_detections[-100:]
            
            logger.debug(f"Frame analysis complete: {len(detected_anomalies)} anomalies detected")
            return detected_anomalies
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return []
    
    def detect_anomalies(self, temperature_data: np.ndarray, timestamp: datetime) -> List[DetectedAnomaly]:
        """
        Convenience method to detect anomalies from raw temperature data.
        
        Args:
            temperature_data: 2D numpy array of temperature data
            timestamp: Timestamp for the data
            
        Returns:
            List of detected anomalies
        """
        # Create a temporary RadiometricFrame
        height, width = temperature_data.shape
        frame = RadiometricFrame(
            timestamp=timestamp,
            data=temperature_data,
            width=width,
            height=height,
            metadata={"source": "test_data"}
        )
        return self.analyze_frame(frame)
    
    def _get_baseline_for_bucket(self, bucket_id: int) -> Optional[Dict[Tuple[int, int], Dict[str, float]]]:
        """
        Get baseline data for a specific time bucket.
        
        Args:
            bucket_id: Time bucket ID
            
        Returns:
            Dictionary mapping (x, y) coordinates to baseline statistics
        """
        try:
            with sqlite3.connect(self.baseline_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT grid_x, grid_y, mean_temp, std_dev, min_temp, max_temp, sample_count
                    FROM baseline_data
                    WHERE bucket_id = ?
                """, (bucket_id,))
                
                rows = cursor.fetchall()
                logger.info(f"Found {len(rows)} baseline records for bucket {bucket_id}")
                
                baseline_data = {}
                for row in rows:
                    grid_x, grid_y, mean_temp, std_dev, min_temp, max_temp, sample_count = row
                    baseline_data[(grid_x, grid_y)] = {
                        'mean': mean_temp,
                        'std_dev': std_dev,
                        'min': min_temp,
                        'max': max_temp,
                        'sample_count': sample_count
                    }
                
                return baseline_data if baseline_data else None
                
        except Exception as e:
            logger.error(f"Error fetching baseline data for bucket {bucket_id}: {e}")
            return None
    
    def _detect_statistical_outliers(self, 
                                   current_temps: np.ndarray, 
                                   baseline_data: Dict[Tuple[int, int], Dict[str, float]],
                                   timestamp: datetime) -> List[DetectedAnomaly]:
        """
        Detect statistical outliers based on standard deviation thresholds.
        
        Args:
            current_temps: Current temperature grid
            baseline_data: Baseline statistics for comparison
            timestamp: Current timestamp
            
        Returns:
            List of detected statistical anomalies
        """
        anomalies = []
        height, width = current_temps.shape
        outlier_pixels = []
        
        for y in range(height):
            for x in range(width):
                if (x, y) not in baseline_data:
                    continue
                
                current_temp = current_temps[y, x]
                baseline = baseline_data[(x, y)]
                
                # Skip if insufficient baseline data
                if baseline['sample_count'] < 5:
                    continue
                
                # Use minimum std_dev to account for natural variation
                # This prevents overly sensitive detection due to artificially low std_dev
                min_std_dev = max(baseline['std_dev'], baseline['mean'] * 0.10)  # 10% of mean as minimum
                
                # Calculate z-score (number of standard deviations from mean)
                z_score = abs(current_temp - baseline['mean']) / min_std_dev
                
                if z_score >= self.config['std_dev_threshold']:
                    outlier_pixels.append({
                        'position': (x, y),
                        'z_score': z_score,
                        'current_temp': current_temp,
                        'baseline_mean': baseline['mean'],
                        'deviation': current_temp - baseline['mean']
                    })
        
        # Group outliers into anomalies (to avoid single-pixel noise)
        if outlier_pixels and len(outlier_pixels) >= self.config['cluster_min_size']:
            # Create anomaly for statistical outliers
            avg_z_score = np.mean([p['z_score'] for p in outlier_pixels])
            severity = self._calculate_severity(avg_z_score)
            
            # Find center of outlier region
            center_x = int(np.mean([p['position'][0] for p in outlier_pixels]))
            center_y = int(np.mean([p['position'][1] for p in outlier_pixels]))
            
            anomaly = DetectedAnomaly(
                anomaly_id=f"stat_{timestamp.strftime('%Y%m%d_%H%M%S')}_{center_x}_{center_y}",
                timestamp=timestamp,
                anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                severity=severity,
                region_center=(center_x, center_y),
                affected_pixels=[p['position'] for p in outlier_pixels],
                confidence_score=min(avg_z_score / 5.0, 1.0),  # Normalize to 0-1
                temperature_deviation=np.mean([p['deviation'] for p in outlier_pixels]),
                baseline_mean=np.mean([p['baseline_mean'] for p in outlier_pixels]),
                observed_mean=np.mean([p['current_temp'] for p in outlier_pixels]),
                description=f"Statistical outlier region with {len(outlier_pixels)} pixels, avg z-score: {avg_z_score:.2f}",
                metadata={
                    'outlier_count': len(outlier_pixels),
                    'avg_z_score': avg_z_score,
                    'max_z_score': max(p['z_score'] for p in outlier_pixels)
                }
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_regional_clusters(self, 
                                current_temps: np.ndarray, 
                                baseline_data: Dict[Tuple[int, int], Dict[str, float]],
                                timestamp: datetime) -> List[DetectedAnomaly]:
        """
        Detect regional clusters of anomalous temperatures.
        
        Args:
            current_temps: Current temperature grid
            baseline_data: Baseline statistics for comparison
            timestamp: Current timestamp
            
        Returns:
            List of detected regional cluster anomalies
        """
        # This is a more advanced algorithm that looks for spatially connected
        # regions of anomalous temperatures (like solar flares or large sunspots)
        anomalies = []
        height, width = current_temps.shape
        
        # Create anomaly mask
        anomaly_mask = np.zeros((height, width), dtype=bool)
        deviation_grid = np.zeros((height, width), dtype=float)
        
        for y in range(height):
            for x in range(width):
                if (x, y) not in baseline_data:
                    continue
                
                baseline = baseline_data[(x, y)]
                if baseline['sample_count'] < 5:
                    continue
                
                # Use minimum std_dev to account for natural variation
                # This prevents overly sensitive detection due to artificially low std_dev
                min_std_dev = max(baseline['std_dev'], baseline['mean'] * 0.10)  # 10% of mean as minimum
                
                current_temp = current_temps[y, x]
                z_score = abs(current_temp - baseline['mean']) / min_std_dev
                
                if z_score >= self.config['std_dev_threshold']:
                    anomaly_mask[y, x] = True
                    deviation_grid[y, x] = current_temp - baseline['mean']
        
        # Find connected components (clusters)
        clusters = self._find_connected_components(anomaly_mask)
        
        for cluster in clusters:
            if len(cluster) >= self.config['cluster_min_size']:
                # Calculate cluster statistics
                cluster_temps = [current_temps[y, x] for x, y in cluster]
                cluster_deviations = [deviation_grid[y, x] for x, y in cluster]
                
                center_x = int(np.mean([x for x, y in cluster]))
                center_y = int(np.mean([y for x, y in cluster]))
                
                avg_deviation = np.mean(cluster_deviations)
                severity = self._calculate_severity(abs(avg_deviation) / 100)  # Rough normalization
                
                # Determine anomaly type based on deviation direction
                if avg_deviation > 0:
                    anomaly_type = AnomalyType.TEMPERATURE_SPIKE
                    description = f"Temperature spike cluster: {len(cluster)} pixels, +{avg_deviation:.1f}K above baseline"
                else:
                    anomaly_type = AnomalyType.TEMPERATURE_DROP
                    description = f"Temperature drop cluster: {len(cluster)} pixels, {avg_deviation:.1f}K below baseline"
                
                anomaly = DetectedAnomaly(
                    anomaly_id=f"cluster_{timestamp.strftime('%Y%m%d_%H%M%S')}_{center_x}_{center_y}",
                    timestamp=timestamp,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    region_center=(center_x, center_y),
                    affected_pixels=cluster,
                    confidence_score=min(len(cluster) / 50.0, 1.0),  # Larger clusters = higher confidence
                    temperature_deviation=avg_deviation,
                    baseline_mean=0,  # Would need to calculate from baseline data
                    observed_mean=np.mean(cluster_temps),
                    description=description,
                    metadata={
                        'cluster_size': len(cluster),
                        'cluster_area': len(cluster),
                        'max_deviation': max(abs(d) for d in cluster_deviations)
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_temporal_deviations(self, 
                                  current_temps: np.ndarray, 
                                  baseline_data: Dict[Tuple[int, int], Dict[str, float]],
                                  timestamp: datetime) -> List[DetectedAnomaly]:
        """
        Detect temporal deviations by comparing with recent time buckets.
        
        Args:
            current_temps: Current temperature grid
            baseline_data: Baseline statistics for comparison
            timestamp: Current timestamp
            
        Returns:
            List of detected temporal anomalies
        """
        # This would compare current data with recent time buckets to detect
        # trends or sudden changes over time. For now, placeholder implementation.
        return []
    
    def _find_connected_components(self, mask: np.ndarray) -> List[List[Tuple[int, int]]]:
        """
        Find connected components in a binary mask using flood fill.
        
        Args:
            mask: Binary mask where True indicates anomalous pixels
            
        Returns:
            List of connected components, each containing pixel coordinates
        """
        height, width = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        components = []
        
        def flood_fill(start_x, start_y):
            """Flood fill to find connected component."""
            stack = [(start_x, start_y)]
            component = []
            
            while stack:
                x, y = stack.pop()
                
                if (x < 0 or x >= width or y < 0 or y >= height or 
                    visited[y, x] or not mask[y, x]):
                    continue
                
                visited[y, x] = True
                component.append((x, y))
                
                # Add 8-connected neighbors
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        stack.append((x + dx, y + dy))
            
            return component
        
        # Find all connected components
        for y in range(height):
            for x in range(width):
                if mask[y, x] and not visited[y, x]:
                    component = flood_fill(x, y)
                    if component:
                        components.append(component)
        
        return components
    
    def _calculate_severity(self, z_score: float) -> AnomalySeverity:
        """
        Calculate anomaly severity based on z-score or other metrics.
        
        Args:
            z_score: Standard deviations from baseline
            
        Returns:
            AnomalySeverity level
        """
        if z_score >= 5.0:
            return AnomalySeverity.CRITICAL
        elif z_score >= 4.0:
            return AnomalySeverity.HIGH
        elif z_score >= 3.0:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """
        Get current detection statistics.
        
        Returns:
            Dictionary of detection statistics
        """
        recent_count = len([a for a in self.recent_detections 
                           if a.timestamp > datetime.now() - timedelta(hours=1)])
        
        severity_counts = {}
        for severity in AnomalySeverity:
            severity_counts[severity.value] = len([a for a in self.recent_detections 
                                                  if a.severity == severity])
        
        return {
            **self.detection_statistics,
            'recent_detections_1h': recent_count,
            'severity_distribution': severity_counts,
            'detection_rate_per_hour': recent_count,
            'active_anomalies': len(self.recent_detections)
        }
    
    def clear_old_detections(self, hours_to_keep: int = 24) -> None:
        """
        Clear detections older than specified hours.
        
        Args:
            hours_to_keep: Number of hours of detections to retain
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_to_keep)
        self.recent_detections = [a for a in self.recent_detections 
                                 if a.timestamp > cutoff_time]
        
        logger.info(f"Cleared old detections, {len(self.recent_detections)} remaining")


def create_detection_engine(config_overrides: Optional[Dict[str, Any]] = None) -> AnomalyDetectionEngine:
    """
    Factory function to create a configured anomaly detection engine.
    
    Args:
        config_overrides: Optional configuration overrides
        
    Returns:
        Configured AnomalyDetectionEngine instance
    """
    default_config = {
        'std_dev_threshold': 2.5,
        'cluster_min_size': 5,
        'cluster_max_distance': 3,
        'confidence_threshold': 0.7,
        'region_analysis_enabled': True,
        'single_pixel_detection': False
    }
    
    if config_overrides:
        default_config.update(config_overrides)
    
    return AnomalyDetectionEngine(detection_config=default_config)