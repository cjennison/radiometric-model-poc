"""
Baseline data management system for radiometric anomaly detection.

This module provides functionality to capture, store, and analyze baseline
thermographic data patterns for anomaly detection. Data is bucketed by time
intervals and stored with statistical analysis for each grid point.
"""

import logging
import sqlite3
import os
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import json

logger = logging.getLogger(__name__)


@dataclass
class TimeBucket:
    """Represents a time bucket for baseline data collection."""
    bucket_id: int
    start_time: str  # HH:MM format
    end_time: str    # HH:MM format
    sample_count: int = 0


@dataclass
class BaselineGridPoint:
    """Baseline statistics for a single grid point."""
    grid_x: int
    grid_y: int
    mean_temp: float
    std_dev: float
    min_temp: float
    max_temp: float
    sample_count: int


class BaselineDataManager:
    """
    Manages baseline thermographic data collection and storage.
    
    This class handles:
    - Time bucketing (5-minute intervals)
    - Grid-based temperature data collection
    - Statistical analysis and storage
    - Baseline data retrieval for anomaly detection
    """
    
    def __init__(self, db_path: str = "baseline_data.db", bucket_minutes: int = 5):
        """
        Initialize the baseline data manager.
        
        Args:
            db_path: Path to SQLite database file
            bucket_minutes: Time bucket size in minutes
        """
        self.db_path = db_path
        self.bucket_minutes = bucket_minutes
        self.buckets_per_day = (24 * 60) // bucket_minutes  # 288 for 5-min buckets
        
        # Current data collection state
        self._current_bucket_id: Optional[int] = None
        self._bucket_accumulator: Dict[Tuple[int, int], List[float]] = {}
        self._collection_enabled = False
        self._current_session_id: Optional[int] = None
        self._frame_count = 0
        
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create time buckets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS baseline_buckets (
                    bucket_id INTEGER PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    sample_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create baseline data table for grid points
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS baseline_data (
                    bucket_id INTEGER,
                    grid_x INTEGER,
                    grid_y INTEGER,
                    mean_temp REAL,
                    std_dev REAL,
                    min_temp REAL,
                    max_temp REAL,
                    sample_count INTEGER,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (bucket_id, grid_x, grid_y),
                    FOREIGN KEY (bucket_id) REFERENCES baseline_buckets(bucket_id)
                )
            """)
            
            # Create collection sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collection_sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    frames_collected INTEGER DEFAULT 0,
                    description TEXT
                )
            """)
            
            conn.commit()
            
            # Initialize time buckets if they don't exist
            self._initialize_time_buckets()
    
    def _initialize_time_buckets(self) -> None:
        """Initialize all 288 time buckets for a 24-hour period."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if buckets already exist
            cursor.execute("SELECT COUNT(*) FROM baseline_buckets")
            if cursor.fetchone()[0] > 0:
                return  # Buckets already initialized
            
            # Create all time buckets
            for bucket_id in range(self.buckets_per_day):
                start_minutes = bucket_id * self.bucket_minutes
                end_minutes = start_minutes + self.bucket_minutes
                
                start_hour = start_minutes // 60
                start_min = start_minutes % 60
                end_hour = end_minutes // 60
                end_min = end_minutes % 60
                
                # Handle day rollover for last bucket
                if end_hour >= 24:
                    end_hour = 0
                    end_min = 0
                
                start_time = f"{start_hour:02d}:{start_min:02d}"
                end_time = f"{end_hour:02d}:{end_min:02d}"
                
                cursor.execute("""
                    INSERT INTO baseline_buckets (bucket_id, start_time, end_time)
                    VALUES (?, ?, ?)
                """, (bucket_id, start_time, end_time))
            
            conn.commit()
        
        logger.info(f"Initialized {self.buckets_per_day} time buckets")
    
    def get_time_bucket_id(self, current_time: datetime) -> int:
        """
        Get the time bucket ID for a given datetime.
        
        Args:
            current_time: Datetime to get bucket for
            
        Returns:
            Bucket ID (0 to buckets_per_day-1)
        """
        minutes_since_midnight = current_time.hour * 60 + current_time.minute
        bucket_id = minutes_since_midnight // self.bucket_minutes
        return bucket_id % self.buckets_per_day  # Handle edge cases
    
    def start_collection(self, description: str = "Baseline data collection") -> int:
        """
        Start a new baseline data collection session.
        
        Args:
            description: Description of the collection session
            
        Returns:
            Session ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO collection_sessions (start_time, description)
                VALUES (CURRENT_TIMESTAMP, ?)
            """, (description,))
            session_id = cursor.lastrowid
            conn.commit()
        
        self._collection_enabled = True
        self._current_bucket_id = None
        self._bucket_accumulator.clear()
        self._current_session_id = session_id
        self._frame_count = 0
        
        logger.info(f"Started baseline collection session {session_id}: {description}")
        return session_id
    
    def stop_collection(self, session_id: int) -> dict:
        """
        Stop the current collection session and finalize any pending data.
        
        Args:
            session_id: Session ID to stop
            
        Returns:
            Collection statistics
        """
        # Finalize any pending bucket data
        if self._bucket_accumulator:
            self._finalize_current_bucket()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Update final frame count and end time
            cursor.execute("""
                UPDATE collection_sessions 
                SET end_time = CURRENT_TIMESTAMP, frames_collected = ?
                WHERE session_id = ?
            """, (self._frame_count if self._current_session_id == session_id else 0, session_id))
            conn.commit()
            
            # Get session statistics
            cursor.execute("""
                SELECT start_time, end_time, frames_collected, description
                FROM collection_sessions WHERE session_id = ?
            """, (session_id,))
            session_data = cursor.fetchone()
        
        self._collection_enabled = False
        
        stats = {
            'session_id': session_id,
            'start_time': session_data[0] if session_data else None,
            'end_time': session_data[1] if session_data else None,
            'frames_collected': session_data[2] if session_data else 0,
            'description': session_data[3] if session_data else None
        }
        
        # Reset session tracking if this was the current session
        if self._current_session_id == session_id:
            self._current_session_id = None
            self._frame_count = 0
            self._collection_enabled = False
        
        logger.info(f"Stopped baseline collection session {session_id}")
        return stats
    
    def stop_all_collections(self) -> dict:
        """
        Stop all active baseline data collection sessions.
        
        Returns:
            Collection statistics for the most recent session
        """
        if not self._collection_enabled:
            return {'message': 'No active collections to stop'}
        
        # Finalize any pending bucket data
        if self._bucket_accumulator:
            self._finalize_current_bucket()
        
        # Stop collection
        self._collection_enabled = False
        
        # Update final frame count for current session
        if self._current_session_id:
            self._update_session_frame_count()
        
        # Reset session tracking
        self._current_session_id = None
        self._frame_count = 0
        
        # Get stats for the most recent session
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Update current session with final frame count
            if self._current_session_id:
                cursor.execute("""
                    UPDATE collection_sessions 
                    SET frames_collected = ?
                    WHERE session_id = ?
                """, (self._frame_count, self._current_session_id))
            
            # Update all active sessions to end now
            cursor.execute("""
                UPDATE collection_sessions 
                SET end_time = CURRENT_TIMESTAMP
                WHERE end_time IS NULL
            """)
            
            # Get the most recent session for stats
            cursor.execute("""
                SELECT start_time, end_time, frames_collected, description, session_id
                FROM collection_sessions 
                WHERE end_time IS NOT NULL
                ORDER BY session_id DESC
                LIMIT 1
            """)
            session_data = cursor.fetchone()
            
            conn.commit()
        
        stats = {
            'session_id': session_data[4] if session_data else None,
            'start_time': session_data[0] if session_data else None,
            'end_time': session_data[1] if session_data else None,
            'frames_collected': session_data[2] if session_data else 0,
            'description': session_data[3] if session_data else None,
            'message': 'All collections stopped'
        }
        
        logger.info("Stopped all baseline collection sessions")
        return stats

    def collect_frame_data(self, current_time: datetime, temperature_grid: np.ndarray) -> bool:
        """
        Collect temperature data from a frame for baseline building.
        
        Args:
            current_time: Timestamp of the frame
            temperature_grid: 2D numpy array of temperatures (150x150)
            
        Returns:
            True if data was collected, False if collection is disabled
        """
        if not self._collection_enabled:
            return False
        
        bucket_id = self.get_time_bucket_id(current_time)
        
        # Check if we need to finalize the previous bucket
        if self._current_bucket_id is not None and bucket_id != self._current_bucket_id:
            self._finalize_current_bucket()
        
        self._current_bucket_id = bucket_id
        
        # Accumulate data for each grid point
        height, width = temperature_grid.shape
        for x in range(width):
            for y in range(height):
                grid_point = (x, y)
                temperature = float(temperature_grid[y, x])  # Note: y,x indexing for numpy
                
                if grid_point not in self._bucket_accumulator:
                    self._bucket_accumulator[grid_point] = []
                
                self._bucket_accumulator[grid_point].append(temperature)
        
        # Increment frame count and update database periodically
        self._frame_count += 1
        if self._frame_count % 50 == 0:  # Update every 50 frames to avoid too many DB writes
            self._update_session_frame_count()
        
        return True
    
    def _update_session_frame_count(self) -> None:
        """Update the frame count for the current session in the database."""
        if not self._current_session_id:
            return
            
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE collection_sessions 
                SET frames_collected = ?
                WHERE session_id = ?
            """, (self._frame_count, self._current_session_id))
            conn.commit()
    
    def _finalize_current_bucket(self) -> None:
        """Finalize the current bucket by calculating statistics and storing to database."""
        if not self._bucket_accumulator or self._current_bucket_id is None:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for (grid_x, grid_y), temperatures in self._bucket_accumulator.items():
                if not temperatures:
                    continue
                
                # Calculate statistics
                temp_array = np.array(temperatures)
                mean_temp = float(np.mean(temp_array))
                std_dev = float(np.std(temp_array))
                min_temp = float(np.min(temp_array))
                max_temp = float(np.max(temp_array))
                sample_count = len(temperatures)
                
                # Check if data already exists for this grid point and bucket
                cursor.execute("""
                    SELECT mean_temp, std_dev, min_temp, max_temp, sample_count
                    FROM baseline_data 
                    WHERE bucket_id = ? AND grid_x = ? AND grid_y = ?
                """, (self._current_bucket_id, grid_x, grid_y))
                existing = cursor.fetchone()
                
                if existing:
                    # Accumulate with existing data using incremental statistics
                    old_mean, old_std, old_min, old_max, old_count = existing
                    new_count = old_count + sample_count
                    
                    # Incremental mean calculation
                    combined_mean = (old_mean * old_count + mean_temp * sample_count) / new_count
                    
                    # Combined min/max
                    combined_min = min(old_min, min_temp)
                    combined_max = max(old_max, max_temp)
                    
                    # For std dev, we'll use a simplified approach (not perfect but reasonable)
                    # In a production system, you'd store sum of squares for proper incremental std dev
                    old_variance = old_std ** 2
                    new_variance = std_dev ** 2
                    combined_variance = ((old_count * old_variance + sample_count * new_variance) / new_count)
                    combined_std = float(np.sqrt(combined_variance))
                    
                    # Update existing record
                    cursor.execute("""
                        UPDATE baseline_data 
                        SET mean_temp = ?, std_dev = ?, min_temp = ?, max_temp = ?, sample_count = ?
                        WHERE bucket_id = ? AND grid_x = ? AND grid_y = ?
                    """, (combined_mean, combined_std, combined_min, combined_max, new_count,
                          self._current_bucket_id, grid_x, grid_y))
                    
                    logger.debug(f"Updated existing data for bucket {self._current_bucket_id} point ({grid_x},{grid_y}): "
                               f"{old_count} + {sample_count} = {new_count} samples")
                else:
                    # Insert new baseline data
                    cursor.execute("""
                        INSERT INTO baseline_data 
                        (bucket_id, grid_x, grid_y, mean_temp, std_dev, min_temp, max_temp, sample_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (self._current_bucket_id, grid_x, grid_y, mean_temp, std_dev, 
                          min_temp, max_temp, sample_count))
                    
                    logger.debug(f"Inserted new data for bucket {self._current_bucket_id} point ({grid_x},{grid_y}): "
                               f"{sample_count} samples")
            
            # Update bucket sample count
            cursor.execute("""
                UPDATE baseline_buckets 
                SET sample_count = sample_count + 1, last_updated = CURRENT_TIMESTAMP
                WHERE bucket_id = ?
            """, (self._current_bucket_id,))
            
            conn.commit()
        
        logger.debug(f"Finalized bucket {self._current_bucket_id} with {len(self._bucket_accumulator)} grid points")
        self._bucket_accumulator.clear()
    
    def get_baseline_data(self, bucket_id: int) -> Optional[Dict[Tuple[int, int], BaselineGridPoint]]:
        """
        Get baseline data for a specific time bucket.
        
        Args:
            bucket_id: Time bucket ID
            
        Returns:
            Dictionary mapping (grid_x, grid_y) to BaselineGridPoint, or None if no data
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT grid_x, grid_y, mean_temp, std_dev, min_temp, max_temp, sample_count
                FROM baseline_data WHERE bucket_id = ?
            """, (bucket_id,))
            
            baseline_data = {}
            for row in cursor.fetchall():
                grid_x, grid_y, mean_temp, std_dev, min_temp, max_temp, sample_count = row
                baseline_data[(grid_x, grid_y)] = BaselineGridPoint(
                    grid_x=grid_x, grid_y=grid_y, mean_temp=mean_temp, std_dev=std_dev,
                    min_temp=min_temp, max_temp=max_temp, sample_count=sample_count
                )
        
        return baseline_data if baseline_data else None
    
    def get_baseline_stats(self) -> Dict[str, Any]:
        """
        Get overall baseline collection statistics.
        
        Returns:
            Dictionary with baseline statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get bucket statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_buckets,
                    COUNT(CASE WHEN sample_count > 0 THEN 1 END) as populated_buckets,
                    AVG(sample_count) as avg_samples_per_bucket,
                    MAX(sample_count) as max_samples_per_bucket
                FROM baseline_buckets
            """)
            bucket_stats = cursor.fetchone()
            
            # Get grid point statistics
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT bucket_id) as buckets_with_data,
                    COUNT(*) as total_grid_points,
                    AVG(sample_count) as avg_samples_per_point
                FROM baseline_data
            """)
            grid_stats = cursor.fetchone()
            
            # Get session statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_sessions,
                    SUM(frames_collected) as total_frames
                FROM collection_sessions
            """)
            session_stats = cursor.fetchone()
        
        return {
            'total_buckets': bucket_stats[0] if bucket_stats else 0,
            'populated_buckets': bucket_stats[1] if bucket_stats else 0,
            'avg_samples_per_bucket': bucket_stats[2] if bucket_stats else 0,
            'max_samples_per_bucket': bucket_stats[3] if bucket_stats else 0,
            'buckets_with_data': grid_stats[0] if grid_stats else 0,
            'total_grid_points': grid_stats[1] if grid_stats else 0,
            'avg_samples_per_point': grid_stats[2] if grid_stats else 0,
            'total_sessions': session_stats[0] if session_stats else 0,
            'total_frames': session_stats[1] if session_stats else 0,
            'completion_percentage': (bucket_stats[1] / bucket_stats[0] * 100) if bucket_stats and bucket_stats[0] > 0 else 0
        }
    
    def is_collecting(self) -> bool:
        """Check if baseline data collection is currently active."""
        return self._collection_enabled
    
    def clear_baseline_data(self) -> None:
        """Clear all baseline data (use with caution)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM baseline_data")
            cursor.execute("DELETE FROM collection_sessions")
            cursor.execute("UPDATE baseline_buckets SET sample_count = 0")
            conn.commit()
        
        logger.warning("All baseline data cleared")
    
    def export_baseline_data(self, export_path: str) -> None:
        """
        Export baseline data to JSON file for analysis or backup.
        
        Args:
            export_path: Path to save the exported data
        """
        data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'bucket_minutes': self.bucket_minutes,
                'buckets_per_day': self.buckets_per_day
            },
            'statistics': self.get_baseline_stats(),
            'buckets': []
        }
        
        # Export data for each bucket
        for bucket_id in range(self.buckets_per_day):
            baseline_data = self.get_baseline_data(bucket_id)
            if baseline_data:
                bucket_data = {
                    'bucket_id': bucket_id,
                    'grid_points': {
                        f"{x},{y}": {
                            'mean_temp': point.mean_temp,
                            'std_dev': point.std_dev,
                            'min_temp': point.min_temp,
                            'max_temp': point.max_temp,
                            'sample_count': point.sample_count
                        }
                        for (x, y), point in baseline_data.items()
                    }
                }
                data['buckets'].append(bucket_data)
        
        with open(export_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Baseline data exported to: {export_path}")