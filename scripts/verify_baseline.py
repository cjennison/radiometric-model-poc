#!/usr/bin/env python3
"""
Baseline Data Verification Script

This script analyzes and verifies the baseline data collected by the radiometric engine.
It provides detailed statistics, data quality checks, and visualizations to validate
the baseline collection system.

Usage:
    python scripts/verify_baseline.py
    python scripts/verify_baseline.py --detailed
    python scripts/verify_baseline.py --export-csv baseline_analysis.csv
"""

import sys
import os
import sqlite3
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.radiometric_engine.services.baseline_manager import BaselineDataManager


class BaselineVerifier:
    """Verification and analysis tool for baseline data."""
    
    def __init__(self, db_path: str = "baseline_data.db"):
        """
        Initialize the verifier.
        
        Args:
            db_path: Path to the baseline database
        """
        self.db_path = db_path
        self.manager = BaselineDataManager(bucket_minutes=5)
        
    def verify_database_integrity(self) -> Dict[str, any]:
        """Check database integrity and schema."""
        print("🔍 Verifying Database Integrity...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                expected_tables = ['baseline_buckets', 'baseline_data', 'collection_sessions']
                missing_tables = set(expected_tables) - set(tables)
                
                # Check table schemas
                schema_info = {}
                for table in expected_tables:
                    if table in tables:
                        cursor.execute(f"PRAGMA table_info({table})")
                        schema_info[table] = cursor.fetchall()
                
                integrity_results = {
                    'database_exists': os.path.exists(self.db_path),
                    'tables_exist': len(missing_tables) == 0,
                    'missing_tables': list(missing_tables),
                    'table_schemas': schema_info,
                    'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
                }
                
                print(f"✅ Database exists: {integrity_results['database_exists']}")
                print(f"✅ All tables exist: {integrity_results['tables_exist']}")
                print(f"📊 Database size: {integrity_results['database_size_mb']:.2f} MB")
                
                if missing_tables:
                    print(f"❌ Missing tables: {missing_tables}")
                
                return integrity_results
                
        except Exception as e:
            print(f"❌ Database integrity check failed: {e}")
            return {'error': str(e)}
    
    def get_collection_summary(self) -> Dict[str, any]:
        """Get overall collection statistics."""
        print("\n📈 Collection Summary...")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Session statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_sessions,
                    SUM(frames_collected) as total_frames,
                    MIN(start_time) as first_collection,
                    MAX(COALESCE(end_time, start_time)) as last_collection,
                    AVG(frames_collected) as avg_frames_per_session
                FROM collection_sessions
            """)
            session_stats = cursor.fetchone()
            
            # Bucket statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_buckets,
                    COUNT(CASE WHEN sample_count > 0 THEN 1 END) as populated_buckets,
                    AVG(sample_count) as avg_samples_per_bucket,
                    MAX(sample_count) as max_samples_per_bucket
                FROM baseline_buckets
            """)
            bucket_stats = cursor.fetchone()
            
            # Data coverage
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT bucket_id) as buckets_with_data,
                    COUNT(*) as total_grid_points,
                    COUNT(DISTINCT grid_x || ',' || grid_y) as unique_positions,
                    AVG(sample_count) as avg_samples_per_point,
                    MIN(sample_count) as min_samples_per_point,
                    MAX(sample_count) as max_samples_per_point
                FROM baseline_data
            """)
            data_stats = cursor.fetchone()
            
            summary = {
                'sessions': {
                    'total': session_stats[0],
                    'total_frames': session_stats[1] or 0,
                    'first_collection': session_stats[2],
                    'last_collection': session_stats[3],
                    'avg_frames_per_session': session_stats[4] or 0
                },
                'buckets': {
                    'total': bucket_stats[0],
                    'populated': bucket_stats[1],
                    'coverage_percent': (bucket_stats[1] / bucket_stats[0] * 100) if bucket_stats[0] > 0 else 0,
                    'avg_samples': bucket_stats[2] or 0,
                    'max_samples': bucket_stats[3] or 0
                },
                'data': {
                    'buckets_with_data': data_stats[0] or 0,
                    'total_grid_points': data_stats[1] or 0,
                    'unique_positions': data_stats[2] or 0,
                    'avg_samples_per_point': data_stats[3] or 0,
                    'min_samples_per_point': data_stats[4] or 0,
                    'max_samples_per_point': data_stats[5] or 0
                }
            }
            
            print(f"📅 Collection Period: {summary['sessions']['first_collection']} to {summary['sessions']['last_collection']}")
            print(f"🎬 Total Sessions: {summary['sessions']['total']}")
            print(f"📊 Total Frames: {summary['sessions']['total_frames']}")
            print(f"🕐 Time Coverage: {summary['buckets']['populated']}/{summary['buckets']['total']} buckets ({summary['buckets']['coverage_percent']:.1f}%)")
            print(f"📍 Grid Points: {summary['data']['total_grid_points']:,} ({summary['data']['unique_positions']:,} unique positions)")
            print(f"📈 Samples per Point: {summary['data']['avg_samples_per_point']:.1f} avg, {summary['data']['min_samples_per_point']}-{summary['data']['max_samples_per_point']} range")
            
            return summary
    
    def analyze_time_coverage(self) -> Dict[str, any]:
        """Analyze temporal coverage patterns."""
        print("\n⏰ Time Coverage Analysis...")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get populated buckets by hour of day
            cursor.execute("""
                SELECT 
                    bucket_id,
                    bucket_id % 288 as bucket_in_day,
                    (bucket_id % 288) * 5 / 60.0 as hour_of_day,
                    sample_count
                FROM baseline_buckets 
                WHERE sample_count > 0
                ORDER BY bucket_id
            """)
            bucket_data = cursor.fetchall()
            
            if not bucket_data:
                print("❌ No bucket data found")
                return {}
            
            # Group by hour of day
            hourly_coverage = {}
            for bucket_id, bucket_in_day, hour_of_day, sample_count in bucket_data:
                hour = int(hour_of_day)
                if hour not in hourly_coverage:
                    hourly_coverage[hour] = []
                hourly_coverage[hour].append(sample_count)
            
            # Calculate hourly statistics
            hourly_stats = {}
            for hour in range(24):
                if hour in hourly_coverage:
                    samples = hourly_coverage[hour]
                    hourly_stats[hour] = {
                        'buckets': len(samples),
                        'total_samples': sum(samples),
                        'avg_samples': np.mean(samples),
                        'coverage': True
                    }
                else:
                    hourly_stats[hour] = {
                        'buckets': 0,
                        'total_samples': 0,
                        'avg_samples': 0,
                        'coverage': False
                    }
            
            # Find coverage gaps
            gaps = [hour for hour in range(24) if not hourly_stats[hour]['coverage']]
            best_hours = sorted(hourly_stats.keys(), 
                              key=lambda h: hourly_stats[h]['total_samples'], reverse=True)[:5]
            
            coverage_analysis = {
                'hourly_stats': hourly_stats,
                'coverage_gaps': gaps,
                'best_covered_hours': best_hours,
                'total_hours_covered': len([h for h in hourly_stats if hourly_stats[h]['coverage']]),
                'coverage_percentage': (24 - len(gaps)) / 24 * 100
            }
            
            print(f"🕐 Hours Covered: {coverage_analysis['total_hours_covered']}/24 ({coverage_analysis['coverage_percentage']:.1f}%)")
            if gaps:
                print(f"⚠️  Coverage Gaps: {gaps} (hours)")
            print(f"🏆 Best Coverage: {best_hours[:3]} (hours)")
            
            return coverage_analysis
    
    def analyze_spatial_coverage(self) -> Dict[str, any]:
        """Analyze spatial coverage patterns."""
        print("\n🗺️  Spatial Coverage Analysis...")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get grid coverage statistics - aggregate by position
            cursor.execute("""
                SELECT 
                    grid_x, grid_y,
                    COUNT(*) as bucket_count,
                    SUM(sample_count) as total_samples,
                    AVG(mean_temp) as avg_temperature,
                    MIN(min_temp) as min_temperature,
                    MAX(max_temp) as max_temperature,
                    AVG(std_dev) as avg_std_dev
                FROM baseline_data
                GROUP BY grid_x, grid_y
                ORDER BY total_samples DESC
            """)
            grid_data = cursor.fetchall()
            
            if not grid_data:
                print("❌ No grid data found")
                return {}
            
            # Convert to arrays for analysis
            positions = [(row[0], row[1]) for row in grid_data]
            samples = [row[3] for row in grid_data]  # total_samples per position
            temperatures = [row[4] for row in grid_data]
            
            # Calculate center of grid (should be sun center)
            center_x, center_y = 75, 75  # 150/2
            
            # Calculate distances from center
            distances = [np.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in positions]
            
            spatial_analysis = {
                'total_positions': len(positions),
                'expected_positions': 150 * 150,
                'coverage_percent': len(positions) / (150 * 150) * 100,
                'sample_statistics': {
                    'min': min(samples),
                    'max': max(samples),
                    'mean': np.mean(samples),
                    'std': np.std(samples)
                },
                'temperature_statistics': {
                    'min': min(temperatures),
                    'max': max(temperatures),
                    'mean': np.mean(temperatures),
                    'std': np.std(temperatures)
                },
                'distance_from_center': {
                    'min': min(distances),
                    'max': max(distances),
                    'mean': np.mean(distances)
                }
            }
            
            print(f"📍 Grid Coverage: {spatial_analysis['total_positions']:,}/{spatial_analysis['expected_positions']:,} positions ({spatial_analysis['coverage_percent']:.1f}%)")
            print(f"🌡️  Temperature Range: {spatial_analysis['temperature_statistics']['min']:.1f}K - {spatial_analysis['temperature_statistics']['max']:.1f}K")
            print(f"📊 Sample Range: {spatial_analysis['sample_statistics']['min']} - {spatial_analysis['sample_statistics']['max']} samples per position (avg: {spatial_analysis['sample_statistics']['mean']:.1f})")
            print(f"📐 Distance from Center: {spatial_analysis['distance_from_center']['mean']:.1f} pixels average")
            
            return spatial_analysis
    
    def check_data_quality(self) -> Dict[str, any]:
        """Check for data quality issues."""
        print("\n🔍 Data Quality Checks...")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            quality_issues = []
            
            # Check for impossible temperatures
            cursor.execute("""
                SELECT COUNT(*) FROM baseline_data 
                WHERE mean_temp < 0 OR mean_temp > 10000
            """)
            impossible_temps = cursor.fetchone()[0]
            if impossible_temps > 0:
                quality_issues.append(f"Found {impossible_temps} points with impossible temperatures")
            
            # Check for negative standard deviations
            cursor.execute("""
                SELECT COUNT(*) FROM baseline_data 
                WHERE std_dev < 0
            """)
            negative_std = cursor.fetchone()[0]
            if negative_std > 0:
                quality_issues.append(f"Found {negative_std} points with negative standard deviations")
            
            # Check for min > max temperatures
            cursor.execute("""
                SELECT COUNT(*) FROM baseline_data 
                WHERE min_temp > max_temp
            """)
            invalid_ranges = cursor.fetchone()[0]
            if invalid_ranges > 0:
                quality_issues.append(f"Found {invalid_ranges} points where min_temp > max_temp")
            
            # Check for extremely low sample counts per grid position (aggregate by position)
            cursor.execute("""
                SELECT COUNT(*) FROM (
                    SELECT grid_x, grid_y, SUM(sample_count) as total_samples_per_position
                    FROM baseline_data 
                    GROUP BY grid_x, grid_y
                    HAVING total_samples_per_position < 5
                )
            """)
            low_sample_positions = cursor.fetchone()[0]
            
            # Check for missing data in expected sun region (center circle) - count unique positions
            cursor.execute("""
                SELECT COUNT(DISTINCT grid_x || ',' || grid_y) as unique_positions
                FROM baseline_data 
                WHERE (grid_x - 75) * (grid_x - 75) + (grid_y - 75) * (grid_y - 75) <= 60 * 60
            """)
            sun_region_positions = cursor.fetchone()[0]
            expected_sun_positions = int(np.pi * 60 * 60)  # Approximate circle area in pixels
            
            quality_results = {
                'quality_issues': quality_issues,
                'impossible_temperatures': impossible_temps,
                'negative_std_devs': negative_std,
                'invalid_ranges': invalid_ranges,
                'low_sample_positions': low_sample_positions,
                'sun_region_coverage': {
                    'positions': sun_region_positions,
                    'expected': expected_sun_positions,
                    'percentage': (sun_region_positions / expected_sun_positions * 100) if expected_sun_positions > 0 else 0
                }
            }
            
            if quality_issues:
                print("⚠️  Quality Issues Found:")
                for issue in quality_issues:
                    print(f"   - {issue}")
            else:
                print("✅ No major quality issues detected")
            
            print(f"⚠️  Low Sample Positions: {low_sample_positions} (< 5 samples total)")
            print(f"☀️  Sun Region Coverage: {sun_region_positions}/{expected_sun_positions} positions ({quality_results['sun_region_coverage']['percentage']:.1f}%)")
            
            return quality_results
    
    def export_analysis_csv(self, filename: str) -> None:
        """Export detailed analysis to CSV."""
        print(f"\n💾 Exporting analysis to {filename}...")
        
        with sqlite3.connect(self.db_path) as conn:
            # Export detailed grid data
            query = """
                SELECT 
                    bd.bucket_id,
                    bd.grid_x,
                    bd.grid_y,
                    (bd.bucket_id % 288) * 5 / 60.0 as hour_of_day,
                    bd.mean_temp,
                    bd.std_dev,
                    bd.min_temp,
                    bd.max_temp,
                    bd.sample_count,
                    bb.last_updated
                FROM baseline_data bd
                JOIN baseline_buckets bb ON bd.bucket_id = bb.bucket_id
                ORDER BY bd.bucket_id, bd.grid_x, bd.grid_y
            """
            
            df = pd.read_sql_query(query, conn)
            df.to_csv(filename, index=False)
            print(f"✅ Exported {len(df)} records to {filename}")
    
    def create_visualizations(self, output_dir: str = "baseline_analysis") -> None:
        """Create visualization plots."""
        print(f"\n📊 Creating visualizations in {output_dir}/...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Time coverage heatmap
            df_time = pd.read_sql_query("""
                SELECT 
                    (bucket_id % 288) * 5 / 60.0 as hour_of_day,
                    sample_count
                FROM baseline_buckets 
                WHERE sample_count > 0
            """, conn)
            
            if not df_time.empty:
                plt.figure(figsize=(12, 6))
                plt.hist(df_time['hour_of_day'], bins=24, weights=df_time['sample_count'], 
                        alpha=0.7, edgecolor='black')
                plt.xlabel('Hour of Day')
                plt.ylabel('Total Samples')
                plt.title('Baseline Data Collection by Hour of Day')
                plt.grid(True, alpha=0.3)
                plt.savefig(f"{output_dir}/time_coverage.png", dpi=150, bbox_inches='tight')
                plt.close()
                
            # Temperature distribution
            df_temp = pd.read_sql_query("""
                SELECT mean_temp, sample_count FROM baseline_data
            """, conn)
            
            if not df_temp.empty:
                plt.figure(figsize=(10, 6))
                plt.hist(df_temp['mean_temp'], bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel('Mean Temperature (K)')
                plt.ylabel('Number of Grid Points')
                plt.title('Distribution of Mean Temperatures')
                plt.grid(True, alpha=0.3)
                plt.savefig(f"{output_dir}/temperature_distribution.png", dpi=150, bbox_inches='tight')
                plt.close()
                
        print(f"✅ Visualizations saved to {output_dir}/")


def main():
    """Main script execution."""
    parser = argparse.ArgumentParser(description="Verify and analyze baseline data")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    parser.add_argument("--export-csv", help="Export analysis to CSV file")
    parser.add_argument("--visualize", action="store_true", help="Create visualization plots")
    parser.add_argument("--db-path", default="baseline_data.db", help="Path to baseline database")
    
    args = parser.parse_args()
    
    print("🔬 Baseline Data Verification Tool")
    print("=" * 50)
    
    verifier = BaselineVerifier(args.db_path)
    
    # Always run basic checks
    integrity = verifier.verify_database_integrity()
    if 'error' in integrity:
        print(f"❌ Cannot proceed: {integrity['error']}")
        return 1
    
    summary = verifier.get_collection_summary()
    
    if args.detailed:
        time_analysis = verifier.analyze_time_coverage()
        spatial_analysis = verifier.analyze_spatial_coverage()
        quality_results = verifier.check_data_quality()
    
    if args.export_csv:
        verifier.export_analysis_csv(args.export_csv)
    
    if args.visualize:
        verifier.create_visualizations()
    
    print("\n✅ Verification complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())