#!/usr/bin/env python3
"""
Comprehensive Anomaly Detection Test Suite

This script validates the anomaly detection engine by simulating various extreme
scenarios and confirming the system can distinguish between true anomalies and
normal variations.

Test Categories:
1. Catastrophic Events (should trigger anomalies)
   - Complete sun extinction
   - Permanent hole in sun
   - Massive temperature increases
   
2. Normal Variations (should NOT trigger anomalies)
   - Minor temperature fluctuations (5% changes)
   - Gradual temperature changes
   - Normal daily pattern variations

3. Edge Cases
   - Partial sun obscuration
   - Localized temperature spikes
   - Multi-region anomalies
"""

import sys
import os
import numpy as np
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.radiometric_engine.services.baseline_manager import BaselineDataManager
from src.radiometric_engine.services.anomaly_detection import (
    AnomalyDetectionEngine, DetectedAnomaly, AnomalyType, AnomalySeverity
)
from src.radiometric_engine.services.sun_simulator import SunSimulator
from src.radiometric_engine.models import RadiometricFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestScenario:
    """Represents a single test scenario."""
    name: str
    description: str
    should_detect_anomaly: bool
    expected_anomaly_types: List[AnomalyType]
    modifier_function: callable
    severity_threshold: AnomalySeverity = AnomalySeverity.LOW


class AnomalyDetectionTester:
    """
    Comprehensive test suite for anomaly detection system.
    
    This class creates controlled test scenarios to validate that the anomaly
    detection engine correctly identifies genuine anomalies while ignoring
    normal variations.
    """
    
    def __init__(self, baseline_db_path: str = None):
        """
        Initialize the anomaly detection tester.
        
        Args:
            baseline_db_path: Path to baseline database (uses default if None)
        """
        # Set database path
        self.baseline_db_path = baseline_db_path or "baseline_data.db"
        
        # Initialize baseline manager
        self.baseline_manager = BaselineDataManager(db_path=self.baseline_db_path)
        
        # Initialize anomaly detector with more relaxed thresholds for testing
        detection_config = {
            'std_dev_threshold': 2.0,        # Reduced from 3.0 to 2.0 for even better sensitivity
            'cluster_min_size': 10,          # Increased from 5 to 10  
            'cluster_max_distance': 5,       # Increased from 3 to 5
            'confidence_threshold': 0.8,     # Increased from 0.7 to 0.8
            'temporal_window_minutes': 15,
            'region_analysis_enabled': True,
            'single_pixel_detection': False
        }
        
        self.anomaly_detector = AnomalyDetectionEngine(
            baseline_db_path=self.baseline_db_path,
            detection_config=detection_config
        )
        
        # Initialize sun simulator for generating test data
        self.sun_simulator = SunSimulator()
        
        # Test results storage
        self.test_results: List[Dict] = []
        
        # Grid size from simulator
        self.grid_size = 150
    
    def create_baseline_frame(self, time_offset_hours: float = 12.0) -> np.ndarray:
        """
        Create a baseline-representative frame at a specific time.
        
        Args:
            time_offset_hours: Hours from midnight (12.0 = noon)
            
        Returns:
            2D numpy array representing temperature data
        """
        # Create a time at the specified offset
        test_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        test_time += timedelta(hours=time_offset_hours)
        
        # Generate a normal frame
        frame = self.sun_simulator.generate_frame(
            anomaly_chance=0.0,  # No anomalies for baseline
            current_time=test_time
        )
        
        return frame.data
    
    def modify_sun_extinction(self, data: np.ndarray) -> np.ndarray:
        """
        Simulate complete sun extinction - all temperatures drop to space temperature.
        
        This should definitely trigger multiple anomalies as the sun completely disappears.
        """
        modified = data.copy()
        # Set everything to space temperature (~3K)
        modified[:, :] = 3.0
        logger.info("Applied sun extinction - all temperatures set to 3K")
        return modified
    
    def modify_sun_hole(self, data: np.ndarray, hole_size: int = 30) -> np.ndarray:
        """
        Create a permanent hole in the sun - a cold region in the center.
        
        Args:
            hole_size: Diameter of the hole in pixels
        """
        modified = data.copy()
        center_x, center_y = self.grid_size // 2, self.grid_size // 2
        
        # Create a circular hole
        y, x = np.ogrid[:self.grid_size, :self.grid_size]
        mask = (x - center_x)**2 + (y - center_y)**2 <= (hole_size // 2)**2
        
        # Set hole region to space temperature
        modified[mask] = 3.0
        
        logger.info(f"Applied sun hole - {hole_size}px diameter hole at center")
        return modified
    
    def modify_massive_temperature_increase(self, data: np.ndarray, increase_kelvin: float = 500.0) -> np.ndarray:
        """
        Increase sun temperature by a massive amount.
        
        Args:
            increase_kelvin: Temperature increase in Kelvin
        """
        modified = data.copy()
        
        # Only increase temperatures above space temperature (likely sun regions)
        sun_mask = data > 1000  # Anything above 1000K is likely sun
        modified[sun_mask] += increase_kelvin
        
        logger.info(f"Applied massive temperature increase - +{increase_kelvin}K to sun regions")
        return modified
    
    def modify_minor_temperature_increase(self, data: np.ndarray, percentage: float = 5.0) -> np.ndarray:
        """
        Apply a minor temperature increase (should NOT trigger anomalies).
        
        Args:
            percentage: Percentage increase in temperature
        """
        modified = data.copy()
        
        # Only modify sun regions
        sun_mask = data > 1000
        modified[sun_mask] *= (1.0 + percentage / 100.0)
        
        logger.info(f"Applied minor temperature increase - +{percentage}% to sun regions")
        return modified
    
    def modify_gradual_temperature_drift(self, data: np.ndarray, drift_kelvin: float = 50.0) -> np.ndarray:
        """
        Apply a gradual temperature drift (should NOT trigger anomalies).
        
        Args:
            drift_kelvin: Gradual drift in Kelvin
        """
        modified = data.copy()
        
        # Apply gradient across the grid to simulate gradual change
        sun_mask = data > 1000
        gradient = np.linspace(0, drift_kelvin, self.grid_size)
        gradient_2d = np.tile(gradient, (self.grid_size, 1))
        
        modified[sun_mask] += gradient_2d[sun_mask]
        
        logger.info(f"Applied gradual temperature drift - max {drift_kelvin}K gradient")
        return modified
    
    def modify_localized_spike(self, data: np.ndarray, spike_size: int = 5, spike_temp: float = 1000.0) -> np.ndarray:
        """
        Create a small localized temperature spike.
        
        Args:
            spike_size: Size of the spike region in pixels
            spike_temp: Additional temperature for the spike
        """
        modified = data.copy()
        
        # Place spike at a random sun location
        sun_positions = np.where(data > 1000)
        if len(sun_positions[0]) > 0:
            # Pick a random sun position
            idx = np.random.randint(len(sun_positions[0]))
            center_y, center_x = sun_positions[0][idx], sun_positions[1][idx]
            
            # Create small spike region
            y_min = max(0, center_y - spike_size // 2)
            y_max = min(self.grid_size, center_y + spike_size // 2)
            x_min = max(0, center_x - spike_size // 2)
            x_max = min(self.grid_size, center_x + spike_size // 2)
            
            modified[y_min:y_max, x_min:x_max] += spike_temp
            
            logger.info(f"Applied localized spike - +{spike_temp}K over {spike_size}x{spike_size} region")
        
        return modified
    
    def modify_random_noise(self, data: np.ndarray, noise_std: float = 10.0) -> np.ndarray:
        """
        Add random noise to the data (should NOT trigger anomalies if small).
        
        Args:
            noise_std: Standard deviation of the noise
        """
        modified = data.copy()
        noise = np.random.normal(0, noise_std, data.shape)
        modified += noise
        
        # Clip to prevent negative temperatures (Kelvin scale)
        modified = np.clip(modified, 0.1, None)
        
        logger.info(f"Applied random noise - std={noise_std}K")
        return modified
    
    def create_test_scenarios(self) -> List[TestScenario]:
        """
        Create comprehensive test scenarios.
        
        Returns:
            List of test scenarios to execute
        """
        scenarios = [
            # Catastrophic events (should trigger anomalies)
            TestScenario(
                name="Complete Sun Extinction",
                description="Entire sun goes out - all temperatures drop to space level",
                should_detect_anomaly=True,
                expected_anomaly_types=[AnomalyType.TEMPERATURE_DROP, AnomalyType.REGIONAL_CLUSTER],
                modifier_function=self.modify_sun_extinction,
                severity_threshold=AnomalySeverity.CRITICAL
            ),
            
            TestScenario(
                name="Permanent Sun Hole",
                description="Large hole appears in center of sun",
                should_detect_anomaly=True,
                expected_anomaly_types=[AnomalyType.TEMPERATURE_DROP, AnomalyType.REGIONAL_CLUSTER],
                modifier_function=lambda data: self.modify_sun_hole(data, hole_size=30),
                severity_threshold=AnomalySeverity.HIGH
            ),
            
            TestScenario(
                name="Massive Temperature Increase",
                description="Sun temperature increases by 1500K",
                should_detect_anomaly=True,
                expected_anomaly_types=[AnomalyType.TEMPERATURE_SPIKE, AnomalyType.STATISTICAL_OUTLIER],
                modifier_function=lambda data: self.modify_massive_temperature_increase(data, 1500.0),
                severity_threshold=AnomalySeverity.HIGH
            ),
            
            TestScenario(
                name="Large Localized Spike",
                description="Significant temperature spike in localized region",
                should_detect_anomaly=True,
                expected_anomaly_types=[AnomalyType.TEMPERATURE_SPIKE, AnomalyType.REGIONAL_CLUSTER],
                modifier_function=lambda data: self.modify_localized_spike(data, spike_size=20, spike_temp=2000.0),
                severity_threshold=AnomalySeverity.MEDIUM
            ),
            
            # Normal variations (should NOT trigger anomalies)
            TestScenario(
                name="Minor Temperature Increase (5%)",
                description="Small 5% increase in sun temperature",
                should_detect_anomaly=False,
                expected_anomaly_types=[],
                modifier_function=lambda data: self.modify_minor_temperature_increase(data, 5.0)
            ),
            
            TestScenario(
                name="Gradual Temperature Drift",
                description="Slow gradual temperature change across sun",
                should_detect_anomaly=False,
                expected_anomaly_types=[],
                modifier_function=lambda data: self.modify_gradual_temperature_drift(data, 50.0)
            ),
            
            TestScenario(
                name="Small Random Noise",
                description="Random temperature noise within normal bounds",
                should_detect_anomaly=False,
                expected_anomaly_types=[],
                modifier_function=lambda data: self.modify_random_noise(data, 10.0)
            ),
            
            TestScenario(
                name="Very Minor Temperature Increase (2%)",
                description="Tiny 2% increase in sun temperature",
                should_detect_anomaly=False,
                expected_anomaly_types=[],
                modifier_function=lambda data: self.modify_minor_temperature_increase(data, 2.0)
            ),
            
            # Edge cases
            TestScenario(
                name="Small Localized Spike",
                description="Small temperature spike that might be borderline",
                should_detect_anomaly=True,  # Should detect but with low severity
                expected_anomaly_types=[AnomalyType.TEMPERATURE_SPIKE],
                modifier_function=lambda data: self.modify_localized_spike(data, spike_size=5, spike_temp=1500.0),
                severity_threshold=AnomalySeverity.LOW
            ),
            
            TestScenario(
                name="Moderate Random Noise",
                description="Moderate noise that should trigger some anomalies",
                should_detect_anomaly=True,
                expected_anomaly_types=[AnomalyType.STATISTICAL_OUTLIER],
                modifier_function=lambda data: self.modify_random_noise(data, 100.0),
                severity_threshold=AnomalySeverity.LOW
            ),
        ]
        
        return scenarios
    
    def run_single_test(self, scenario: TestScenario) -> Dict:
        """
        Run a single test scenario.
        
        Args:
            scenario: Test scenario to execute
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING TEST: {scenario.name}")
        logger.info(f"Description: {scenario.description}")
        logger.info(f"Expected to detect anomaly: {scenario.should_detect_anomaly}")
        
        try:
            # Create baseline data for noon (sun at peak)
            baseline_data = self.create_baseline_frame(time_offset_hours=12.0)
            
            # Apply scenario modification
            modified_data = scenario.modifier_function(baseline_data)
            
            # Run anomaly detection
            test_time = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
            detected_anomalies = self.anomaly_detector.detect_anomalies(modified_data, test_time)
            
            # Analyze results
            anomaly_detected = len(detected_anomalies) > 0
            
            # Check severity if anomalies were detected
            meets_severity_threshold = False
            detected_types = []
            max_severity = None
            
            if detected_anomalies:
                detected_types = [anomaly.anomaly_type for anomaly in detected_anomalies]
                severities = [anomaly.severity for anomaly in detected_anomalies]
                max_severity = max(severities, key=lambda s: list(AnomalySeverity).index(s))
                
                # Check if any anomaly meets the severity threshold
                severity_order = list(AnomalySeverity)
                threshold_index = severity_order.index(scenario.severity_threshold)
                for severity in severities:
                    if severity_order.index(severity) >= threshold_index:
                        meets_severity_threshold = True
                        break
            
            # Determine test result
            test_passed = True
            failure_reasons = []
            
            if scenario.should_detect_anomaly and not anomaly_detected:
                test_passed = False
                failure_reasons.append("Expected anomaly detection but none found")
            elif not scenario.should_detect_anomaly and anomaly_detected:
                test_passed = False
                failure_reasons.append(f"Unexpected anomaly detection: {len(detected_anomalies)} anomalies")
            
            # Check expected anomaly types
            if scenario.should_detect_anomaly and detected_anomalies:
                expected_types_found = any(
                    expected_type in detected_types 
                    for expected_type in scenario.expected_anomaly_types
                )
                if not expected_types_found and scenario.expected_anomaly_types:
                    test_passed = False
                    failure_reasons.append(
                        f"Expected anomaly types {scenario.expected_anomaly_types} not found. "
                        f"Detected: {detected_types}"
                    )
                
                # Check severity threshold
                if not meets_severity_threshold:
                    test_passed = False
                    failure_reasons.append(
                        f"Expected severity >= {scenario.severity_threshold.name}, "
                        f"but max severity was {max_severity.name if max_severity else 'None'}"
                    )
            
            # Log results
            if test_passed:
                logger.info("✅ TEST PASSED")
            else:
                logger.error("❌ TEST FAILED")
                for reason in failure_reasons:
                    logger.error(f"  - {reason}")
            
            logger.info(f"Anomalies detected: {len(detected_anomalies)}")
            for anomaly in detected_anomalies:
                center_x, center_y = anomaly.region_center
                logger.info(
                    f"  - {anomaly.anomaly_type.name} at ({center_x}, {center_y}) "
                    f"severity={anomaly.severity.name} confidence={anomaly.confidence_score:.3f}"
                )
            
            # Create result dictionary
            result = {
                'scenario_name': scenario.name,
                'description': scenario.description,
                'should_detect_anomaly': scenario.should_detect_anomaly,
                'anomaly_detected': anomaly_detected,
                'anomaly_count': len(detected_anomalies),
                'detected_types': [t.name for t in detected_types],
                'expected_types': [t.name for t in scenario.expected_anomaly_types],
                'max_severity': max_severity.name if max_severity else None,
                'expected_severity': scenario.severity_threshold.name,
                'meets_severity_threshold': meets_severity_threshold,
                'test_passed': test_passed,
                'failure_reasons': failure_reasons,
                'anomalies': [
                    {
                        'type': a.anomaly_type.name,
                        'severity': a.severity.name,
                        'position': a.region_center,
                        'confidence': a.confidence_score,
                        'description': a.description
                    }
                    for a in detected_anomalies
                ]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ TEST ERROR: {e}", exc_info=True)
            return {
                'scenario_name': scenario.name,
                'description': scenario.description,
                'test_passed': False,
                'failure_reasons': [f"Test execution error: {e}"],
                'error': str(e)
            }
    
    def run_all_tests(self) -> Dict:
        """
        Run all test scenarios.
        
        Returns:
            Dictionary with comprehensive test results
        """
        logger.info("🚀 Starting Comprehensive Anomaly Detection Test Suite")
        logger.info("=" * 80)
        
        # Check if baseline data exists
        baseline_stats = self.baseline_manager.get_baseline_stats()
        if baseline_stats.get('total_buckets', 0) == 0:
            logger.warning("⚠️  No baseline data found! Results may be unreliable.")
            logger.warning("   Run baseline data collection first for better accuracy.")
        else:
            total_buckets = baseline_stats.get('total_buckets', 0)
            total_samples = baseline_stats.get('total_data_points', baseline_stats.get('total_samples', 0))
            logger.info(f"✅ Using baseline data: {total_buckets} buckets, "
                       f"{total_samples} samples")
        
        scenarios = self.create_test_scenarios()
        
        # Run all tests
        all_results = []
        passed_tests = 0
        failed_tests = 0
        
        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"\n[{i}/{len(scenarios)}] Running scenario: {scenario.name}")
            result = self.run_single_test(scenario)
            all_results.append(result)
            
            if result['test_passed']:
                passed_tests += 1
            else:
                failed_tests += 1
        
        # Generate summary
        logger.info("\n" + "=" * 80)
        logger.info("🏁 TEST SUITE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total tests: {len(scenarios)}")
        logger.info(f"Passed: {passed_tests} ✅")
        logger.info(f"Failed: {failed_tests} ❌")
        logger.info(f"Success rate: {passed_tests/len(scenarios)*100:.1f}%")
        
        if failed_tests > 0:
            logger.info("\n❌ FAILED TESTS:")
            for result in all_results:
                if not result['test_passed']:
                    logger.info(f"  - {result['scenario_name']}")
                    for reason in result.get('failure_reasons', []):
                        logger.info(f"    • {reason}")
        
        summary = {
            'total_tests': len(scenarios),
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / len(scenarios) * 100,
            'baseline_stats': baseline_stats,
            'detailed_results': all_results
        }
        
        return summary
    
    def visualize_test_scenario(self, scenario_name: str, save_path: str = None) -> None:
        """
        Visualize a specific test scenario.
        
        Args:
            scenario_name: Name of the scenario to visualize
            save_path: Optional path to save the visualization
        """
        scenarios = self.create_test_scenarios()
        scenario = next((s for s in scenarios if s.name == scenario_name), None)
        
        if not scenario:
            logger.error(f"Scenario '{scenario_name}' not found")
            return
        
        # Create baseline and modified data
        baseline_data = self.create_baseline_frame(time_offset_hours=12.0)
        modified_data = scenario.modifier_function(baseline_data)
        
        # Run anomaly detection
        test_time = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        detected_anomalies = self.anomaly_detector.detect_anomalies(modified_data, test_time)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Baseline data
        im1 = axes[0].imshow(baseline_data, cmap='hot', interpolation='nearest')
        axes[0].set_title('Baseline Data')
        axes[0].set_xlabel('X Position')
        axes[0].set_ylabel('Y Position')
        plt.colorbar(im1, ax=axes[0], label='Temperature (K)')
        
        # Modified data
        im2 = axes[1].imshow(modified_data, cmap='hot', interpolation='nearest')
        axes[1].set_title(f'Modified Data: {scenario_name}')
        axes[1].set_xlabel('X Position')
        axes[1].set_ylabel('Y Position')
        plt.colorbar(im2, ax=axes[1], label='Temperature (K)')
        
        # Difference
        difference = modified_data - baseline_data
        im3 = axes[2].imshow(difference, cmap='RdBu_r', interpolation='nearest')
        axes[2].set_title('Temperature Difference')
        axes[2].set_xlabel('X Position')
        axes[2].set_ylabel('Y Position')
        plt.colorbar(im3, ax=axes[2], label='ΔTemperature (K)')
        
        # Mark detected anomalies
        for anomaly in detected_anomalies:
            for ax in axes[1:]:  # Mark on modified and difference plots
                circle = plt.Circle((anomaly.x, anomaly.y), 3, 
                                  color='cyan', fill=False, linewidth=2)
                ax.add_patch(circle)
                ax.text(anomaly.x + 5, anomaly.y, 
                       f'{anomaly.severity.name}', 
                       color='cyan', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()


def main():
    """Main function to run the test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Anomaly Detection System')
    parser.add_argument('--baseline-db', type=str, 
                       help='Path to baseline database (uses default if not specified)')
    parser.add_argument('--visualize', type=str, 
                       help='Visualize a specific test scenario')
    parser.add_argument('--scenario', type=str,
                       help='Run only a specific scenario by name')
    parser.add_argument('--save-results', type=str,
                       help='Save detailed results to JSON file')
    
    args = parser.parse_args()
    
    # Create tester
    tester = AnomalyDetectionTester(baseline_db_path=args.baseline_db)
    
    if args.visualize:
        # Visualize specific scenario
        tester.visualize_test_scenario(args.visualize)
        return
    
    if args.scenario:
        # Run specific scenario
        scenarios = tester.create_test_scenarios()
        scenario = next((s for s in scenarios if s.name == args.scenario), None)
        if scenario:
            result = tester.run_single_test(scenario)
            print(f"\nTest Result for '{args.scenario}':")
            print(f"Passed: {result['test_passed']}")
            if not result['test_passed']:
                print("Failure reasons:")
                for reason in result.get('failure_reasons', []):
                    print(f"  - {reason}")
        else:
            print(f"Scenario '{args.scenario}' not found")
            print("Available scenarios:")
            for s in scenarios:
                print(f"  - {s.name}")
        return
    
    # Run full test suite
    results = tester.run_all_tests()
    
    # Save results if requested
    if args.save_results:
        import json
        with open(args.save_results, 'w') as f:
            # Convert enum values to strings for JSON serialization
            json_results = json.dumps(results, indent=2, default=str)
            f.write(json_results)
        logger.info(f"Results saved to {args.save_results}")
    
    # Exit with appropriate code
    exit_code = 0 if results['failed_tests'] == 0 else 1
    exit(exit_code)


if __name__ == '__main__':
    main()