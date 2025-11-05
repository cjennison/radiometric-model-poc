# Statistical Methods for Thermal Anomaly Detection

## Overview

This document provides a comprehensive explanation of the statistical methods used in the radiometric anomaly detection system for baseline creation, data comparison, and anomaly identification.

## Table of Contents

1. [Baseline Creation Process](#baseline-creation-process)
2. [Statistical Analysis Methods](#statistical-analysis-methods)
3. [Anomaly Detection Algorithm](#anomaly-detection-algorithm)
4. [Threshold Determination](#threshold-determination)
5. [False Positive Reduction](#false-positive-reduction)
6. [Mathematical Formulations](#mathematical-formulations)
7. [Implementation Details](#implementation-details)

---

## Baseline Creation Process

### Time-Bucketed Analysis

The system creates statistical baselines using a time-bucketed approach that accounts for temporal variations in thermal patterns.

**Time Bucket Structure:**
- **Bucket Size**: 5-minute intervals
- **Daily Buckets**: 288 buckets per day (24 hours × 12 buckets/hour)
- **Data Aggregation**: Each bucket accumulates temperature readings for the same time period across multiple days

**Example Time Buckets:**
```
Bucket 0:   00:00-00:05  (midnight to 12:05 AM)
Bucket 1:   00:05-00:10  (12:05 AM to 12:10 AM)
...
Bucket 144: 12:00-12:05  (noon to 12:05 PM)
...
Bucket 287: 23:55-00:00  (11:55 PM to midnight)
```

### Baseline Statistics Calculation

For each time bucket and each pixel location (x, y), the system calculates:

**Primary Statistics:**
- **Mean (μ)**: Average temperature for that pixel at that time of day
- **Standard Deviation (σ)**: Measure of temperature variability
- **Sample Count (n)**: Number of observations used to calculate statistics
- **Min/Max**: Extreme values observed for outlier detection

**Adaptive Learning:**
- **Initial Period**: First 24-48 hours used to establish preliminary baseline
- **Refinement**: Baseline continuously updated as more data is collected
- **Decay Factor**: Older data has less influence on current baseline (exponential decay)

### Baseline Data Structure

```python
baseline_data = {
    'pixel_x_y': {
        'time_bucket_id': {
            'mean': float,           # μ - average temperature
            'std_dev': float,        # σ - standard deviation  
            'sample_count': int,     # n - number of samples
            'min_temp': float,       # minimum observed temperature
            'max_temp': float,       # maximum observed temperature
            'last_updated': datetime # timestamp of last update
        }
    }
}
```

---

## Statistical Analysis Methods

### Z-Score Calculation

The core statistical method for anomaly detection is the Z-score, which measures how many standard deviations a new temperature reading differs from the baseline.

**Z-Score Formula:**
```
Z = (X - μ) / σ

Where:
- X = current temperature reading
- μ = baseline mean for this pixel/time bucket
- σ = baseline standard deviation for this pixel/time bucket
```

**Interpretation:**
- **Z = 0**: Temperature exactly matches the baseline mean
- **Z = +2**: Temperature is 2 standard deviations above normal
- **Z = -2**: Temperature is 2 standard deviations below normal
- **|Z| > 3**: Statistically significant anomaly (occurs <0.3% of time in normal distribution)

### Statistical Significance Levels

The system uses standard statistical significance thresholds:

| Z-Score Range | Probability | Severity Level | Interpretation |
|---------------|-------------|----------------|----------------|
| |Z| < 2.0     | ~95%        | Normal         | Expected variation |
| 2.0 ≤ |Z| < 3.0 | ~5%       | LOW            | Unusual but not alarming |
| 3.0 ≤ |Z| < 4.0 | ~0.3%     | MEDIUM         | Worth investigating |
| 4.0 ≤ |Z| < 5.0 | ~0.006%   | HIGH           | Likely real anomaly |
| |Z| ≥ 5.0     | ~0.00006%  | CRITICAL       | Almost certainly anomalous |

### Confidence Interval Analysis

In addition to Z-scores, the system calculates confidence intervals to assess the reliability of anomaly detections.

**95% Confidence Interval:**
```
CI_95 = μ ± (1.96 × σ)
```

**99% Confidence Interval:**
```
CI_99 = μ ± (2.58 × σ)
```

Values outside these intervals are flagged with corresponding confidence levels.

---

## Anomaly Detection Algorithm

### Step-by-Step Process

1. **Input Frame Analysis**
   - Receive new thermal frame (150×150 temperature grid)
   - Determine current time bucket based on timestamp
   - For each pixel, retrieve corresponding baseline statistics

2. **Z-Score Calculation**
   ```python
   for x in range(grid_width):
       for y in range(grid_height):
           current_temp = thermal_frame[x][y]
           baseline_mean = baseline[x][y][time_bucket]['mean']
           baseline_std = baseline[x][y][time_bucket]['std_dev']
           
           if baseline_std > 0:  # Avoid division by zero
               z_score = (current_temp - baseline_mean) / baseline_std
           else:
               z_score = 0
   ```

3. **Threshold Application**
   - Apply severity thresholds to Z-scores
   - Create binary masks for each severity level
   - Identify pixels exceeding each threshold

4. **Spatial Clustering**
   - Group adjacent anomalous pixels using connected component analysis
   - Apply minimum cluster size filter (default: 5 pixels)
   - Calculate cluster centroids and bounding boxes

5. **Temporal Consistency Check**
   - Track anomalies across multiple frames
   - Require persistence over multiple detections
   - Filter out single-frame noise spikes

### Clustering Algorithm

The system uses a connected components algorithm to group adjacent anomalous pixels:

```python
def find_anomaly_clusters(anomaly_mask, min_cluster_size=5):
    """
    Find connected components in the anomaly mask.
    
    Args:
        anomaly_mask: Binary mask where True indicates anomalous pixel
        min_cluster_size: Minimum number of pixels to form a cluster
    
    Returns:
        List of cluster objects with center, size, and bounding box
    """
    # Use 8-connectivity (including diagonal neighbors)
    labeled_array, num_features = scipy.ndimage.label(
        anomaly_mask, 
        structure=np.ones((3,3))
    )
    
    clusters = []
    for i in range(1, num_features + 1):
        cluster_pixels = np.where(labeled_array == i)
        cluster_size = len(cluster_pixels[0])
        
        if cluster_size >= min_cluster_size:
            # Calculate cluster properties
            center_x = np.mean(cluster_pixels[1])
            center_y = np.mean(cluster_pixels[0])
            min_x, max_x = np.min(cluster_pixels[1]), np.max(cluster_pixels[1])
            min_y, max_y = np.min(cluster_pixels[0]), np.max(cluster_pixels[0])
            
            clusters.append({
                'center': (center_x, center_y),
                'size': cluster_size,
                'bbox': (min_x, min_y, max_x, max_y),
                'pixels': cluster_pixels
            })
    
    return clusters
```

---

## Threshold Determination

### Adaptive Thresholds

While the system uses standard statistical thresholds, it can adapt these based on:

**Environmental Factors:**
- **Noise Level**: Higher noise environments may require higher thresholds
- **Baseline Stability**: More stable baselines allow for lower thresholds
- **Temporal Patterns**: Some time periods may naturally have more variation

**Automatic Threshold Adjustment:**
```python
def calculate_adaptive_threshold(baseline_std, noise_level, confidence_level=0.95):
    """
    Calculate adaptive threshold based on baseline characteristics.
    
    Args:
        baseline_std: Standard deviation of baseline
        noise_level: Estimated sensor/environmental noise
        confidence_level: Desired confidence level (0.95 or 0.99)
    
    Returns:
        Adjusted Z-score threshold
    """
    # Base threshold from confidence level
    if confidence_level == 0.95:
        base_threshold = 1.96
    elif confidence_level == 0.99:
        base_threshold = 2.58
    else:
        base_threshold = 3.0
    
    # Adjust for noise
    noise_factor = min(2.0, max(0.5, noise_level / baseline_std))
    adjusted_threshold = base_threshold * noise_factor
    
    return adjusted_threshold
```

### Severity Level Mapping

The system maps Z-scores to severity levels using configurable thresholds:

```python
SEVERITY_THRESHOLDS = {
    'LOW': 2.0,      # 2 standard deviations
    'MEDIUM': 3.0,   # 3 standard deviations  
    'HIGH': 4.0,     # 4 standard deviations
    'CRITICAL': 5.0  # 5 standard deviations
}

def determine_severity(z_score):
    """Determine severity level based on absolute Z-score."""
    abs_z = abs(z_score)
    
    if abs_z >= SEVERITY_THRESHOLDS['CRITICAL']:
        return 'CRITICAL'
    elif abs_z >= SEVERITY_THRESHOLDS['HIGH']:
        return 'HIGH'
    elif abs_z >= SEVERITY_THRESHOLDS['MEDIUM']:
        return 'MEDIUM'
    elif abs_z >= SEVERITY_THRESHOLDS['LOW']:
        return 'LOW'
    else:
        return 'NORMAL'
```

---

## False Positive Reduction

### Multiple Filter Stages

The system employs several techniques to reduce false positive detections:

#### 1. Minimum Cluster Size Filter
- **Purpose**: Eliminate isolated noisy pixels
- **Method**: Require at least N connected pixels (default: 5)
- **Rationale**: Real thermal anomalies typically affect multiple adjacent pixels

#### 2. Temporal Consistency Filter
- **Purpose**: Reduce single-frame noise spikes
- **Method**: Track anomalies across multiple frames
- **Implementation**: 
  ```python
  def temporal_consistency_check(current_anomalies, history_buffer, min_persistence=3):
      """
      Check if anomalies persist across multiple frames.
      
      Args:
          current_anomalies: List of current frame anomalies
          history_buffer: Buffer of previous frame anomalies
          min_persistence: Minimum frames anomaly must persist
      
      Returns:
          Filtered list of consistent anomalies
      """
      consistent_anomalies = []
      
      for anomaly in current_anomalies:
          persistence_count = 1  # Current frame
          
          # Check previous frames
          for historical_frame in history_buffer:
              if is_anomaly_nearby(anomaly, historical_frame, tolerance=5):
                  persistence_count += 1
          
          if persistence_count >= min_persistence:
              anomaly['confidence'] = min(1.0, persistence_count / min_persistence)
              consistent_anomalies.append(anomaly)
      
      return consistent_anomalies
  ```

#### 3. Confidence Scoring
- **Multi-factor Confidence**: Combines statistical significance, cluster size, and temporal persistence
- **Confidence Formula**:
  ```python
  def calculate_confidence(z_score, cluster_size, persistence_frames):
      """Calculate overall confidence score for anomaly detection."""
      # Statistical confidence (based on Z-score)
      stat_confidence = min(1.0, abs(z_score) / 5.0)
      
      # Spatial confidence (based on cluster size)
      spatial_confidence = min(1.0, cluster_size / 20.0)
      
      # Temporal confidence (based on persistence)
      temporal_confidence = min(1.0, persistence_frames / 5.0)
      
      # Combined confidence (weighted average)
      overall_confidence = (
          0.5 * stat_confidence +
          0.3 * spatial_confidence +
          0.2 * temporal_confidence
      )
      
      return overall_confidence
  ```

#### 4. Edge Effect Handling
- **Problem**: Anomalies often first appear at edges due to smaller sample sizes
- **Solution**: Apply higher thresholds near baseline edges
- **Implementation**: Use Welch's t-test for unequal variances

---

## Mathematical Formulations

### Baseline Update Algorithm

The system uses an exponential moving average to update baselines:

```python
def update_baseline(current_mean, current_std, new_value, alpha=0.1):
    """
    Update baseline statistics with new temperature reading.
    
    Args:
        current_mean: Current baseline mean
        current_std: Current baseline standard deviation
        new_value: New temperature reading
        alpha: Learning rate (0.1 = 10% weight to new data)
    
    Returns:
        Updated mean and standard deviation
    """
    # Update mean using exponential moving average
    new_mean = (1 - alpha) * current_mean + alpha * new_value
    
    # Update variance using Welford's online algorithm
    delta = new_value - current_mean
    new_variance = (1 - alpha) * (current_std ** 2) + alpha * (delta ** 2)
    new_std = np.sqrt(new_variance)
    
    return new_mean, new_std
```

### Welford's Online Algorithm

For numerical stability when updating variance:

```python
class OnlineStats:
    """Online calculation of mean and variance using Welford's algorithm."""
    
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squares of differences from mean
    
    def update(self, x):
        """Add new value and update statistics."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    
    @property
    def variance(self):
        """Calculate variance."""
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)
    
    @property
    def std_dev(self):
        """Calculate standard deviation."""
        return np.sqrt(self.variance)
```

### Statistical Tests

#### Shapiro-Wilk Test for Normality
The system can optionally test if temperature distributions are normal:

```python
from scipy import stats

def test_normality(temperature_samples, alpha=0.05):
    """
    Test if temperature samples follow normal distribution.
    
    Args:
        temperature_samples: Array of temperature readings
        alpha: Significance level for test
    
    Returns:
        Boolean indicating if distribution is normal
    """
    if len(temperature_samples) < 3:
        return True  # Assume normal for small samples
    
    statistic, p_value = stats.shapiro(temperature_samples)
    return p_value > alpha  # Null hypothesis: data is normally distributed
```

#### Grubbs' Test for Outliers
Detect outliers before updating baseline:

```python
def grubbs_test(data, alpha=0.05):
    """
    Apply Grubbs' test to detect outliers.
    
    Args:
        data: Array of data points
        alpha: Significance level
    
    Returns:
        Boolean array indicating outliers
    """
    n = len(data)
    if n < 3:
        return np.zeros(n, dtype=bool)
    
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    
    # Calculate Grubbs' statistic for each point
    G = np.abs(data - mean) / std_dev
    
    # Critical value from Grubbs' table (approximation)
    t_critical = stats.t.ppf(1 - alpha/(2*n), n-2)
    G_critical = ((n-1) / np.sqrt(n)) * np.sqrt(t_critical**2 / (n-2 + t_critical**2))
    
    return G > G_critical
```

---

## Implementation Details

### Database Schema

The baseline data is stored in an SQLite database with the following schema:

```sql
CREATE TABLE baseline_data (
    pixel_x INTEGER,
    pixel_y INTEGER,
    time_bucket INTEGER,
    mean_temp REAL,
    std_dev REAL,
    sample_count INTEGER,
    min_temp REAL,
    max_temp REAL,
    last_updated TIMESTAMP,
    PRIMARY KEY (pixel_x, pixel_y, time_bucket)
);

CREATE INDEX idx_baseline_lookup ON baseline_data (pixel_x, pixel_y, time_bucket);
CREATE INDEX idx_baseline_updated ON baseline_data (last_updated);
```

### Performance Optimizations

#### Vectorized Operations
Use NumPy for efficient array operations:

```python
def vectorized_z_score_calculation(thermal_frame, baseline_means, baseline_stds):
    """
    Calculate Z-scores for entire frame using vectorized operations.
    
    Args:
        thermal_frame: 150x150 array of temperatures
        baseline_means: 150x150 array of baseline means
        baseline_stds: 150x150 array of baseline standard deviations
    
    Returns:
        150x150 array of Z-scores
    """
    # Avoid division by zero
    safe_stds = np.where(baseline_stds > 0, baseline_stds, 1.0)
    
    # Vectorized Z-score calculation
    z_scores = (thermal_frame - baseline_means) / safe_stds
    
    # Set Z-score to 0 where standard deviation was 0
    z_scores = np.where(baseline_stds > 0, z_scores, 0.0)
    
    return z_scores
```

#### Memory-Efficient Baseline Storage
Use sparse storage for inactive pixels:

```python
class SparseBaseline:
    """Memory-efficient baseline storage using sparse matrices."""
    
    def __init__(self, grid_width, grid_height, num_time_buckets):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_time_buckets = num_time_buckets
        
        # Use dictionaries for sparse storage
        self.means = {}
        self.std_devs = {}
        self.sample_counts = {}
    
    def get_baseline(self, x, y, time_bucket):
        """Get baseline statistics for pixel and time bucket."""
        key = (x, y, time_bucket)
        
        return {
            'mean': self.means.get(key, 0.0),
            'std_dev': self.std_devs.get(key, 1.0),
            'sample_count': self.sample_counts.get(key, 0)
        }
    
    def update_baseline(self, x, y, time_bucket, mean, std_dev, count):
        """Update baseline statistics."""
        key = (x, y, time_bucket)
        self.means[key] = mean
        self.std_devs[key] = std_dev
        self.sample_counts[key] = count
```

### Error Handling

#### Numerical Stability
Handle edge cases in statistical calculations:

```python
def safe_z_score(value, mean, std_dev, min_std=1e-6):
    """
    Calculate Z-score with numerical stability checks.
    
    Args:
        value: Current temperature value
        mean: Baseline mean
        std_dev: Baseline standard deviation
        min_std: Minimum standard deviation to prevent division by zero
    
    Returns:
        Z-score with numerical safeguards
    """
    # Handle missing baseline data
    if np.isnan(mean) or np.isnan(std_dev):
        return 0.0
    
    # Prevent division by very small numbers
    safe_std = max(std_dev, min_std)
    
    # Calculate Z-score
    z_score = (value - mean) / safe_std
    
    # Clip extreme values to prevent overflow
    z_score = np.clip(z_score, -10.0, 10.0)
    
    return z_score
```

#### Data Validation
Validate input data before processing:

```python
def validate_thermal_frame(thermal_frame):
    """
    Validate thermal frame data before processing.
    
    Args:
        thermal_frame: 2D numpy array of temperatures
    
    Returns:
        Boolean indicating if frame is valid
    
    Raises:
        ValueError: If frame format is invalid
    """
    # Check dimensions
    if thermal_frame.shape != (150, 150):
        raise ValueError(f"Invalid frame dimensions: {thermal_frame.shape}")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(thermal_frame)) or np.any(np.isinf(thermal_frame)):
        raise ValueError("Frame contains NaN or infinite values")
    
    # Check temperature range (Kelvin)
    if np.any(thermal_frame < 0) or np.any(thermal_frame > 10000):
        raise ValueError("Temperatures outside valid range (0-10000K)")
    
    return True
```

---

## Conclusion

This statistical framework provides robust anomaly detection by:

1. **Learning Normal Patterns**: Time-bucketed baselines capture temporal variations
2. **Statistical Rigor**: Z-score analysis with proper confidence intervals
3. **False Positive Reduction**: Multiple filtering stages ensure reliable detections
4. **Adaptive Learning**: Baselines continuously improve with more data
5. **Numerical Stability**: Robust implementation handles edge cases

The combination of these methods enables accurate detection of thermal anomalies while maintaining low false positive rates in real-world deployment scenarios.