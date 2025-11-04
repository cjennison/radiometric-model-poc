# Radiometric Engine - Thermographic Simulation POC

A proof-of-concept system for radiometric data simulation and abnormality detection using thermographic camera simulation pointed at the sun.

## Features

- **Real-time Simulation**: 150x150 grid simulation of radiometric data from a thermographic camera
- **Sun Modeling**: Realistic sun simulation with time-of-day variations and atmospheric effects
- **Live Visualization**: Real-time heatmap display with interactive controls
- **Anomaly Simulation**: Configurable solar phenomena (sunspots, solar flares) for abnormality detection testing
- **Web Interface**: Modern web dashboard with live updates via WebSocket
- **Time-based Variations**: Automatic temperature variations based on time of day (cooler morning/evening, hotter midday)

## Technical Stack

- **Language**: Python 3.9+
- **Core Libraries**: NumPy, Matplotlib
- **Web Framework**: Flask with SocketIO for real-time updates
- **Architecture**: Modular design with services, models, and configuration

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd radiometric-model-poc
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements/base.txt
   ```

3. **For development**:
   ```bash
   pip install -r requirements/dev.txt
   ```

## Quick Start

### 1. Simple Demo (Matplotlib GUI)

Run the basic demonstration with a standalone matplotlib window:

```bash
python demo.py
```

This will show a live updating heatmap of the sun simulation with:

- Real-time temperature data
- Atmospheric effects and noise
- Random solar anomalies (sunspots/flares)
- Performance statistics

### 2. Web Dashboard

Launch the full web interface:

```bash
python -m src.radiometric_engine.main --mode web
```

Then open your browser to: `http://localhost:5000`

The web dashboard provides:

- Live radiometric visualization
- Interactive controls for forcing anomalies
- Real-time statistics and performance metrics
- Configurable anomaly probability

### 3. Command Line Options

```bash
# Standalone GUI mode (default)
python -m src.radiometric_engine.main --mode gui

# Web interface mode
python -m src.radiometric_engine.main --mode web

# Headless demo mode
python -m src.radiometric_engine.main --mode demo --duration 60

# Custom anomaly rate (0.0 to 1.0)
python -m src.radiometric_engine.main --anomaly-rate 0.05
```

## System Architecture

```
src/radiometric_engine/
├── config/          # Configuration management
├── models/          # Data models (RadiometricFrame, ThermalAnomaly, etc.)
├── services/        # Core services
│   ├── sun_simulator.py     # Sun simulation logic
│   ├── data_stream.py       # Real-time data streaming
│   └── visualization.py     # Matplotlib visualization
├── web/             # Flask web application
└── main.py          # Application entry point
```

## Key Components

### Sun Simulator (`services/sun_simulator.py`)

- Generates realistic 150x150 radiometric data
- Models solar disc with radial temperature gradients
- Simulates atmospheric effects (turbulence, scintillation)
- Creates solar phenomena (sunspots, flares) on demand
- Time-based temperature variations

### Data Stream Engine (`services/data_stream.py`)

- Real-time frame generation at configurable FPS
- Thread-safe consumer pattern for multiple visualizers
- Automatic frame queuing with overflow handling
- Performance monitoring and statistics

### Visualization (`services/visualization.py`)

- Real-time matplotlib heatmap display
- Anomaly highlighting with severity-based colors
- Performance statistics overlay
- Animated updates for smooth visualization

### Web Interface (`web/`)

- Flask application with SocketIO for real-time updates
- Interactive dashboard with live controls
- Real-time statistics and performance metrics
- Responsive design for mobile/desktop

## Configuration

Environment variables can be used to customize behavior:

```bash
# Simulation parameters
export GRID_WIDTH=150
export GRID_HEIGHT=150
export UPDATE_FREQUENCY_HZ=10.0
export SUN_BASE_TEMP=5778.0
export SUN_RADIUS_PIXELS=60
export NOISE_AMPLITUDE=50.0

# Web server
export FLASK_HOST=localhost
export FLASK_PORT=5000
export FLASK_DEBUG=True

# Logging
export LOG_LEVEL=INFO
```

## Usage Examples

### Forcing Solar Anomalies

In web mode, use the "Force Solar Anomaly" button or via API:

```bash
curl -X POST http://localhost:5000/api/force_anomaly
```

### Adjusting Anomaly Probability

```bash
curl -X POST http://localhost:5000/api/set_anomaly_rate \
     -H "Content-Type: application/json" \
     -d '{"rate": 0.05}'
```

### Getting System Statistics

```bash
curl http://localhost:5000/api/stats
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
isort src/
```

### Type Checking

```bash
mypy src/
```

## Simulation Details

### Sun Model

- **Base Temperature**: 5778K (actual sun surface temperature)
- **Size**: 60 pixel radius on 150x150 grid
- **Temperature Gradient**: Radial falloff simulating solar limb darkening
- **Time Variations**: Sinusoidal variation based on time of day (±20%)

### Atmospheric Effects

- **Turbulence**: Slow-changing noise patterns
- **Scintillation**: High-frequency random variations
- **Amplitude**: Configurable noise levels

### Solar Phenomena

- **Sunspots**: Cooler regions (500-1000K below surface)
- **Solar Flares**: Hotter elongated regions (800-1500K above surface)
- **Frequency**: Configurable probability per frame

## Future Enhancements

- [ ] Abnormality detection algorithm implementation
- [ ] Baseline data capture and storage
- [ ] SQLite database integration for event logging
- [ ] Areas of interest (AOI) capture system
- [ ] Real thermographic camera integration
- [ ] Advanced anomaly detection algorithms
- [ ] Historical data analysis tools

## Performance

- **Target Frame Rate**: 10 FPS
- **Memory Usage**: ~50MB for simulation + visualization
- **CPU Usage**: ~5-10% on modern systems
- **Grid Processing**: Optimized NumPy operations for real-time performance

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root directory
2. **Web Interface Not Loading**: Check that Flask is running on the correct port
3. **Slow Performance**: Reduce update frequency or grid size via environment variables
4. **Matplotlib Issues**: Install tkinter for GUI backend: `apt-get install python3-tk` (Linux)

### Logging

Check the log file `radiometric_engine.log` for detailed debugging information.

## License

This project is part of a proof-of-concept for radiometric abnormality detection systems.
