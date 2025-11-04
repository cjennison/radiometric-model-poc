"""
Flask web application for radiometric engine visualization.

This module provides a web-based interface for viewing real-time radiometric data
with interactive controls and statistics display.
"""

import io
import base64
import json
import math
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from ..services import DataStreamEngine
from ..services.visualization import WebVisualizationManager
from ..models import RadiometricFrame

# Global variables for web app state
data_stream: Optional[DataStreamEngine] = None
socketio: Optional[SocketIO] = None
latest_frame: Optional[RadiometricFrame] = None


def _calculate_simulated_time(time_factor: float) -> dict:
    """
    Calculate simulated time of day from the time factor.
    
    The time factor varies sinusoidally: factor = 1.0 + 0.2 * cos(2π * (hour - 6) / 24)
    We need to reverse this to find the simulated hour.
    
    Args:
        time_factor: Current time factor (0.8 to 1.2)
        
    Returns:
        Dictionary with simulated time information
    """
    # Reverse the cosine function: cos(angle) = (factor - 1.0) / 0.2
    cos_value = (time_factor - 1.0) / 0.2
    
    # Clamp to valid cosine range [-1, 1]
    cos_value = max(-1.0, min(1.0, cos_value))
    
    # Get angle from arccos (returns 0 to π)
    angle = math.acos(abs(cos_value))
    
    # Convert back to hour: angle = 2π * (hour - 6) / 24
    # hour = (angle * 24) / (2π) + 6
    hour_offset = (angle * 24) / (2 * math.pi)
    
    # Determine if we're in AM or PM based on time factor trend
    # For simplicity, assume we're always in the increasing part of the day
    if cos_value >= 0:  # Morning to noon
        simulated_hour = 12 - hour_offset  # Peak at noon (12:00)
    else:  # Noon to evening
        simulated_hour = 12 + hour_offset
    
    # Ensure hour is in valid range [0, 24)
    simulated_hour = simulated_hour % 24
    
    # Extract hour and minute
    hour = int(simulated_hour)
    minute = int((simulated_hour - hour) * 60)
    
    # Create time description
    if hour < 6:
        description = "Early Morning"
    elif hour < 12:
        description = "Morning"
    elif hour == 12:
        description = "Noon"
    elif hour < 18:
        description = "Afternoon"
    elif hour < 21:
        description = "Evening"
    else:
        description = "Night"
    
    return {
        'hour': hour,
        'minute': minute,
        'time_string': f"{hour:02d}:{minute:02d}",
        'description': description,
        'time_factor': time_factor
    }


def _make_json_safe(obj):
    """
    Convert objects to JSON-serializable format.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: _make_json_safe(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Convert objects with attributes to dictionaries
        return _make_json_safe(obj.__dict__)
    else:
        return obj


def create_app(stream_engine: DataStreamEngine) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        stream_engine: Data stream engine instance
        
    Returns:
        Configured Flask application
    """
    global data_stream, socketio, web_visualizer
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'radiometric_secret_key'
    
    # Initialize SocketIO for real-time updates
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Store reference to data stream
    data_stream = stream_engine
    
    # Initialize web-optimized visualizer
    web_visualizer = WebVisualizationManager(figsize=(8, 8))
    
    # Register frame consumer for web updates
    data_stream.add_consumer(_handle_new_frame)
    
    # Register routes
    _register_routes(app)
    _register_socketio_events()
    
    return app


def _handle_new_frame(frame: RadiometricFrame) -> None:
    """
    Handle new frames from the data stream for web distribution.
    
    Args:
        frame: New radiometric frame
    """
    global latest_frame, socketio, web_visualizer
    
    latest_frame = frame
    
    if socketio and web_visualizer:
        # Create optimized image (returns None if throttled)
        image_data = web_visualizer.create_image(frame)
        
        # Only emit if we have a new image (not throttled)
        if image_data:
            # Get actual simulated time from the engine
            time_factor = frame.metadata.get('time_factor', 1.0) if frame.metadata else 1.0
            sim_datetime = data_stream.get_simulated_time() if data_stream else None
            
            # Format simulated time for web client
            if sim_datetime:
                simulated_time = {
                    'time_string': sim_datetime.strftime('%H:%M:%S'),
                    'description': f"Simulated time at {sim_datetime.strftime('%H:%M')}"
                }
            else:
                # Fallback to old method if engine not available
                simulated_time = _calculate_simulated_time(time_factor)
            
            # Get anomaly statistics from metadata
            anomaly_stats = frame.metadata.get('active_anomalies', {}) if frame.metadata else {}
            frame_count = frame.metadata.get('frame_count', 0) if frame.metadata else 0
            
            # Prepare frame data for web client
            frame_data = {
                'timestamp': frame.timestamp.isoformat(),
                'image': image_data,
                'stats': {
                    'min_temp': float(np.min(frame.data)),
                    'max_temp': float(np.max(frame.data)),
                    'mean_temp': float(np.mean(frame.data)),
                    'time_factor': time_factor,
                    'simulated_time': simulated_time,
                    'frame_count': frame_count,
                },
                'anomalies': anomaly_stats
            }
            
            # Emit to all connected clients
            socketio.emit('new_frame', frame_data)


def _register_routes(app: Flask) -> None:
    """Register Flask routes."""
    
    @app.route('/')
    def index():
        """Main dashboard page."""
        return render_template('dashboard.html')
    
    @app.route('/api/stats')
    def get_stats():
        """Get current engine statistics."""
        stats = data_stream.get_stats() if data_stream else {}
        return jsonify(stats)
    
    @app.route('/api/force_anomaly', methods=['POST'])
    def force_anomaly():
        """Force the next frame to contain an anomaly."""
        if data_stream:
            data_stream.force_anomaly()
            return jsonify({'status': 'success', 'message': 'Anomaly forced'})
        return jsonify({'status': 'error', 'message': 'Data stream not available'})
    
    @app.route('/api/set_anomaly_rate', methods=['POST'])
    def set_anomaly_rate():
        """Set the anomaly probability rate."""
        try:
            data = request.get_json()
            rate = float(data.get('rate', 0.01))
            
            if not (0.0 <= rate <= 1.0):
                return jsonify({'status': 'error', 'message': 'Rate must be between 0.0 and 1.0'})
            
            if data_stream:
                data_stream.set_anomaly_probability(rate)
                return jsonify({'status': 'success', 'message': f'Anomaly rate set to {rate}'})
            
            return jsonify({'status': 'error', 'message': 'Data stream not available'})
            
        except (ValueError, TypeError) as e:
            return jsonify({'status': 'error', 'message': f'Invalid rate value: {e}'})
    
    @app.route('/api/set_web_frequency', methods=['POST'])
    def set_web_frequency():
        """Set the web interface update frequency."""
        try:
            data = request.get_json()
            every_n_frames = int(data.get('every_n_frames', 3))
            
            if not (1 <= every_n_frames <= 10):
                return jsonify({'status': 'error', 'message': 'Frequency must be between 1 and 10'})
            
            if web_visualizer:
                web_visualizer.set_update_frequency(every_n_frames)
                return jsonify({'status': 'success', 'message': f'Web frequency set to every {every_n_frames} frames'})
            
            return jsonify({'status': 'error', 'message': 'Web visualizer not available'})
            
        except (ValueError, TypeError) as e:
            return jsonify({'status': 'error', 'message': f'Invalid frequency value: {e}'})
    
    @app.route('/api/set_time_speed', methods=['POST'])
    def set_time_speed():
        """Set the time speed multiplier."""
        try:
            data = request.get_json()
            speed = int(data.get('speed', 1))
            
            if not (1 <= speed <= 1000):
                return jsonify({'status': 'error', 'message': 'Speed must be between 1 and 1000'})
            
            if data_stream:
                data_stream.set_time_speed(speed)
                return jsonify({'status': 'success', 'message': f'Time speed set to {speed}x'})
            
            return jsonify({'status': 'error', 'message': 'Data stream not available'})
            
        except (ValueError, TypeError) as e:
            return jsonify({'status': 'error', 'message': f'Invalid speed value: {e}'})
    
    @app.route('/api/reset_time', methods=['POST'])
    def reset_time():
        """Reset the simulation time to current real time."""
        try:
            if data_stream:
                data_stream.reset_time()
                return jsonify({'status': 'success', 'message': 'Time reset successfully'})
            
            return jsonify({'status': 'error', 'message': 'Data stream not available'})
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error resetting time: {e}'})
    
    @app.route('/api/current_frame')
    def get_current_frame():
        """Get the current frame data."""
        if latest_frame:
            return jsonify({
                'timestamp': latest_frame.timestamp.isoformat(),
                'width': latest_frame.width,
                'height': latest_frame.height,
                'stats': {
                    'min_temp': float(np.min(latest_frame.data)),
                    'max_temp': float(np.max(latest_frame.data)),
                    'mean_temp': float(np.mean(latest_frame.data)),
                }
            })
        return jsonify({'error': 'No frame available'})


def _register_socketio_events() -> None:
    """Register SocketIO event handlers."""
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        print(f"Client connected: {request.sid}")
        
        # Send current frame if available
        if latest_frame and web_visualizer:
            # Force an immediate update for new connections
            web_visualizer._frame_count = 0  # Reset counter to force update
            image_data = web_visualizer.create_image(latest_frame)
            
            if image_data:
                frame_data = {
                    'timestamp': latest_frame.timestamp.isoformat(),
                    'image': image_data,
                    'stats': {
                        'min_temp': float(np.min(latest_frame.data)),
                        'max_temp': float(np.max(latest_frame.data)),
                        'mean_temp': float(np.mean(latest_frame.data)),
                        'time_factor': latest_frame.metadata.get('time_factor', 1.0) if latest_frame.metadata else 1.0,
                    }
                }
                emit('new_frame', frame_data)
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        print(f"Client disconnected: {request.sid}")
    
    @socketio.on('request_stats')
    def handle_stats_request():
        """Handle request for engine statistics."""
        stats = data_stream.get_stats() if data_stream else {}
        
        # Convert datetime objects to ISO format strings for JSON serialization
        json_safe_stats = _make_json_safe(stats)
        
        emit('stats_update', json_safe_stats)