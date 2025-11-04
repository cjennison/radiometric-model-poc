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

from src.radiometric_engine.services import DataStreamEngine
from src.radiometric_engine.services.visualization import WebVisualizationManager
from src.radiometric_engine.models import RadiometricFrame

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
    
    # Connect web visualizer to data stream for anomaly image saving
    data_stream.set_web_visualizer(web_visualizer)
    
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
            
            # Get temperature graph data
            temperature_graph = data_stream.get_temperature_graph_data() if data_stream else {}
            
            # Get anomaly data
            anomaly_summary = data_stream.get_anomaly_summary() if data_stream else {}
            recent_anomalies = data_stream.get_recent_anomalies(max_count=10) if data_stream else []
            
            # Convert anomalies to JSON-serializable format
            anomaly_data = []
            for anomaly in recent_anomalies:
                anomaly_data.append({
                    'type': anomaly.anomaly_type.name,
                    'severity': anomaly.severity.name,
                    'position': [anomaly.region_center[0], anomaly.region_center[1]],
                    'confidence': anomaly.confidence_score,
                    'timestamp': anomaly.timestamp.isoformat(),
                    'description': anomaly.description,
                    'affected_area': len(anomaly.affected_pixels),
                    'deviation_magnitude': anomaly.temperature_deviation
                })
            
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
                'anomalies': anomaly_stats,
                'anomaly_detection': {
                    'summary': anomaly_summary,
                    'recent_anomalies': anomaly_data
                },
                'temperature_graph': temperature_graph
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
        """Force the next frame to contain a specific type of anomaly."""
        try:
            data = request.get_json() or {}
            anomaly_type = data.get('type')  # 'sunspot', 'flare', 'prominence', or None
            intensity = data.get('intensity')  # float or None
            size = data.get('size')  # float or None
            
            # Validate anomaly type if provided
            valid_types = ['sunspot', 'flare', 'prominence']
            if anomaly_type and anomaly_type not in valid_types:
                return jsonify({
                    'status': 'error', 
                    'message': f'Invalid anomaly type. Must be one of: {valid_types}'
                })
            
            # Validate intensity if provided
            if intensity is not None:
                try:
                    intensity = float(intensity)
                    if intensity <= 0:
                        return jsonify({
                            'status': 'error',
                            'message': 'Intensity must be greater than 0'
                        })
                except (ValueError, TypeError):
                    return jsonify({
                        'status': 'error',
                        'message': 'Intensity must be a valid number'
                    })
            
            # Validate size if provided
            if size is not None:
                try:
                    size = float(size)
                    if size <= 0:
                        return jsonify({
                            'status': 'error',
                            'message': 'Size must be greater than 0'
                        })
                except (ValueError, TypeError):
                    return jsonify({
                        'status': 'error',
                        'message': 'Size must be a valid number'
                    })
            
            if data_stream:
                data_stream.force_anomaly(anomaly_type, intensity, size)
                
                # Build response message
                msg_parts = ['Anomaly forced']
                if anomaly_type:
                    msg_parts.append(f"type: {anomaly_type}")
                if intensity is not None:
                    msg_parts.append(f"intensity: {intensity}")
                if size is not None:
                    msg_parts.append(f"size: {size}")
                
                return jsonify({
                    'status': 'success', 
                    'message': ' - '.join(msg_parts),
                    'anomaly_type': anomaly_type or 'random',
                    'intensity': intensity,
                    'size': size
                })
            
            return jsonify({'status': 'error', 'message': 'Data stream not available'})
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error forcing anomaly: {e}'})
    
    @app.route('/api/anomaly_types')
    def get_anomaly_types():
        """Get available anomaly types with their descriptions."""
        return jsonify({
            'status': 'success',
            'anomaly_types': [
                {
                    'type': 'sunspot',
                    'name': 'Sunspot',
                    'description': 'Cool dark regions on the sun\'s surface',
                    'effect': 'Temperature drops by 800-1200K',
                    'default_intensity_range': [0.3, 0.7],
                    'default_size_range': [8, 20],
                    'probability': '60%'
                },
                {
                    'type': 'flare',
                    'name': 'Solar Flare',
                    'description': 'Explosive bursts of radiation from the sun',
                    'effect': 'Temperature rises by 1000-1800K',
                    'default_intensity_range': [1.3, 1.8],
                    'default_size_range': [5, 15],
                    'probability': '20%'
                },
                {
                    'type': 'prominence',
                    'name': 'Solar Prominence',
                    'description': 'Large loops of plasma extending from the sun',
                    'effect': 'Temperature rises by 600-1200K',
                    'default_intensity_range': [1.1, 1.4],
                    'default_size_range': [10, 25],
                    'probability': '20%'
                }
            ]
        })
    
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
    
    # Baseline Data Management API Routes
    
    @app.route('/api/baseline/start', methods=['POST'])
    def start_baseline_collection():
        """Start baseline data collection."""
        try:
            data = request.get_json() or {}
            description = data.get('description', 'Web interface baseline collection')
            
            if data_stream:
                session_id = data_stream.start_baseline_collection(description)
                return jsonify({
                    'status': 'started', 
                    'message': 'Baseline collection started',
                    'session_id': session_id
                })
            
            return jsonify({'status': 'error', 'message': 'Data stream not available'})
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error starting baseline collection: {e}'})
    
    @app.route('/api/baseline/stop', methods=['POST'])
    def stop_baseline_collection():
        """Stop baseline data collection."""
        try:
            data = request.get_json() or {}
            session_id = data.get('session_id')
            
            if data_stream:
                # Pass session_id if provided, otherwise stop all collections
                stats = data_stream.stop_baseline_collection(session_id)
                return jsonify({
                    'status': 'stopped',
                    'message': 'Baseline collection stopped',
                    'stats': stats
                })
            
            return jsonify({'status': 'error', 'message': 'Data stream not available'})
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error stopping baseline collection: {e}'})
    
    @app.route('/api/baseline/stats')
    def get_baseline_stats():
        """Get baseline data statistics."""
        try:
            if data_stream:
                stats = data_stream.get_baseline_stats()
                is_collecting = data_stream.is_collecting_baseline()
                return jsonify({
                    'status': 'success',
                    'stats': stats,
                    'is_collecting': is_collecting
                })
            
            return jsonify({'status': 'error', 'message': 'Data stream not available'})
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error getting baseline stats: {e}'})
    
    @app.route('/api/anomalies')
    def get_anomalies():
        """Get recent anomaly detection results."""
        try:
            if data_stream:
                anomalies = data_stream.get_recent_anomalies(max_count=50)  # Last 50 anomalies
                anomaly_summary = data_stream.get_anomaly_summary()
                
                # Convert anomalies to JSON-serializable format
                anomaly_data = []
                for anomaly in anomalies:
                    anomaly_data.append({
                        'type': anomaly.anomaly_type.name,
                        'severity': anomaly.severity.name,
                        'position': [anomaly.region_center[0], anomaly.region_center[1]],
                        'confidence': anomaly.confidence_score,
                        'timestamp': anomaly.timestamp.isoformat(),
                        'description': anomaly.description,
                        'affected_area': len(anomaly.affected_pixels),
                        'deviation_magnitude': anomaly.temperature_deviation
                    })
                
                return jsonify({
                    'status': 'success',
                    'anomalies': anomaly_data,
                    'summary': anomaly_summary
                })
            
            return jsonify({'status': 'error', 'message': 'Data stream not available'})
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error getting anomalies: {e}'})
    
    @app.route('/api/anomalies/clear', methods=['POST'])
    def clear_anomalies():
        """Clear anomaly detection history."""
        try:
            if data_stream:
                data_stream.clear_anomaly_history()
                return jsonify({'status': 'success', 'message': 'Anomaly history cleared'})
            
            return jsonify({'status': 'error', 'message': 'Data stream not available'})
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error clearing anomalies: {e}'})
    
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