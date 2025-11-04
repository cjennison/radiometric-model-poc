"""
Flask web application for radiometric engine visualization.

This module provides a web-based interface for viewing real-time radiometric data
with interactive controls and statistics display.
"""

import io
import base64
import json
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from ..services import DataStreamEngine
from ..services.web_visualization import WebVisualizationManager
from ..models import RadiometricFrame

# Global variables for web app state
data_stream: Optional[DataStreamEngine] = None
socketio: Optional[SocketIO] = None
latest_frame: Optional[RadiometricFrame] = None
web_visualizer: Optional[WebVisualizationManager] = None


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
            # Prepare frame data for web client
            frame_data = {
                'timestamp': frame.timestamp.isoformat(),
                'image': image_data,
                'stats': {
                    'min_temp': float(np.min(frame.data)),
                    'max_temp': float(np.max(frame.data)),
                    'mean_temp': float(np.mean(frame.data)),
                    'time_factor': frame.metadata.get('time_factor', 1.0) if frame.metadata else 1.0,
                }
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
        emit('stats_update', stats)