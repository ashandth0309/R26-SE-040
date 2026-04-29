"""
Simple Web Interface for monitoring
Requires: pip install flask flask-socketio
"""

from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import threading
import json
from main import SmartAIDogRobot
import base64

app = Flask(__name__)
socketio = SocketIO(app)

# Global robot instance
robot = None
camera_thread = None

@app.route('/')
def index():
    """Main page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart AI Dog Robot Dashboard</title>
        <style>
            body { font-family: Arial; margin: 20px; background: #f0f0f0; }
            .container { max-width: 1200px; margin: auto; }
            .header { background: #333; color: white; padding: 20px; border-radius: 10px; }
            .video-container { margin: 20px 0; }
            img { width: 100%; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.5); }
            .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .emotion-bar { height: 20px; background: #4CAF50; margin: 5px 0; border-radius: 10px; }
            .controls { margin: 20px 0; }
            button { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
            .btn-start { background: #4CAF50; color: white; }
            .btn-stop { background: #f44336; color: white; }
            .btn-enroll { background: #ff9800; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🐶 Smart AI Dog Robot Dashboard</h1>
            </div>
            
            <div class="controls">
                <button class="btn-start" onclick="startRobot()">Start Robot</button>
                <button class="btn-stop" onclick="stopRobot()">Stop Robot</button>
                <button class="btn-enroll" onclick="enrollPerson()">Enroll New Person</button>
            </div>
            
            <div class="video-container">
                <img id="video-feed" src="/video_feed">
            </div>
            
            <div class="stats">
                <div class="card">
                    <h3>Current Status</h3>
                    <div id="status">
                        <p>Name: <span id="current-name">-</span></p>
                        <p>Emotion: <span id="current-emotion">-</span></p>
                        <p>Confidence: <span id="current-confidence">-</span></p>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Emotion Distribution</h3>
                    <div id="emotion-chart"></div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
        <script>
            const socket = io();
            
            socket.on('status_update', function(data) {
                document.getElementById('current-name').textContent = data.name || '-';
                document.getElementById('current-emotion').textContent = data.emotion || '-';
                document.getElementById('current-confidence').textContent = data.confidence || '-';
            });
            
            function startRobot() {
                fetch('/start', {method: 'POST'});
            }
            
            function stopRobot() {
                fetch('/stop', {method: 'POST'});
            }
            
            function enrollPerson() {
                const name = prompt('Enter name for new person:');
                if (name) {
                    fetch('/enroll', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({name: name})
                    });
                }
            }
        </script>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        global robot
        if robot and robot.cap:
            while True:
                ret, frame = robot.cap.read()
                if ret:
                    # Process frame
                    frame = robot.process_frame(frame)
                    
                    # Encode as JPEG
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               jpeg.tobytes() + b'\r\n')
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    """Start robot"""
    global robot, camera_thread
    if not robot:
        robot = SmartAIDogRobot()
        robot.start_camera()
    return jsonify({"status": "started"})

@app.route('/stop', methods=['POST'])
def stop():
    """Stop robot"""
    global robot
    if robot:
        robot.cleanup()
        robot = None
    return jsonify({"status": "stopped"})

@app.route('/enroll', methods=['POST'])
def enroll():
    """Start enrollment"""
    global robot
    if robot:
        data = request.json
        robot.start_enrollment(data['name'])
    return jsonify({"status": "enrolling"})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'data': 'Connected to robot'})

def run_web_server():
    """Run web server"""
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    run_web_server()