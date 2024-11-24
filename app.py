import socket
import cv2
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
from datetime import datetime
import csv
import os

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Video capture with resolution set to reduce processing load
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Dictionary to track detected objects with entry and exit times
tracked_objects = {}

# Create "Detected objects" folder if it doesn't exist
folder_name = "Detected objects"
os.makedirs(folder_name, exist_ok=True)

# Create CSV file for detections if it doesn't exist
date_str = datetime.now().strftime("%Y-%m-%d")
csv_file = os.path.join(folder_name, f"{date_str}_detections.csv")
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Object', 'Entry Time', 'Exit Time'])

def get_local_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return local_ip

def generate_frames():
    global tracked_objects
    counter = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        frame = cv2.resize(frame, (640, 480))

        if counter % 3 == 0:
            results = model(frame)
            current_objects = set()

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = box.cls[0]
                    label = model.names[int(cls)]

                    if label not in tracked_objects:
                        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        tracked_objects[label] = {'entry_time': entry_time, 'exit_time': None}
                        with open(csv_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([label, entry_time, 'IN PROGRESS'])

                    current_objects.add(label)

            for obj_label in list(tracked_objects):
                if obj_label not in current_objects:
                    if tracked_objects[obj_label]['exit_time'] is None:
                        exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        tracked_objects[obj_label]['exit_time'] = exit_time
                        with open(csv_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([obj_label, tracked_objects[obj_label]['entry_time'], exit_time])
                        del tracked_objects[obj_label]

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame")
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cv2.waitKey(1)
        counter += 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detections')
def get_detections():
    return jsonify({label: "Detected" for label in tracked_objects.keys()})

if __name__ == '__main__':
    local_ip = get_local_ip()
    print(f"Access the application on another device using: http://{local_ip}:5000")
    print("Ensure both devices are connected to the same network.")
    app.run(host='0.0.0.0', port=5000, debug=True)
