from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO  # Import YOLO from the ultralytics package

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Ensure you have the yolov8n.pt model file

# Video capture
# Change to your camera index or video file
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break

        # Object detection
        results = model(frame)
        if not results:
            print("No results detected")
            continue
        
        for result in results:  # Iterate through results
            boxes = result.boxes
            for box in boxes:  # Process each detected object
                x1, y1, x2, y2 = box.xyxy[0]  # Get bounding box coordinates
                conf = box.conf[0]  # Get confidence score
                cls = box.cls[0]  # Get class index
                label = f'{model.names[int(cls)]} {conf:.2f}'  # Format label
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame")
            break
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)