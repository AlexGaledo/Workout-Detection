from ultralytics import YOLO 


model = YOLO('yolov8n-pose.pt')  # Load a YOLOv8n model
model.predict(source=0, show=True)  # Use webcam (0) as source

