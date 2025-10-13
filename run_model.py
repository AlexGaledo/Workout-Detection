from ultralytics import YOLO

model = YOLO("models/finetunedv4.pt")

model.predict(source=0, show=True, conf=0.4) 




