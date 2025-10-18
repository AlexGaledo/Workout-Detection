from ultralytics import YOLO
import joblib
import numpy as np
import pandas as pd
import cv2

# Load models
pose_model = YOLO('yolov8n-pose.pt')
clf = joblib.load('pushup_classifier.pkl')

# Function to predict pushup stage from keypoints
def predict_stage_from_keypoints(keypoints_xy, keypoints_conf):
    features = np.zeros(51)
    for i in range(17):
        features[i*3] = keypoints_xy[i*2]
        features[i*3+1] = keypoints_xy[i*2+1]
        features[i*3+2] = keypoints_conf[i]
    feature_names = [f'col_{i}' for i in range(51)]
    features_df = pd.DataFrame([features], columns=feature_names)
    pred = clf.predict(features_df)[0]
    return "Push-up" if pred == 1 else "Push-down"

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO expects image path or numpy array
    results = pose_model(frame, max_det=1)

    for r in results:
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            xy = r.keypoints.xy[0].cpu().numpy().flatten()
            conf = r.keypoints.conf[0].cpu().numpy().flatten()

            label = predict_stage_from_keypoints(xy, conf)

            # Annotate the frame
            annotated = r.plot()
            cv2.putText(annotated, label, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Show frame
            cv2.imshow("Push-up Counter", annotated)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
