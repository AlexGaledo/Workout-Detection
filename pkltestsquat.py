from ultralytics import YOLO
import joblib
import pandas as pd
import numpy as np
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# === Load YOLO pose model ===
pose_model = YOLO("yolov8n-pose.pt")

# === Load your .pkl classifier ===
clf_data = joblib.load("models/squat_classifier2.pkl")

# Check if it's a dict or a model
if isinstance(clf_data, dict):
    clf = clf_data['model']
    feature_names = clf_data.get('feature_names', None)
    if feature_names is None:
        feature_names = [f'col_{i}' for i in range(51)]  # fallback
else:
    clf = clf_data
    feature_names = clf.feature_names_in_

# === Test image ===
img_path = "sample/squat_test3.png"      # ← change this

# Run YOLO pose detection
results = pose_model(img_path, verbose=False)

# Check if YOLO detected a person
r = results[0]

if r.keypoints is None or len(r.keypoints.xy) == 0:
    print("❌ No keypoints detected.")
else:
    # Extract XY and confidence
    xy = r.keypoints.xy[0].cpu().numpy().flatten()
    conf = r.keypoints.conf[0].cpu().numpy().flatten()

    # Build feature vector: [x1, y1, c1, x2, y2, c2, ...]
    row = []
    for i in range(len(xy)//2):
        row += [xy[i*2], xy[i*2+1], conf[i]]

    # Convert to DataFrame (must match training feature order)
    X_df = pd.DataFrame([row], columns=feature_names)

    # Predict class
    pred_class = clf.predict(X_df)[0]
    print("✅ Predicted class:", pred_class)
