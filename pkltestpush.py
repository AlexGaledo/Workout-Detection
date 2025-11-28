
from ultralytics import YOLO
import joblib
import pandas as pd
import numpy as np
import warnings

# ------------------------------
# 1️⃣ Suppress sklearn warnings
# ------------------------------
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ------------------------------
# 2️⃣ Load YOLO pose model
# ------------------------------
pose_model = YOLO("yolov8n-pose.pt")  # Make sure you have this model in your folder

# ------------------------------
# 3️⃣ Load your push-up classifier
# ------------------------------
clf_data = joblib.load("models/pushup_pose_classifier_merged.pkl")  # Change path if needed

# Handle dict or model
if isinstance(clf_data, dict):
    clf = clf_data['model']
    feature_names = clf_data.get('feature_names', None)
    if feature_names is None:
        feature_names = [f'col_{i}' for i in range(51)]
else:
    clf = clf_data
    feature_names = clf.feature_names_in_

# ------------------------------
# 4️⃣ Test image
# ------------------------------
img_path = "sample/samplepushup.jpg"  # Change this to your image path
results = pose_model(img_path, verbose=False)

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

    # Convert to DataFrame (must match training features)
    X_df = pd.DataFrame([row], columns=feature_names)

    # Predict class
    pred_class = clf.predict(X_df)[0]

    # Map to readable labels
    class_map = {0: "Push-up down", 1: "Push-up up"}
    print("✅ Predicted class:", class_map[pred_class])
