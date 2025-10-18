from ultralytics import YOLO
import joblib
import numpy as np
import pandas as pd
import cv2
import os

def supports_cv2_gui():
    """Return True if cv2.imshow is usable in this environment."""
    try:
        test_img = np.zeros((2, 2, 3), dtype=np.uint8)
        cv2.namedWindow("_cv2_gui_test", cv2.WINDOW_NORMAL)
        cv2.imshow("_cv2_gui_test", test_img)
        cv2.waitKey(1)
        cv2.destroyWindow("_cv2_gui_test")
        return True
    except Exception:
        return False


# === Load models ===
pose_model = YOLO('yolov8n-pose.pt')        # YOLOv8 pose model
clf = joblib.load('pushup_classifier.pkl')  # Your trained RandomForest model


def predict_pushup_stage(image_path):
    # Run YOLO pose detection (limit to 1 person)
    results = pose_model(image_path, max_det=1)
    img = cv2.imread(image_path)

    for r in results:
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            # --- only take first detected person ---
            xy = r.keypoints.xy[0].cpu().numpy().flatten()
            conf = r.keypoints.conf[0].cpu().numpy().flatten()

            # --- build 51 feature vector (x, y, conf for each 17 keypoints) ---
            features = np.zeros(51)
            for i in range(17):
                features[i*3] = xy[i*2]
                features[i*3+1] = xy[i*2+1]
                features[i*3+2] = conf[i]

            feature_names = [f'col_{i}' for i in range(51)]
            features_df = pd.DataFrame([features], columns=feature_names)

            # --- predict using classifier ---
            pred = clf.predict(features_df)[0]
            label = "Push-up" if pred == 1 else "Push-down"

            # --- annotate frame ---
            annotated = r.plot()
            cv2.putText(annotated, label, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # --- display or save output ---
            if supports_cv2_gui():
                cv2.imshow("Prediction", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                output_path = os.path.join(os.path.dirname(__file__), "prediction_result.jpg")
                cv2.imwrite(output_path, annotated)
                print(f"💾 Saved annotated image to: {output_path}")

            print(f"✅ Prediction: {label}")
            return label
        else:
            print("⚠️ No person detected.")
            return None


# === Example test ===
predict_pushup_stage("samplepushup.jpg")
