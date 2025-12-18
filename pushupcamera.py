import cv2
import joblib
from ultralytics import YOLO
import numpy as np
import os
import time
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def supports_cv2_gui():
    """Return True if cv2.imshow is usable in this environment."""
    try:
        test_img = np.zeros((2, 2, 3), dtype=np.uint8)
        cv2.namedWindow("_cv2_gui_test", cv2.WINDOW_NORMAL)
        cv2.imshow("_cv2_gui_test", test_img)
        cv2.waitKey(1)
        cv2.destroyWindow("_cv2_gui_test")
        return True
    except cv2.error:
        return False
    except Exception:
        return False

# === CONFIGURATION ===
MODEL_PATH = "models/pushup_pose_classifier_merged.pkl"  # Your trained model
POSE_MODEL = "yolov8n-pose.pt"  
THRESHOLD = 0.4  # Optimized threshold

# === LOAD MODELS ===
print("Loading models...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at: {MODEL_PATH}")
    print("Please download squat_classifier2.pkl from Google Drive!")
    exit()

pose_model = YOLO(POSE_MODEL)
artifact = joblib.load(MODEL_PATH)

# Handle both dict and direct model formats
if isinstance(artifact, dict):
    classifier = artifact['model']
else:
    classifier = artifact
    
print("✅ Models loaded!")

# === INITIALIZE ===
rep_count = 0
last_state = None
state_history = []

# Check GUI support
has_gui = supports_cv2_gui()
writer = None
out_path = "pushup_counter_output.avi"

if not has_gui:
    print("💾 GUI not available. Saving video to:", out_path)
    print("Press Ctrl+C to stop")

# === START WEBCAM ===
cap = cv2.VideoCapture(0)  # 0 = default webcam, try 1 if 0 doesn't work

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✅ Webcam opened!")
if has_gui:
    print("Press 'q' to quit")
    print("Press 'r' to reset rep counter")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        h, w, _ = frame.shape
        
        # === POSE DETECTION ===
        results = pose_model(frame, verbose=False, max_det=1)
        
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            # Extract keypoints
            xy = results[0].keypoints.xy[0]
            conf = results[0].keypoints.conf[0]
            
            # Build feature vector (51 features)
            features = []
            for kp_xy, kp_conf in zip(xy, conf):
                nx = kp_xy[0].item() / w
                ny = kp_xy[1].item() / h
                features.extend([nx, ny, kp_conf.item()])
            
            # === PREDICTION WITH THRESHOLD 0.4 ===
            proba = classifier.predict_proba([features])[0, 1]
            current_state = 1 if proba >= THRESHOLD else 0  # 0=down, 1=up
            
            # === SMOOTHING ===
            state_history.append(current_state)
            if len(state_history) > 3:
                state_history.pop(0)
            smoothed_state = int(np.round(np.mean(state_history)))
            
            # === COUNT REPS (down → up transition) ===
            if last_state == 0 and smoothed_state == 1:
                rep_count += 1
            
            last_state = smoothed_state
            
            # === VISUALIZE ===
            # Draw skeleton
            annotated_frame = results[0].plot()
            
            # State colors
            state_text = "UP" if smoothed_state == 1 else "DOWN"
            state_color = (0, 255, 0) if smoothed_state == 1 else (0, 0, 255)
            
            # Rep counter (top-left, large text)
            cv2.putText(annotated_frame, f"REPS: {rep_count}",  
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
            
            # State indicator (top-right)
            cv2.putText(annotated_frame, state_text, 
                        (w - 180, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, state_color, 4)
            
            # Confidence bar (bottom)
            bar_width = int(proba * 400)
            cv2.rectangle(annotated_frame, (20, h - 50), (20 + bar_width, h - 30), 
                        (0, 255, 0), -1)
            cv2.rectangle(annotated_frame, (20, h - 50), (420, h - 30), 
                        (255, 255, 255), 2)  # Border
            cv2.putText(annotated_frame, f"Confidence: {proba*100:.1f}%", 
                        (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if has_gui:
                cv2.imshow("Push-up Rep Counter", annotated_frame)
            else:
                # Lazy-init writer
                if writer is None:
                    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
                    manual_fourcc = (ord("X") | (ord("V") << 8) | (ord("I") << 16) | (ord("D") << 24))
                    fourcc = manual_fourcc
                    if callable(fourcc_fn):
                        try:
                            ftmp = fourcc_fn(*'XVID')
                            if isinstance(ftmp, int):
                                fourcc = ftmp
                        except Exception:
                            pass
                    writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
                if writer is not None:
                    writer.write(annotated_frame)
        else:
            # No person detected
            cv2.putText(frame, "No person detected - Step into frame", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if has_gui:
                cv2.imshow("Push-up Rep Counter", frame)
            else:
                if writer is None:
                    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
                    manual_fourcc = (ord("X") | (ord("V") << 8) | (ord("I") << 16) | (ord("D") << 24))
                    fourcc = manual_fourcc
                    if callable(fourcc_fn):
                        try:
                            ftmp = fourcc_fn(*'XVID')
                            if isinstance(ftmp, int):
                                fourcc = ftmp
                        except Exception:
                            pass
                    writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
                if writer is not None:
                    writer.write(frame)
        
        # Keyboard controls
        if has_gui:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                rep_count = 0
                print("Rep counter reset!")
        else:
            time.sleep(0.01)

except KeyboardInterrupt:
    print("\n⏹️ Stopped by user")
finally:
    cap.release()
    if writer is not None:
        writer.release()
    if has_gui:
        cv2.destroyAllWindows()
    print(f"\n🎉 Final rep count: {rep_count}")