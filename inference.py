import cv2
import numpy as np
import torch
import os
import time
from utils import (
    get_keypoints, POSE_CLASSES, draw_skeleton, extract_features_directly,
    DETECTION_THRESHOLD
)
from train import PoseClassifier


def check_model_files():
    """Check if necessary model files exist"""
    if not os.path.exists('models/pose_classifier.pth'):
        print("Error: Model file 'models/pose_classifier.pth' not found.")
        print("Please run data collection (collect_data.py) and"
              " training (train.py) first.")
        return False
        
    if not os.path.exists('models/scaler.npy'):
        print("Error: Scaler file 'models/scaler.npy' not found.")
        print("Please run data collection (collect_data.py) and"
              " training (train.py) first.")
        return False
        
    return True


def load_model_and_scaler():
    """Load the trained model and scaler"""
    # Load scaler parameters
    print("Loading scaler parameters...")
    scaler_params = np.load('models/scaler.npy', allow_pickle=True).item()
    print(f"Scaler params shape: mean={scaler_params['mean'].shape}, "
          f"scale={scaler_params['scale'].shape}")
    
    # Create and load model
    print("Loading model...")
    # Dynamic input size based on scaler
    input_size = scaler_params['mean'].shape[0]
    model = PoseClassifier(input_size, len(POSE_CLASSES))
    model.load_state_dict(torch.load('models/pose_classifier.pth'))
    model.eval()
    
    return model, scaler_params


def preprocess_features(features, scaler_params):
    """Scale features using saved scaler parameters"""
    # Check for dimension mismatch
    if features.shape[1] != scaler_params['mean'].shape[0]:
        print(f"ERROR: Feature dimension ({features.shape[1]}) doesn't match "
              f"scaler dimension ({scaler_params['mean'].shape[0]})")
        raise ValueError("Feature dimension mismatch")
        
    scaled = (features - scaler_params['mean']) / scaler_params['scale']
    return scaled


class PoseBuffer:
    """Buffer to store recent pose predictions for temporal filtering"""
    def __init__(self, size=5, threshold=0.6):
        self.buffer = []
        self.size = size
        self.threshold = threshold
        
    def add(self, pose_class, confidence):
        """Add new pose prediction to buffer"""
        self.buffer.append((pose_class, confidence))
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
    
    def get_majority_pose(self):
        """Get majority pose from buffer if above threshold"""
        if not self.buffer:
            return None, 0.0
            
        # Count occurrences of each pose class
        poses = {}
        for pose, conf in self.buffer:
            if pose not in poses:
                poses[pose] = []
            poses[pose].append(conf)
        
        # Find the pose with the most occurrences
        majority_pose = None
        max_count = 0
        max_conf = 0.0
        
        for pose, confs in poses.items():
            count = len(confs)
            avg_conf = sum(confs) / count
            
            if count > max_count or (count == max_count and avg_conf > max_conf):
                majority_pose = pose
                max_count = count
                max_conf = avg_conf
        
        # Only return if above threshold occurrence ratio
        if max_count / len(self.buffer) >= self.threshold:
            return majority_pose, max_conf
        return None, 0.0


def run_inference():
    """Run real-time pose classification with temporal filtering"""
    # Check if model files exist
    if not check_model_files():
        return
        
    # Load model and scaler
    try:
        model, scaler_params = load_model_and_scaler()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please ensure you have run data collection and training first.")
        return
    
    # Initialize video capture
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Wait for camera to initialize
    print("Warming up camera...")
    for _ in range(30):  # Wait for 30 frames
        ret, _ = cap.read()
        if not ret:
            print("Error: Could not read frame")
            cap.release()
            return
        cv2.waitKey(1)
    
    # Create pose buffer for temporal filtering
    pose_buffer = PoseBuffer(size=10, threshold=0.6)
    
    # FPS calculation variables
    fps_counter = 0
    fps_to_display = 0
    last_fps_time = time.time()
    
    print("Starting real-time pose classification with YOLOv11...")
    print(f"Detection threshold: {DETECTION_THRESHOLD}")
    print("Press 'q' to quit")
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)
        
        try:
            # Get keypoints with confidence
            keypoints, detection_conf = get_keypoints(frame)
            
            # Predicted pose and its confidence
            pose_class = None
            pose_conf = 0.0
            
            if keypoints is not None and detection_conf >= DETECTION_THRESHOLD:
                # Extract features directly from keypoints
                features = extract_features_directly(keypoints)
                
                if features is not None:
                    # Reshape features for preprocessing
                    features_reshaped = features.reshape(1, -1)
                    
                    try:
                        # Preprocess features
                        features = preprocess_features(
                            features_reshaped, scaler_params
                        )
                        
                        # Make prediction
                        with torch.no_grad():
                            features_tensor = torch.FloatTensor(features)
                            outputs = model(features_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            confidence, predicted = torch.max(probabilities, 1)
                            
                            pose_class = POSE_CLASSES[predicted.item()]
                            pose_conf = confidence.item()
                            
                            # Only consider predictions with sufficient confidence
                            if pose_conf < 0.7:  # Threshold for prediction confidence
                                pose_class = None
                            else:
                                # Add to buffer for temporal filtering
                                pose_buffer.add(pose_class, pose_conf)
                    except ValueError as e:
                        print(f"Error: {str(e)}")
                        print("Feature dimensions don't match the trained model.")
                        print("You need to retrain the model with the new features.")
                        print("Run collect_data.py and train.py again.")
                        break
            
            # Get stabilized pose from buffer (temporal filtering)
            stable_pose, stable_conf = pose_buffer.get_majority_pose()
            
            # Draw skeleton with either stable pose or current pose
            if stable_pose:
                draw_skeleton(frame, keypoints, stable_pose, detection_conf)
            else:
                draw_skeleton(frame, keypoints, None, detection_conf)
                
            # Calculate and display FPS
            fps_counter += 1
            if time.time() - last_fps_time >= 1.0:  # Update FPS every second
                fps_to_display = fps_counter
                fps_counter = 0
                last_fps_time = time.time()
                
            # Display FPS
            cv2.putText(
                frame, f"FPS: {fps_to_display}", 
                (frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
                
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            draw_skeleton(frame, None, None, 0.0)
        
        # Display frame
        cv2.imshow('YOLOv11 Pose Classification', frame)
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference() 