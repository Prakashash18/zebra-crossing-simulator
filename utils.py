import numpy as np
import cv2
from ultralytics import YOLO

# Initialize YOLOv11 model
model = YOLO('yolo11n-pose.pt')

POSE_CLASSES = {
    0: 'Looking Left',
    1: 'Looking Right',
    2: 'Looking Straight',
    3: 'Looking Down',
    4: 'Raise Right Hand',
    5: 'Raise Left Hand'
}

# Define colors for each pose class (BGR format)
POSE_COLORS = {
    0: (255, 130, 0),    # Orange for Looking Left
    1: (0, 180, 255),    # Yellow for Looking Right
    2: (0, 255, 0),      # Green for Looking Straight
    3: (255, 0, 0),      # Blue for Looking Down
    4: (180, 0, 255),    # Purple for Raise Right Hand
    5: (0, 0, 255)       # Red for Raise Left Hand
}

# Minimum confidence thresholds for keypoint detection and pose classification
DETECTION_THRESHOLD = 0.5    # Overall detection confidence
KEYPOINT_THRESHOLD = 0.5     # Individual keypoint confidence
MIN_KEYPOINTS_REQUIRED = 10  # Minimum number of keypoints required for valid pose

def get_keypoints(frame):
    """
    Extract keypoints from a frame using YOLOv11
    Returns keypoints and overall detection confidence
    """
    try:
        results = model(frame, verbose=False)
        
        # Check if any detections were made
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, 0.0
            
        # Get detection confidence
        detection_conf = float(results[0].boxes.conf[0])
        
        # If detection confidence is too low, return None
        if detection_conf < DETECTION_THRESHOLD:
            return None, detection_conf
            
        # Check if keypoints were detected
        if (not hasattr(results[0], 'keypoints') or 
                results[0].keypoints is None):
            return None, detection_conf
            
        # Check if keypoints data exists and is not empty
        if len(results[0].keypoints.data) == 0:
            return None, detection_conf
        
        # Get the keypoints of the first person detected
        keypoints = results[0].keypoints.data[0].cpu().numpy()
        
        # Verify keypoints shape
        if keypoints.shape[0] == 0:
            return None, detection_conf
        
        # Count keypoints with sufficient confidence
        valid_keypoints = np.sum(keypoints[:, 2] > KEYPOINT_THRESHOLD)
        if valid_keypoints < MIN_KEYPOINTS_REQUIRED:
            return None, detection_conf
            
        return keypoints, detection_conf
        
    except Exception as e:
        print(f"Error in keypoint detection: {str(e)}")
        return None, 0.0

def extract_features_directly(keypoints):
    """
    Extract features directly from keypoints in one step
    YOLOv11 provides more accurate keypoints which we can use better
    """
    if keypoints is None:
        return None
    
    features = []
    
    # Keypoint indices based on COCO dataset used by YOLOv11
    # 0: Nose, 1: Left Eye, 2: Right Eye, 
    # 5: Left Shoulder, 6: Right Shoulder
    # 7: Left Elbow, 8: Right Elbow
    
    # Check if critical keypoints have sufficient confidence
    critical_keypoints = [0, 1, 2, 5, 6]  # Nose, eyes, shoulders
    for kp_idx in critical_keypoints:
        if kp_idx >= len(keypoints) or keypoints[kp_idx][2] < KEYPOINT_THRESHOLD:
            return None
    
    # Head orientation features (more comprehensive)
    nose = keypoints[0][:2]  # x, y of nose
    left_eye = keypoints[1][:2]
    right_eye = keypoints[2][:2]
    left_ear = keypoints[3][:2] if keypoints.shape[0] > 3 else nose
    right_ear = keypoints[4][:2] if keypoints.shape[0] > 4 else nose
    
    # Normalized head keypoints relative to nose
    features.extend(left_eye - nose)    # Left eye relative to nose
    features.extend(right_eye - nose)   # Right eye relative to nose
    
    # Add ear positions for better head orientation detection
    if keypoints.shape[0] > 4:
        features.extend(left_ear - nose)    # Left ear relative to nose
        features.extend(right_ear - nose)   # Right ear relative to nose
    
    # Shoulder positions for posture detection
    left_shoulder = keypoints[5][:2]
    right_shoulder = keypoints[6][:2]
    shoulder_center = (left_shoulder + right_shoulder) / 2
    
    # Head position relative to shoulders
    features.extend(nose - shoulder_center)
    
    # Arm positions for hand raising detection
    if keypoints.shape[0] > 8:
        left_elbow = keypoints[7][:2]
        right_elbow = keypoints[8][:2]
        
        # Check if wrists are detected with sufficient confidence
        left_wrist = (keypoints[9][:2] if keypoints.shape[0] > 9 and 
                     keypoints[9][2] > KEYPOINT_THRESHOLD else left_elbow)
        
        right_wrist = (keypoints[10][:2] if keypoints.shape[0] > 10 and 
                      keypoints[10][2] > KEYPOINT_THRESHOLD else right_elbow)
        
        # Normalized arm keypoints for hand raising detection
        # Left elbow height relative to shoulder
        features.extend([left_elbow[1] - left_shoulder[1]])  
        # Right elbow height relative to shoulder
        features.extend([right_elbow[1] - right_shoulder[1]])  
        # Left wrist height relative to shoulder
        features.extend([left_wrist[1] - left_shoulder[1]])  
        # Right wrist height relative to shoulder
        features.extend([right_wrist[1] - right_shoulder[1]])  
    
    # Calculate head tilt angle
    head_tilt = np.arctan2(
        right_eye[1] - left_eye[1],
        right_eye[0] - left_eye[0]
    )
    features.append(head_tilt)
    
    return np.array(features)

def draw_skeleton(frame, keypoints, pose_class=None, detection_conf=0.0):
    """
    Draw skeleton on frame without text overlay
    """
    if keypoints is None:
        # Just return the original frame without any text overlay
        return frame
    
    # Draw keypoints with larger radius
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > KEYPOINT_THRESHOLD:
            # Color-code keypoints based on importance
            if i in [0, 1, 2]:  # Nose and eyes
                color = (0, 0, 255)  # Red for face
            elif i in [5, 6]:  # Shoulders
                color = (255, 0, 0)  # Blue for shoulders
            elif i in [7, 8, 9, 10]:  # Arms
                color = (0, 255, 0)  # Green for arms
            else:
                color = (0, 255, 255)  # Yellow for others
                
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
    
    # Draw connections between keypoints for better visualization
    # Connections are defined as pairs of keypoint indices
    connections = [
        (0, 1), (0, 2),  # Nose to eyes
        (1, 3), (2, 4),  # Eyes to ears
        (5, 6),          # Shoulders
        (5, 7), (6, 8),  # Arms
        (7, 9), (8, 10)  # Forearms
    ]
    
    # Only draw connections if both keypoints are present
    for connection in connections:
        if len(keypoints) > max(connection):
            pt1 = (int(keypoints[connection[0]][0]), 
                  int(keypoints[connection[0]][1]))
            pt2 = (int(keypoints[connection[1]][0]), 
                  int(keypoints[connection[1]][1]))
            if (keypoints[connection[0]][2] > KEYPOINT_THRESHOLD and 
                    keypoints[connection[1]][2] > KEYPOINT_THRESHOLD):
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
    
    # No text overlay for pose class or confidence, as this is displayed in the frontend
    return frame 