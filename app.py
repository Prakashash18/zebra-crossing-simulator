from flask import Flask, render_template, Response, jsonify, request
import torch
import cv2
import numpy as np
from pathlib import Path
import time
import traceback
import base64

# Import from utils module
from utils import get_keypoints, extract_features_directly, draw_skeleton, POSE_CLASSES
from train import PoseClassifier

app = Flask(__name__)
app.static_folder = "app/static"
app.template_folder = "app/templates"

# Check if model files exist
model_path = Path('models/pose_classifier.pth')
scaler_path = Path('models/scaler.npy')

if not (model_path.exists() and scaler_path.exists()):
    raise FileNotFoundError("Required model files not found. Please train the model first.")

# Load the model and scaler
def load_model_and_scaler():
    """Load the trained model and scaler"""
    # Load scaler parameters
    print("Loading scaler parameters...")
    scaler_params = np.load(scaler_path, allow_pickle=True).item()
    
    # Create and load model
    print("Loading model...")
    # Dynamic input size based on scaler
    input_size = scaler_params['mean'].shape[0]
    model = PoseClassifier(input_size, len(POSE_CLASSES))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model, scaler_params

# Create a temporal filtering buffer
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

    def has_stable_pose(self):
        """Check if the buffer has any stable poses."""
        return len(self.buffer) > 0
    
    def get_most_recent_stable_pose(self):
        """Get the most recent stable pose from the buffer."""
        if not self.buffer:
            return {'pose_name': 'Looking Straight', 'confidence': 0.5}
        
        # Count occurrences of each pose
        pose_counts = {}
        for pose in self.buffer:
            pose_name = pose['pose_name']
            if pose_name in pose_counts:
                pose_counts[pose_name] += 1
            else:
                pose_counts[pose_name] = 1
        
        # Find the most common pose
        most_common_pose = max(pose_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate average confidence for this pose
        confidence_sum = 0
        count = 0
        for pose in self.buffer:
            if pose['pose_name'] == most_common_pose:
                confidence_sum += pose['confidence']
                count += 1
        
        avg_confidence = confidence_sum / count if count > 0 else 0.5
        
        return {'pose_name': most_common_pose, 'confidence': avg_confidence}

# Function to preprocess features
def preprocess_features(features, scaler_params):
    """Scale features using saved scaler parameters"""
    # Check for dimension mismatch
    if features.shape[1] != scaler_params['mean'].shape[0]:
        print(f"ERROR: Feature dimension ({features.shape[1]}) doesn't match "
              f"scaler dimension ({scaler_params['mean'].shape[0]})")
        raise ValueError("Feature dimension mismatch")
        
    scaled = (features - scaler_params['mean']) / scaler_params['scale']
    return scaled

# Initialize model and buffer
model, scaler_params = load_model_and_scaler()
pose_buffer = PoseBuffer(size=5, threshold=0.5)  # Lower threshold for more responsive detection

# Global variable to store the latest frame from the frontend
latest_frame = None
last_frame_time = 0

# This function is no longer needed since we'll get frames from the frontend
# def get_camera():
#     global camera
#     if camera is None:
#         camera = cv2.VideoCapture(0)
#     return camera

# This function is still needed for cleanup
def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def predict_pose(frame):
    """
    Predict pose from a frame using the models.
    Returns: annotated frame, pose class index, and confidence
    """
    if frame is None:
        print("Warning: Frame is None, cannot predict pose")
        return None, None, 0.0
        
    # Mirror the frame for more intuitive interaction
    frame = cv2.flip(frame, 1)
    
    # Get keypoints with confidence
    keypoints, detection_conf = get_keypoints(frame)
    
    # Default values
    pose_idx = None
    pose_name = None
    confidence_val = 0.0
    
    if keypoints is not None:
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
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                    
                    pose_idx = predicted.item()
                    pose_name = POSE_CLASSES[pose_idx]
                    confidence_val = confidence.item()
                    
                    print(f"Detected pose: {pose_name} with confidence: {confidence_val:.2f}")
                    
                    # Only consider predictions with sufficient confidence
                    min_confidence = 0.4  # Even lower threshold for testing
                    if confidence_val >= min_confidence:
                        # Add to buffer for temporal filtering
                        pose_buffer.add(pose_name, confidence_val)
            except ValueError as e:
                print(f"Error in preprocessing: {str(e)}")
    
    # Get stabilized pose from buffer (temporal filtering)
    stable_pose, stable_conf = pose_buffer.get_majority_pose()
    
    # Use stabilized pose if available
    if stable_pose:
        pose_name = stable_pose
        confidence_val = stable_conf
        
        # Find the index for the pose name
        for idx, name in POSE_CLASSES.items():
            if name == pose_name:
                pose_idx = idx
                break
                
        print(f"Stable pose: {pose_name} with confidence: {confidence_val:.2f}")
    
    # Draw skeleton with pose information
    annotated_frame = draw_skeleton(
        frame, keypoints, pose_name, confidence_val
    )
    
    return annotated_frame, pose_idx, confidence_val

# New endpoint to receive frames from the frontend
@app.route('/submit_frame', methods=['POST'])
def submit_frame():
    global latest_frame, last_frame_time
    
    try:
        # Get the frame data from the request
        frame_data = request.json.get('frame')
        
        if not frame_data:
            return jsonify({
                'success': False,
                'error': 'No frame data received'
            })
        
        # Remove the data URL prefix
        if frame_data.startswith('data:image/jpeg;base64,'):
            frame_data = frame_data.replace('data:image/jpeg;base64,', '')
        elif frame_data.startswith('data:image/png;base64,'):
            frame_data = frame_data.replace('data:image/png;base64,', '')
        
        # Decode the base64 string
        img_bytes = base64.b64decode(frame_data)
        
        # Convert to numpy array
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        
        # Decode the image
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({
                'success': False,
                'error': 'Failed to decode frame'
            })
        
        # Update the latest frame
        latest_frame = frame
        last_frame_time = time.time()
        
        # Process the frame and get pose information
        annotated_frame, pose_idx, confidence = predict_pose(frame)
        
        # Only return pose information, not the full frame
        if pose_idx is not None:
            pose_name = POSE_CLASSES[pose_idx]
            return jsonify({
                'success': True,
                'pose_idx': pose_idx,
                'pose_name': pose_name,
                'confidence': float(confidence)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No pose detected'
            })
            
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error processing frame: {str(e)}'
        }), 500

# Modified endpoint for video feed - will use annotated frames received from frontend
def generate_frames():
    global latest_frame
    
    while True:
        if latest_frame is not None:
            # We already have the latest frame, process it
            annotated_frame, _, _ = predict_pose(latest_frame.copy())
            
            if annotated_frame is not None:
                # Encode and yield the frame
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Generate a blank frame with a message if no frame is available
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "Waiting for camera...", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Add a small delay to avoid flooding the browser with frames
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_pose')
def get_pose():
    """
    Endpoint to get the current user pose.
    Returns pose name and confidence as JSON.
    """
    global latest_frame, last_frame_time
    
    try:
        # Add timeout mechanism to avoid hanging connections
        max_detection_time = 2.0  # seconds
        start_time = time.time()
        
        # Check if we have a recent frame (within the last 5 seconds)
        frame_age = time.time() - last_frame_time
        if latest_frame is None or frame_age > 5.0:
            print("No recent frame available")
            return jsonify({
                'success': False,
                'error': 'No recent camera frame available. Please ensure your camera is active.'
            })
        
        # Process frame for pose detection with timeout handling
        try:
            # Use existing prediction function with the latest frame from frontend
            annotated_frame, pose_idx, confidence = predict_pose(latest_frame.copy())
            
            # Check if pose detection is taking too long
            if time.time() - start_time > max_detection_time:
                print("Warning: Pose detection taking too long")
                
                # Get stable pose from buffer as fallback
                stable_pose, stable_conf = pose_buffer.get_majority_pose()
                if stable_pose:
                    print(f"Using fallback stable pose: {stable_pose}")
                    # Find the index for the stable pose name
                    for idx, name in POSE_CLASSES.items():
                        if name == stable_pose:
                            pose_idx = idx
                            confidence = stable_conf
                            break
                            
            # If valid pose detected
            if pose_idx is not None and confidence > 0.4:  # Lower threshold for testing
                pose_name = POSE_CLASSES[pose_idx]
                
                print(f"Returning pose: {pose_name} with confidence: {confidence:.2f}")
                
                return jsonify({
                    'success': True,
                    'pose_idx': pose_idx,
                    'pose_name': pose_name,
                    'confidence': float(confidence)  # Ensure it's a Python float for JSON
                })
                
            # If no valid pose detected, use the most recent from buffer
            stable_pose, stable_conf = pose_buffer.get_majority_pose()
            if stable_pose:
                # Find the index for the pose name
                for idx, name in POSE_CLASSES.items():
                    if name == stable_pose:
                        pose_idx = idx
                        break
                        
                print(f"Returning stable pose: {stable_pose} with confidence: {stable_conf:.2f}")
                
                return jsonify({
                    'success': True,
                    'pose_idx': pose_idx,
                    'pose_name': stable_pose,
                    'confidence': float(stable_conf),
                    'is_fallback': True
                })
                
        except Exception as e:
            print(f"Error during pose detection: {str(e)}")
            traceback.print_exc()
            
            # Try to return the last stable pose from buffer
            stable_pose, stable_conf = pose_buffer.get_majority_pose()
            if stable_pose:
                # Find the index for the pose name
                for idx, name in POSE_CLASSES.items():
                    if name == stable_pose:
                        pose_idx = idx
                        break
                        
                print(f"Returning stable pose after error: {stable_pose}")
                
                return jsonify({
                    'success': True,
                    'pose_idx': pose_idx,
                    'pose_name': stable_pose,
                    'confidence': float(stable_conf),
                    'is_fallback': True,
                    'had_error': True
                })
            
            # If no fallback available, return error
            return jsonify({
                'success': False,
                'error': f'Error during pose detection: {str(e)}'
            })
        
        # If we get here, no pose was detected
        print("No pose detected with sufficient confidence")
        return jsonify({
            'success': False,
            'error': 'No pose detected with sufficient confidence'
        })
        
    except Exception as e:
        # Catch-all for any other errors
        print(f"Server error in get_pose: {str(e)}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/simulator')
def simulator():
    return render_template('simulator.html')


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5001, debug=True)
    finally:
        # No need to release the camera as we're not using it directly
        pass 