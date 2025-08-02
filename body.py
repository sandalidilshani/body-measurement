import cv2
import numpy as np
import mediapipe as mp
import torch
from flask import Flask, request, jsonify
import torch.nn.functional as F

app = Flask(__name__)

# Load pose detection and holistic model
pose_model = mp.solutions.pose.Pose(model_complexity=2)
holistic_model = mp.solutions.holistic.Holistic()

# Constants
A4_WIDTH_CM = 21.0
DEFAULT_FOCAL_LENGTH = 600
DEFAULT_USER_HEIGHT_CM = 152.0

# Load AI depth estimation model
def load_depth_estimator():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    return model

depth_estimator = load_depth_estimator()

# Calibrate focal length using a known object (like A4 paper)
def calibrate_focal_length(image, real_width_cm, object_width_px):
    if object_width_px:
        return (object_width_px * DEFAULT_FOCAL_LENGTH) / real_width_cm
    return DEFAULT_FOCAL_LENGTH

# Try to detect A4 paper or similar object in the image
def detect_scale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        focal = calibrate_focal_length(image, A4_WIDTH_CM, w)
        scale = A4_WIDTH_CM / w
        return scale, focal
    return 0.05, DEFAULT_FOCAL_LENGTH

# Estimate depth map for an image
def get_depth_map(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    tensor = torch.tensor(rgb_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(384, 384), mode="bilinear", align_corners=False)
    with torch.no_grad():
        depth = depth_estimator(tensor)
    return depth.squeeze().numpy()

# Convert pixel measurements to cm
def pixel_to_cm(value, scale):
    return round(value * scale, 2)

# Calculate circumference from width and depth
def get_circumference(width_px, scale, depth_ratio=1.0):
    width_cm = width_px * scale
    depth_cm = width_cm * depth_ratio * 0.7
    a = width_cm / 2
    b = depth_cm / 2
    return round(2 * np.pi * np.sqrt((a**2 + b**2) / 2), 2)

# Validate front image contains person and body

def validate_front_image(image):
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with mp.solutions.holistic.Holistic(static_image_mode=True) as holistic:
            results = holistic.process(rgb)

        if not results.pose_landmarks:
            return False, "No person detected."

        landmarks = results.pose_landmarks.landmark
        required = [
            mp.solutions.holistic.PoseLandmark.NOSE,
            mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER,
            mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER
        ]

        for lm in required:
            mark = landmarks[lm.value]
            if mark.visibility < 0.5:
                return False, "Body not fully visible."

        return True, "Valid image"
    except:
        return False, "Image processing error."

@app.route("/upload", methods=["POST"])
def process_images():
    if "front" not in request.files:
        return jsonify({"error": "Front image required."}), 400

    front_img = request.files["front"]
    image_data = np.frombuffer(front_img.read(), np.uint8)
    front = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    is_valid, message = validate_front_image(front)
    if not is_valid:
        return jsonify({"error": message}), 400

    user_height = request.form.get("height_cm")
    try:
        user_height = float(user_height)
    except:
        user_height = DEFAULT_USER_HEIGHT_CM

    image_height, image_width = front.shape[:2]
    rgb = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(rgb)

    if results.pose_landmarks:
        top = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE.value].y * image_height
        bottom = max(
            results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y,
            results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y
        ) * image_height

        height_px = abs(bottom - top)
        scale_factor = user_height / height_px if height_px else 0.05
    else:
        scale_factor, _ = detect_scale(front)

    # Dummy measurement (replace with actual later)
    measurements = {
        "shoulder_width": pixel_to_cm(100, scale_factor),
        "waist_circumference": get_circumference(80, scale_factor),
        "hip_circumference": get_circumference(90, scale_factor),
    }

    return jsonify({
        "measurements": measurements,
        "scale": scale_factor,
        "user_height": user_height
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
