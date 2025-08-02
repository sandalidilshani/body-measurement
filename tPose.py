import cv2
import mediapipe as mp
import time
import os

# Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# State flags
countdown_started = False
countdown_start_time = None
t_pose_saved = False
side_pose_saved = False
current_stage = "t_pose"  # "t_pose" or "side_pose"

# T-pose detection
def is_t_pose(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    y_thresh = 0.05
    x_thresh = 0.3

    left_arm = abs(left_shoulder.y - left_elbow.y) < y_thresh and abs(left_elbow.y - left_wrist.y) < y_thresh
    right_arm = abs(right_shoulder.y - right_elbow.y) < y_thresh and abs(right_elbow.y - right_wrist.y) < y_thresh
    wide_enough = abs(left_wrist.x - right_wrist.x) > x_thresh

    return left_arm and right_arm and wide_enough

# Side pose detection (profile view with one arm visible)
def is_side_pose(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
    
    # Check if person is facing sideways
    # In side view, shoulders should be roughly aligned vertically (similar x position)
    shoulder_alignment = abs(left_shoulder.x - right_shoulder.x) < 0.15
    
    # Check if head is in profile (one ear more visible than the other)
    ear_visibility_diff = abs(left_ear.visibility - right_ear.visibility)
    profile_head = ear_visibility_diff > 0.3
    
    # Additional check: nose should be roughly aligned with one of the ears
    nose_ear_alignment = (abs(nose.x - left_ear.x) < 0.1) or (abs(nose.x - right_ear.x) < 0.1)
    
    return shoulder_alignment and (profile_head or nose_ear_alignment)

# Full body visible check
def is_full_body_visible(landmarks):
    nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
    left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y

    # y = 0 is top, y = 1 is bottom of image
    head_near_top = nose_y < 0.2
    feet_near_bottom = left_ankle_y > 0.8 and right_ankle_y > 0.8

    return head_near_top and feet_near_bottom

# Create output directory
if not os.path.exists("captured"):
    os.makedirs("captured")

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = result.pose_landmarks.landmark
        full_body = is_full_body_visible(landmarks)
        
        if current_stage == "t_pose":
            # T-pose capture stage
            t_pose = is_t_pose(landmarks)
            
            if t_pose and full_body:
                if not countdown_started:
                    countdown_started = True
                    countdown_start_time = time.time()

                # Countdown logic
                seconds_passed = int(time.time() - countdown_start_time)
                countdown = 3 - seconds_passed
                if countdown > 0:
                    cv2.putText(frame, f"Taking T-pose photo in {countdown}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                elif not t_pose_saved:
                    filename = f"captured/t_pose_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"T-pose image saved: {filename}")
                    t_pose_saved = True
                    current_stage = "side_pose"
                    countdown_started = False
                    countdown_start_time = None
                    # Give user time to read the message
                    time.sleep(1)
            else:
                countdown_started = False
                countdown_start_time = None

            # Status text for T-pose
            if not full_body:
                cv2.putText(frame, "Make sure your head and feet are visible", (30, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif not t_pose:
                cv2.putText(frame, "Please make a T-pose", (30, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
        elif current_stage == "side_pose":
            # Side pose capture stage
            side_pose = is_side_pose(landmarks)
            
            # Display instruction
            cv2.putText(frame, "T-pose captured! Now turn sideways", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            if side_pose and full_body:
                if not countdown_started:
                    countdown_started = True
                    countdown_start_time = time.time()

                # Countdown logic
                seconds_passed = int(time.time() - countdown_start_time)
                countdown = 3 - seconds_passed
                if countdown > 0:
                    cv2.putText(frame, f"Taking side photo in {countdown}", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                elif not side_pose_saved:
                    filename = f"captured/side_pose_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Side pose image saved: {filename}")
                    side_pose_saved = True
                    print("Both images captured successfully!")
                    break
            else:
                countdown_started = False
                countdown_start_time = None

            # Status text for side pose
            if not full_body:
                cv2.putText(frame, "Make sure your head and feet are visible", (30, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif not side_pose:
                cv2.putText(frame, "Please turn to your side (profile view)", (30, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show current stage indicator
    stage_text = f"Stage: {current_stage.replace('_', ' ').title()}"
    cv2.putText(frame, stage_text, (frame.shape[1] - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Pose Capture System", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

print("Capture session completed!")