import cv2
import mediapipe as mp
import time

# Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# State flags
countdown_started = False
countdown_start_time = None
image_saved = False

# Side-pose detection
def is_side_pose(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Shoulders nearly on top of each other in X → person is sideways
    x_diff = abs(left_shoulder.x - right_shoulder.x)
    y_diff = abs(left_shoulder.y - right_shoulder.y)

    return x_diff < 0.1 and y_diff < 0.2

# Full body visible check
def is_full_body_visible(landmarks):
    nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
    left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y

    head_near_top = nose_y < 0.2
    feet_near_bottom = left_ankle_y > 0.8 and right_ankle_y > 0.8

    return head_near_top and feet_near_bottom

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
        side_pose = is_side_pose(landmarks)

        if side_pose and full_body:
            if not countdown_started:
                countdown_started = True
                countdown_start_time = time.time()

            # Countdown logic
            seconds_passed = int(time.time() - countdown_start_time)
            countdown = 3 - seconds_passed
            if countdown > 0:
                cv2.putText(frame, f"Taking side photo in {countdown}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            elif not image_saved:
                filename = f"captured/side_pose_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Image saved: {filename}")
                image_saved = True
                break
        else:
            countdown_started = False
            countdown_start_time = None
            image_saved = False

        # Status text
        if not full_body:
            cv2.putText(frame, "Make sure your head and feet are visible", (30, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif not side_pose:
            cv2.putText(frame, "Please turn sideways", (30, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Side Pose Full Body Capture", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
