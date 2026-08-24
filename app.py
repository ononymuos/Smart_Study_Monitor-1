import os
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from ultralytics import YOLO
import pygame
import numpy as np

# ------------------- INITIALIZE AUDIO & MODELS -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pygame.mixer.init()
alarm_sleep = pygame.mixer.Sound(os.path.join(BASE_DIR, "alarm.mp3"))       # Sleep alarm
alarm_facehide = pygame.mixer.Sound(os.path.join(BASE_DIR, "faudio.mp3"))   # Face hidden/away alarm
alarm_phone = pygame.mixer.Sound(os.path.join(BASE_DIR, "paudio.mp3"))      # Phone alarm

current_playing = None

# Initialize Camera & AI Models
cap = cv2.VideoCapture(0)
face_detector = FaceMeshDetector(maxFaces=1)
phone_detector = YOLO("yolov8n.pt")
classNames = phone_detector.names

# ------------------- LANDMARK INDICES -------------------
# Sleep detection (Left eye)
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
FACE_LEFT = 130
FACE_RIGHT = 243

# 3D Head Pose Landmark Points for cv2.solvePnP
# 1: Nose tip, 152: Chin, 33: Left eye outer, 263: Right eye outer, 61: Left mouth corner, 291: Right mouth corner
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

# Standard 3D generic facial model points
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye outer corner
    (225.0, 170.0, -135.0),      # Right eye outer corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

# ------------------- REALISTIC THRESHOLDS & BUFFERS -------------------
# Assuming ~30 FPS webcam:
# 60 frames = ~2.0 seconds of continuous closed eyes (while upright)
# 120 frames = ~4.0 seconds of completely missing/covered face
SLEEP_THRESHOLD_FRAMES = 60
COVER_THRESHOLD_FRAMES = 120
READING_PITCH_THRESHOLD = -10.0   # Degrees (tilt downwards -> reading / writing notes)

closed_frames = 0
covered_frames = 0

def estimate_head_pose(face, img_shape):
    """Calculates head pitch, yaw, and roll angles in degrees using solvePnP."""
    h, w, _ = img_shape
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    image_points = []
    for idx in POSE_LANDMARKS:
        pt = face[idx]
        image_points.append([pt[0], pt[1]])
    image_points = np.array(image_points, dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = angles[0] * 360  # Negative = looking down, Positive = looking up
    yaw = angles[1] * 360    # Turning left / right
    roll = angles[2] * 360   # Tilting side to side
    return pitch, yaw, roll

# ------------------- MAIN LOOP -------------------
while True:
    success, img = cap.read()
    if not success:
        break

    # ------------------- 1. POSE & EYE DETECTION -------------------
    img, faces = face_detector.findFaceMesh(img, draw=False)
    is_sleepy = False
    is_face_covered = False
    is_reading = False
    pitch, yaw, roll = 0.0, 0.0, 0.0

    if faces:
        covered_frames = 0
        face = faces[0]

        # 1.1 Calculate Head Pitch (Angle)
        try:
            pitch, yaw, roll = estimate_head_pose(face, img.shape)
        except Exception:
            pitch = 0.0

        # If looking down into a notebook / desk
        if pitch < READING_PITCH_THRESHOLD:
            is_reading = True

        # 1.2 Calculate Eye Aspect Ratio
        eye_dist, _ = face_detector.findDistance(face[LEFT_EYE_TOP], face[LEFT_EYE_BOTTOM])
        face_dist, _ = face_detector.findDistance(face[FACE_LEFT], face[FACE_RIGHT])
        ratio = (eye_dist / face_dist) * 100

        # Smoothly accumulate closed eyes with decay buffer instead of hard 0 reset
        if not is_reading and ratio < 11.0:
            closed_frames = min(closed_frames + 2, SLEEP_THRESHOLD_FRAMES + 10)
        else:
            # Decay buffer: step down gradually so single-frame blinks/flickers don't reset to 0
            closed_frames = max(0, closed_frames - 1)

        if closed_frames >= SLEEP_THRESHOLD_FRAMES:
            is_sleepy = True

        # On-screen HUD metrics
        status_text = "Studying / Reading Notes" if is_reading else "Active / Looking Up"
        status_color = (0, 255, 0) if is_reading else (255, 255, 255)
        cvzone.putTextRect(img, f"Mode: {status_text}", (30, 40), scale=1, thickness=1, colorR=status_color)
        cvzone.putTextRect(img, f"Eye Ratio: {int(ratio)} | Pitch: {int(pitch)} deg", (30, 75), scale=1, thickness=1)

    else:
        # Face not detected (head completely down, obstructed, or left room)
        closed_frames = 0
        covered_frames += 1

        if covered_frames >= COVER_THRESHOLD_FRAMES:
            is_face_covered = True

        # Progress visual towards missing/covered warning
        sec_left = max(0, (COVER_THRESHOLD_FRAMES - covered_frames) // 30)
        cvzone.putTextRect(img, f"Face Not Detected... ({sec_left}s buffer)", (30, 40), scale=1, thickness=1, colorR=(50, 50, 50))

    # ------------------- 2. PHONE DETECTION -------------------
    results = phone_detector.predict(img, stream=True, verbose=False)
    phone_detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if classNames[cls_id] == "cell phone" and conf > 0.5:
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cvzone.putTextRect(img, f"Phone detected! {int(conf*100)}%", (x1, max(y1 - 10, 30)), scale=1, thickness=1, colorR=(255, 0, 255))

    # ------------------- 3. RESPONSIVE ALARM & DISPLAY LOGIC -------------------
    # Determine what alarm should be active right now
    desired_alarm = None
    if is_face_covered:
        desired_alarm = 'facehide'
    elif is_sleepy:
        desired_alarm = 'sleep'
    elif phone_detected:
        desired_alarm = 'phone'

    # State change: Start or stop sounds immediately
    if desired_alarm != current_playing:
        pygame.mixer.stop()  # Immediately cut off any running sound
        current_playing = desired_alarm

        if current_playing == 'facehide':
            alarm_facehide.play(-1)  # Loop until condition resolves
        elif current_playing == 'sleep':
            alarm_sleep.play(-1)     # Loop until condition resolves
        elif current_playing == 'phone':
            alarm_phone.play(-1)     # Loop until condition resolves

    # Display active warning overlay only while condition persists
    if current_playing == 'facehide':
        cvzone.putTextRect(img, "DONT COVER YOUR FACE!", (50, 130), scale=2, thickness=3, colorR=(0, 0, 255))
    elif current_playing == 'sleep':
        cvzone.putTextRect(img, "WAKE UP & STUDY!", (50, 130), scale=2, thickness=3, colorR=(0, 0, 255))
    elif current_playing == 'phone':
        cvzone.putTextRect(img, "PUT THE PHONE AWAY!", (50, 130), scale=2, thickness=3, colorR=(0, 165, 255))

    cv2.imshow("Smart Study Monitor", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pygame.mixer.stop()
cap.release()
cv2.destroyAllWindows()
