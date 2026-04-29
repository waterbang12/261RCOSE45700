# MediaPipe - Copyright 2019 The MediaPipe Authors (Apache License 2.0)
# https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE

import os
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from datetime import datetime

SAVE_DIR = 'saved_faces'
MODEL_PATH = 'hand_landmarker.task'
CONFIDENCE_THRESHOLD = 80
OCCLUSION_TRIGGER_FRAMES = 10
PERSON_GONE_TRIGGER_FRAMES = 30

os.makedirs(SAVE_DIR, exist_ok=True)

hand_options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=mp_vision.RunningMode.VIDEO
)
hands_detector = mp_vision.HandLandmarker.create_from_options(hand_options)


def log_event(**kwargs):
    parts = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"[EVENT] {parts}")


def hand_near_face(hand_results, face_region, frame_shape, margin_ratio=0.6):
    """손 랜드마크가 얼굴 영역 근처에 있는지 확인합니다."""
    if not hand_results.hand_landmarks or face_region is None:
        return False
    h_img, w_img = frame_shape[:2]
    fx, fy, fw, fh = face_region
    margin = int(max(fw, fh) * margin_ratio)
    rx1 = max(0, fx - margin)
    ry1 = max(0, fy - margin)
    rx2 = min(w_img, fx + fw + margin)
    ry2 = min(h_img, fy + fh + margin)

    for hand_landmarks in hand_results.hand_landmarks:
        for lm in hand_landmarks:
            hx, hy = int(lm.x * w_img), int(lm.y * h_img)
            if rx1 <= hx <= rx2 and ry1 <= hy <= ry2:
                return True
    return False


def cleanup_old_faces(days=7):
    cutoff = datetime.now().timestamp() - days * 86400
    removed = 0
    for filename in os.listdir(SAVE_DIR):
        if not filename.lower().endswith('.jpg'):
            continue
        try:
            base = filename[len('face_'):-len('.jpg')]
            parts = base.rsplit('_', 2)
            ts = datetime.strptime(f"{parts[-2]}_{parts[-1]}", '%Y%m%d_%H%M%S').timestamp()
            if ts < cutoff:
                os.remove(os.path.join(SAVE_DIR, filename))
                print(f"파기 완료 (7일 경과): {filename}")
                removed += 1
        except (ValueError, IndexError):
            pass
    if removed:
        print(f"총 {removed}개 파일 파기됨")


def load_training_data():
    faces, labels, label_map = [], [], {}
    current_label = 0
    name_groups = {}

    for filename in os.listdir(SAVE_DIR):
        if not filename.lower().endswith('.jpg'):
            continue
        base = filename[len('face_'):-len('.jpg')]
        parts = base.rsplit('_', 2)
        name = parts[0] if len(parts) >= 3 else base
        name_groups.setdefault(name, []).append(os.path.join(SAVE_DIR, filename))

    for name, paths in name_groups.items():
        label_map[current_label] = name
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (100, 100))
                faces.append(img)
                labels.append(current_label)
        current_label += 1

    if not faces:
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    print(f"인식기 학습 완료: {len(label_map)}명 ({len(faces)}장)")
    return recognizer, label_map


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

cleanup_old_faces()
recognizer, label_map = load_training_data()
if recognizer:
    print(f"등록된 고객: {list(label_map.values())}")
else:
    print("저장된 고객 없음 - 감지만 실행됩니다.")

print("시작 - 종료: 'q' | 저장: 's' | 재학습: 'r'")

waiting_consent = False
consent_face_img = None
consent_frame_display = None

# 상태 추적
last_face_region = None
last_known_customer = None
occlusion_frame_count = 0
occlusion_active = False
occlusion_start_time = None
person_gone_count = 0
last_cleanup_hour = datetime.now().hour


def put_text_centered(img, text, y, font, scale, color, thickness):
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_hour = datetime.now().hour
    if current_hour != last_cleanup_hour:
        cleanup_old_faces()
        recognizer, label_map = load_training_data()
        last_cleanup_hour = current_hour

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    hand_results = hands_detector.detect_for_video(mp_image, timestamp_ms)

    if not waiting_consent:
        current_customer = None

        for (x, y, w, h) in faces:
            face_gray = cv2.resize(gray[y:y+h, x:x+w], (100, 100))

            if recognizer and label_map:
                label_id, confidence = recognizer.predict(face_gray)
                if confidence < CONFIDENCE_THRESHOLD:
                    current_customer = label_map.get(label_id, 'Unknown')
                    color = (0, 255, 0)
                    text = f'{current_customer} ({int(confidence)})'
                else:
                    color = (0, 100, 255)
                    text = 'Unknown'
            else:
                color = (0, 255, 0)
                text = 'Face'

            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 손 랜드마크 표시
        if hand_results.hand_landmarks:
            h_img, w_img = frame.shape[:2]
            for hand_lms in hand_results.hand_landmarks:
                for lm in hand_lms:
                    cx, cy = int(lm.x * w_img), int(lm.y * h_img)
                    cv2.circle(display, (cx, cy), 4, (0, 215, 255), -1)

        # 고객 등장 이벤트
        if current_customer and current_customer != last_known_customer:
            log_event(event='face_appeared', customer=current_customer,
                      timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            last_known_customer = current_customer

        # 마지막 인식 얼굴 위치 갱신
        if current_customer and len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            last_face_region = (fx, fy, fw, fh)

        # 가림/퇴장 감지
        known_customer_missing = last_known_customer is not None and current_customer is None

        if known_customer_missing and last_face_region is not None:
            hand_detected = hand_near_face(hand_results, last_face_region, frame.shape)

            if hand_detected:
                occlusion_frame_count += 1
                person_gone_count = 0
                if occlusion_frame_count >= OCCLUSION_TRIGGER_FRAMES and not occlusion_active:
                    occlusion_active = True
                    occlusion_start_time = datetime.now()
                    log_event(event='face_occlusion_start',
                              customer=last_known_customer,
                              timestamp=occlusion_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                              description='손/얼굴 가림 감지')
            else:
                occlusion_frame_count = max(0, occlusion_frame_count - 1)
                person_gone_count += 1
                if person_gone_count >= PERSON_GONE_TRIGGER_FRAMES:
                    if occlusion_active:
                        duration = round((datetime.now() - occlusion_start_time).total_seconds(), 1)
                        log_event(event='face_occlusion_end',
                                  customer=last_known_customer,
                                  timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  duration_sec=duration)
                        occlusion_active = False
                    log_event(event='customer_left',
                              customer=last_known_customer,
                              timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    last_known_customer = None
                    last_face_region = None
                    occlusion_frame_count = 0
                    person_gone_count = 0

        elif current_customer is not None:
            if occlusion_active:
                duration = round((datetime.now() - occlusion_start_time).total_seconds(), 1)
                log_event(event='face_occlusion_end',
                          customer=last_known_customer,
                          timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          duration_sec=duration)
            occlusion_frame_count = 0
            occlusion_active = False
            person_gone_count = 0

        if occlusion_active:
            cv2.putText(display, '! Occlusion Detected', (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

        cv2.putText(display, f'Faces: {len(faces)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display, "'s': save  'r': retrain  'q': exit", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Face Detection', display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            consent_face_img = frame[y:y+h, x:x+w].copy()
            consent_frame_display = display.copy()
            waiting_consent = True

        elif key == ord('r'):
            recognizer, label_map = load_training_data()
            print(f"재학습 완료: {list(label_map.values())}")

        elif key == ord('q'):
            break

    else:
        overlay = consent_frame_display.copy()
        h_frame, w_frame = overlay.shape[:2]
        dark = overlay.copy()
        cv2.rectangle(dark, (0, 0), (w_frame, h_frame), (0, 0, 0), -1)
        cv2.addWeighted(dark, 0.5, overlay, 0.5, 0, overlay)

        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = [
            ('[Personal Information Collection Notice]', 0.6, (0, 255, 255)),
            ('Collector: PA', 0.5, (200, 200, 200)),
            ('Purpose: Customer service', 0.5, (200, 200, 200)),
            ('Items: Photo, Name, Order history', 0.5, (200, 200, 200)),
            ('Retention: 7 days, then deleted', 0.5, (200, 200, 200)),
            ('Third-party sharing: None', 0.5, (200, 200, 200)),
            ('Refusal: No disadvantage', 0.5, (200, 200, 200)),
            ('Do you agree?  [y] Yes   [n] No', 0.65, (255, 255, 255)),
        ]
        start_y = h_frame // 2 - 130
        for i, (text, scale, color) in enumerate(lines):
            put_text_centered(overlay, text, start_y + i * 38, font, scale, color, 1)

        cv2.imshow('Face Detection', overlay)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('y'):
            waiting_consent = False
            customer_name = ''
            typing = True
            while typing:
                input_overlay = consent_frame_display.copy()
                dark = input_overlay.copy()
                cv2.rectangle(dark, (0, 0), (w_frame, h_frame), (0, 0, 0), -1)
                cv2.addWeighted(dark, 0.5, input_overlay, 0.5, 0, input_overlay)

                put_text_centered(input_overlay, 'enter the name',
                                  h_frame // 2 - 40, font, 0.9, (255, 255, 255), 2)
                cv2.rectangle(input_overlay,
                              (w_frame // 2 - 180, h_frame // 2 - 10),
                              (w_frame // 2 + 180, h_frame // 2 + 45), (255, 255, 255), 2)
                cv2.putText(input_overlay, customer_name,
                            (w_frame // 2 - 170, h_frame // 2 + 35),
                            font, 1.0, (0, 255, 255), 2)
                put_text_centered(input_overlay, 'Enter: accept  ESC: cancel',
                                  h_frame // 2 + 80, font, 0.65, (200, 200, 200), 1)

                cv2.imshow('Face Detection', input_overlay)
                k = cv2.waitKey(0) & 0xFF

                if k == 13:
                    if customer_name.strip():
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = os.path.join(SAVE_DIR, f'face_{customer_name.strip()}_{timestamp}.jpg')
                        cv2.imwrite(filename, consent_face_img)
                        print(f"저장 완료: {filename}")
                        recognizer, label_map = load_training_data()
                    typing = False
                elif k == 27:
                    print("저장 취소됨")
                    typing = False
                elif k == 8:
                    customer_name = customer_name[:-1]
                elif 32 <= k <= 126:
                    customer_name += chr(k)

        elif key == ord('n'):
            print("저장 거절됨")
            waiting_consent = False

        elif key == ord('q'):
            break

cap.release()
hands_detector.__exit__(None, None, None)
cv2.destroyAllWindows()
