# MediaPipe - Copyright 2019 The MediaPipe Authors (Apache License 2.0)
# https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE
#
# 필요 패키지:  pip install openai opencv-contrib-python mediapipe python-dotenv

import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경 변수 로드

os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if not os.environ.get('OPENAI_API_KEY'):
    raise EnvironmentError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from datetime import datetime
import time
import base64
import threading
from openai import OpenAI

SAVE_DIR             = 'saved_faces'
DANGER_SCREENSHOT_DIR = 'danger_screenshots'
MODEL_PATH           = 'hand_landmarker.task'
POSE_MODEL_PATH      = 'pose_landmarker_lite.task'
FACE_MODEL_PATH      = 'blaze_face_short_range.tflite'
CONFIDENCE_THRESHOLD = 80
OCCLUSION_TRIGGER_FRAMES  = 10
PERSON_GONE_TRIGGER_FRAMES = 30

# ── 위험 행동 감지 상수 ──────────────────────────────────────────
SWAY_WINDOW_FRAMES    = 15   # 흔들림 측정 슬라이딩 윈도우 크기 (프레임)
SWAY_X_THRESHOLD      = 30   # X축 어깨 중심 이동 범위 임계값 (픽셀)
SWAY_Y_THRESHOLD      = 25   # Y축 어깨 중심 이동 범위 임계값 (픽셀)
SWAY_TRIGGER_FRAMES   = 5    # 연속 흔들림 감지 프레임 수 (이 이상이면 캡처 시작)
SWAY_CAPTURE_COUNT    = 10   # 위험 상황 캡처 장 수
SWAY_CAPTURE_INTERVAL = 3    # 캡처 프레임 간격
OPENAI_API_COOLDOWN_SEC = 30 # OpenAI API 재호출 최소 대기 시간 (초)

# ── MediaPipe Pose 랜드마크 인덱스 ──────────────────────────────
_POSE_LEFT_SHOULDER  = 11
_POSE_RIGHT_SHOULDER = 12

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DANGER_SCREENSHOT_DIR, exist_ok=True)

# ── MediaPipe HandLandmarker ────────────────────────────────────
hand_options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=mp_vision.RunningMode.VIDEO
)
hands_detector = mp_vision.HandLandmarker.create_from_options(hand_options)

# ── MediaPipe PoseLandmarker (Tasks API) ───────────────────────
# pose_landmarker_lite.task 다운로드:
# https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
pose_detector = None
if os.path.exists(POSE_MODEL_PATH):
    _pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_detector = mp_vision.PoseLandmarker.create_from_options(_pose_options)
else:
    print(f"[POSE] {POSE_MODEL_PATH} 없음 → 몸 흔들림 감지 비활성화")
    print("  다운로드 후 스크립트와 같은 폴더에 저장하세요.")


# ═══════════════════════════════════════════════════════════════
#  고객 얼굴 인식 유틸
# ═══════════════════════════════════════════════════════════════

def log_event(**kwargs):
    parts = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"[EVENT] {parts}")


def hand_near_face(hand_results, face_region, frame_shape, margin_ratio=0.6):
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


def put_text_centered(img, text, y, font, scale, color, thickness):
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, y), font, scale, color, thickness)


def show_operator_notice() -> bool:
    """
    프로그램 시작 시 운영자에게 AI 분석 시스템을 고지한다.
    Enter → True (계속), Q → False (종료)
    """
    notice = np.zeros((620, 760, 3), dtype=np.uint8)
    font   = cv2.FONT_HERSHEY_SIMPLEX
    lines  = [
        ('[Operator Notice AI Safety Monitoring System]',  0.65, (0, 255, 255)),
        ('',                                                  0.4,  (255, 255, 255)),
        ('1. Face Recognition & Customer Info',               0.52, (200, 200, 200)),
        ('   Items : Face photo / Nickname / Order history',  0.44, (150, 150, 150)),
        ('   Stored locally  |  Auto-deleted after 7 days',  0.44, (150, 150, 150)),
        ('   Nickname & order history may be sent to',        0.44, (150, 150, 150)),
        ('   external AI API for service purposes',           0.44, (150, 150, 150)),
        ('',                                                  0.4,  (255, 255, 255)),
        ('2. AI Danger Detection  (body sway)',                0.52, (200, 200, 200)),
        ('   Faces anonymized (mosaic) before any transfer',  0.44, (150, 150, 150)),
        ('   Anonymized images sent to OpenAI API',           0.44, (150, 150, 150)),
        ('   Images deleted immediately after analysis',      0.44, (150, 150, 150)),
        ('',                                                  0.4,  (255, 255, 255)),
        ('Third-party provider : OpenAI',                     0.5,  (100, 180, 255)),
        ('',                                                  0.4,  (255, 255, 255)),
        ('Press  ENTER  to start      Q  to quit',            0.62, (255, 255, 255)),
    ]
    y = 30
    for text, scale, color in lines:
        if text:
            put_text_centered(notice, text, y, font, scale, color, 1)
        y += 32

    cv2.imshow('Face Detection', notice)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13:        # Enter
            return True
        elif key == ord('q'):
            return False


# ═══════════════════════════════════════════════════════════════
#  얼굴 모자이크 유틸
# ═══════════════════════════════════════════════════════════════

def apply_face_mosaic(frame, mosaic_scale=0.05):
    """프레임의 모든 얼굴 영역에 픽셀 모자이크를 적용한 복사본 반환"""
    detected = []
    if face_detector_image is not None:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=np.ascontiguousarray(rgb))
        res = face_detector_image.detect(mp_img)   # IMAGE 모드: detect()
        if res.detections:
            for det in res.detections:
                bb = det.bounding_box
                detected.append((bb.origin_x, bb.origin_y, bb.width, bb.height))
    result = frame.copy()
    for (x, y, w, h) in detected:
        roi = result[y:y + h, x:x + w]
        small_w = max(1, int(w * mosaic_scale))
        small_h = max(1, int(h * mosaic_scale))
        small   = cv2.resize(roi, (small_w, small_h))
        mosaic  = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        result[y:y + h, x:x + w] = mosaic
    return result


def save_danger_screenshots(frames):
    """
    위험 감지 프레임들을 얼굴 모자이크 처리 후 JPEG로 저장.
    저장된 파일 경로 목록을 반환한다.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_paths = []
    for i, frame in enumerate(frames):
        mosaiced = apply_face_mosaic(frame)
        path = os.path.join(DANGER_SCREENSHOT_DIR, f'danger_{timestamp}_{i + 1}.jpg')
        cv2.imwrite(path, mosaiced)
        saved_paths.append(path)
        print(f"[DANGER] 저장: {path}")
    return saved_paths


# ═══════════════════════════════════════════════════════════════
#  위험 행동 감지 — 몸 흔들림 (BodySwayDetector)
# ═══════════════════════════════════════════════════════════════

class BodySwayDetector:
    """
    MediaPipe Pose 결과에서 어깨 중심점을 추적하여
    과도한 몸 흔들림(excessive body sway)을 감지한다.

    동작 흐름:
      1) 매 프레임 어깨 중심 위치를 슬라이딩 윈도우에 누적
      2) 윈도우 내 X/Y 범위가 임계값 초과 → is_swaying() = True
      3) SWAY_TRIGGER_FRAMES 연속으로 흔들림 감지 → 캡처 시작
      4) SWAY_CAPTURE_COUNT 장 수집 후 'ready' 반환
    """

    def __init__(self):
        self.position_history: list[tuple] = []
        self.sway_consecutive  = 0
        self.capturing         = False
        self.capture_frames: list = []
        self.capture_countdown = 0
        self.last_api_call_time = 0.0

    # ── 내부 헬퍼 ─────────────────────────────────────────────

    def _shoulder_center(self, pose_results, frame_shape) -> tuple | None:
        """양쪽 어깨 중심 픽셀 좌표 반환. 신뢰도 낮으면 None."""
        if not pose_results.pose_landmarks:
            return None
        lm    = pose_results.pose_landmarks[0]
        h, w  = frame_shape[:2]
        left  = lm[_POSE_LEFT_SHOULDER]
        right = lm[_POSE_RIGHT_SHOULDER]
        if left.visibility < 0.5 or right.visibility < 0.5:
            return None
        cx = int((left.x + right.x) * w / 2)
        cy = int((left.y + right.y) * h / 2)
        return (cx, cy)

    def _is_swaying(self) -> bool:
        """슬라이딩 윈도우 내 위치 범위가 임계값을 초과하는지 확인"""
        if len(self.position_history) < SWAY_WINDOW_FRAMES:
            return False
        xs = [p[0] for p in self.position_history]
        ys = [p[1] for p in self.position_history]
        return (max(xs) - min(xs) > SWAY_X_THRESHOLD or
                max(ys) - min(ys) > SWAY_Y_THRESHOLD)

    # ── 외부 인터페이스 ────────────────────────────────────────

    def get_sway_range(self) -> tuple[int, int]:
        """현재 윈도우의 (X범위, Y범위) 픽셀값 반환"""
        if len(self.position_history) < 2:
            return 0, 0
        xs = [p[0] for p in self.position_history]
        ys = [p[1] for p in self.position_history]
        return max(xs) - min(xs), max(ys) - min(ys)

    def update(self, pose_results, frame, frame_shape) -> str | None:
        """
        매 프레임 호출.

        Returns
        -------
        'trigger'   : 흔들림 감지, 캡처 시작
        'capturing' : 캡처 진행 중
        'ready'     : SWAY_CAPTURE_COUNT 장 캡처 완료
        None        : 이상 없음
        """
        center = self._shoulder_center(pose_results, frame_shape)

        if center is not None:
            self.position_history.append(center)
            if len(self.position_history) > SWAY_WINDOW_FRAMES:
                self.position_history.pop(0)
        else:
            # 포즈 미감지 → 히스토리·연속 카운터 초기화
            self.position_history.clear()
            self.sway_consecutive = 0

        # ── 캡처 진행 중 ──────────────────────────────────────
        if self.capturing:
            self.capture_countdown -= 1
            if self.capture_countdown <= 0:
                self.capture_frames.append(frame.copy())
                self.capture_countdown = SWAY_CAPTURE_INTERVAL
                if len(self.capture_frames) >= SWAY_CAPTURE_COUNT:
                    self.capturing = False
                    self.sway_consecutive = 0
                    return 'ready'
            return 'capturing'

        # ── 흔들림 판단 ───────────────────────────────────────
        if self._is_swaying():
            self.sway_consecutive += 1
        else:
            self.sway_consecutive = max(0, self.sway_consecutive - 1)

        # ── 트리거 조건 ───────────────────────────────────────
        now = time.time()
        if (self.sway_consecutive >= SWAY_TRIGGER_FRAMES
                and now - self.last_api_call_time > OPENAI_API_COOLDOWN_SEC):
            self.capturing = True
            self.capture_frames = [frame.copy()]   # 현재 프레임 포함
            self.capture_countdown = SWAY_CAPTURE_INTERVAL
            print(f"[DANGER] 몸 흔들림 감지! "
                  f"캡처 시작 (연속={self.sway_consecutive}프레임)")
            return 'trigger'

        return None

    def get_captured_frames(self) -> list:
        return list(self.capture_frames)

    def reset_capture(self):
        """캡처 버퍼 초기화 및 API 쿨다운 시작"""
        self.capture_frames.clear()
        self.last_api_call_time = time.time()


def analyze_danger_with_ai_api(image_paths: list[str], result_callback):
    """
    얼굴 모자이크 처리된 이미지들을 OpenAI API에 전송해 위험 여부를 판단한다.
    백그라운드 스레드에서 실행되며, 완료 후 result_callback(result_text) 를 호출한다.

    환경 변수 OPENAI_API_KEY 가 설정되어 있어야 한다.
    """
    try:
        client  = OpenAI()   # OPENAI_API_KEY 자동 참조
        content = []

        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            _, buf  = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_b64 = base64.b64encode(buf).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                },
            })
            # 이미지를 메모리에 올린 즉시 파일 삭제
            try:
                os.remove(path)
                print(f"[DANGER] 삭제 완료: {path}")
            except OSError as e:
                print(f"[DANGER] 삭제 실패: {path} ({e})")

        if not content:
            result_callback("Failed to load images.")
            return

        content.append({
            "type": "text",
            "text": (
                "The following images are consecutive frames from an indoor CCTV where excessive body sway was detected. "
                "Faces have been intentionally blurred to protect personal privacy. "
                "Face identification is not required — please analyze only body posture, movement, and situation.\n\n"
                "다음 항목에 간결하게 한국어로 답해 주세요.\n"
                "1. 위험한 상황인가요? (예 / 아니오)\n"
                "2. 관찰된 위험 행동이나 신체 상태는 무엇인가요? (예: 비틀거림, 쓰러짐, 발작 등)\n"
                "3. 즉각적인 도움이 필요한가요?\n"
                "4. 어떤 상황인지 설명하세요.\n"
                "답변 할 때는 질문도 포함해서 대답하세요."
            ),
        })

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=512,
            messages=[{"role": "user", "content": content}],
        )

        result_callback(response.choices[0].message.content)

    except Exception as exc:
        result_callback(f"[오류] OpenAI API 호출 실패: {exc}")


# ═══════════════════════════════════════════════════════════════
#  초기화
# ═══════════════════════════════════════════════════════════════

# ── MediaPipe FaceDetector (Tasks API) ────────────────────────
# blaze_face_short_range.tflite 다운로드:
# https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
face_detector_video = None   # 메인 루프용 (VIDEO 모드)
face_detector_image = None   # 모자이크 저장용 (IMAGE 모드)
if os.path.exists(FACE_MODEL_PATH):
    face_detector_video = mp_vision.FaceDetector.create_from_options(
        mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_detection_confidence=0.5,
        )
    )
    face_detector_image = mp_vision.FaceDetector.create_from_options(
        mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
    )
    print("[FACE] MediaPipe FaceDetector 로드 완료")
else:
    print(f"[FACE] {FACE_MODEL_PATH} 없음 → 얼굴 검출 비활성화")
    print("  다운로드 후 스크립트와 같은 폴더에 저장하세요.")


def _detect_faces_mp(mp_img, timestamp_ms) -> list[tuple]:
    """
    MediaPipe FaceDetector (VIDEO 모드)로 얼굴 검출.
    반환: [(x, y, w, h), ...] — 픽셀 좌표 (Haar Cascade와 동일 형식)
    """
    if face_detector_video is None:
        return []
    results = face_detector_video.detect_for_video(mp_img, timestamp_ms)
    out = []
    if results.detections:
        for det in results.detections:
            bb = det.bounding_box   # 픽셀 좌표 (상대값 아님)
            out.append((bb.origin_x, bb.origin_y, bb.width, bb.height))
    return out

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

# 얼굴 인식 상태
last_face_region     = None
last_known_customer  = None
occlusion_frame_count = 0
occlusion_active     = False
occlusion_start_time = None
person_gone_count    = 0
last_cleanup_hour    = datetime.now().hour

# 위험 행동 감지 상태
sway_detector              = BodySwayDetector()
danger_analysis_result: str | None = None
danger_analysis_pending    = False

_loop_start_ms = int(time.time() * 1000)


def on_danger_analysis_complete(result: str):
    global danger_analysis_result, danger_analysis_pending
    print("\n" + "=" * 52)
    print("  ⚠   OpenAI 위험 상황 분석 결과  ⚠")
    print("=" * 52)
    print(result)
    print("=" * 52 + "\n")
    danger_analysis_result  = result
    danger_analysis_pending = False


# ═══════════════════════════════════════════════════════════════
#  운영자 고지 — 동의하지 않으면 종료
# ═══════════════════════════════════════════════════════════════

if not show_operator_notice():
    cap.release()
    cv2.destroyAllWindows()
    exit()

# ═══════════════════════════════════════════════════════════════
#  메인 루프
# ═══════════════════════════════════════════════════════════════

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 시간 기반 타임스탬프 (WebCam에서 CAP_PROP_POS_MSEC이 0을 반환하는 경우 대비)
    timestamp_ms = int(time.time() * 1000) - _loop_start_ms

    current_hour = datetime.now().hour
    if current_hour != last_cleanup_hour:
        cleanup_old_faces()
        recognizer, label_map = load_training_data()
        last_cleanup_hour = current_hour

    display  = frame.copy()
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    faces        = _detect_faces_mp(mp_image, timestamp_ms)
    hand_results = hands_detector.detect_for_video(mp_image, timestamp_ms)

    # ── Pose 감지 ─────────────────────────────────────────────────
    pose_results = None
    if pose_detector is not None:
        pose_results = pose_detector.detect_for_video(mp_image, timestamp_ms)

    if not waiting_consent:
        current_customer = None

        # ── 얼굴 인식 ─────────────────────────────────────────────
        for (x, y, w, h) in faces:
            face_gray = cv2.resize(gray[y:y + h, x:x + w], (100, 100))
            if recognizer and label_map:
                label_id, confidence = recognizer.predict(face_gray)
                if confidence < CONFIDENCE_THRESHOLD:
                    current_customer = label_map.get(label_id, 'Unknown')
                    color = (0, 255, 0)
                    text  = f'{current_customer} ({int(confidence)})'
                else:
                    color = (0, 100, 255)
                    text  = 'Unknown'
            else:
                color = (0, 255, 0)
                text  = 'Face'
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ── 손 랜드마크 표시 ──────────────────────────────────────
        if hand_results.hand_landmarks:
            h_img, w_img = frame.shape[:2]
            for hand_lms in hand_results.hand_landmarks:
                for lm in hand_lms:
                    cx, cy = int(lm.x * w_img), int(lm.y * h_img)
                    cv2.circle(display, (cx, cy), 4, (0, 215, 255), -1)

        # ── 고객 등장/퇴장/가림 이벤트 ───────────────────────────
        if current_customer and current_customer != last_known_customer:
            log_event(event='face_appeared', customer=current_customer,
                      timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            last_known_customer = current_customer

        if current_customer and len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            last_face_region = (fx, fy, fw, fh)

        known_customer_missing = last_known_customer is not None and current_customer is None

        if known_customer_missing and last_face_region is not None:
            hand_detected = hand_near_face(hand_results, last_face_region, frame.shape)
            if hand_detected:
                occlusion_frame_count += 1
                person_gone_count = 0
                if occlusion_frame_count >= OCCLUSION_TRIGGER_FRAMES and not occlusion_active:
                    occlusion_active     = True
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
                    last_known_customer   = None
                    last_face_region      = None
                    occlusion_frame_count = 0
                    person_gone_count     = 0

        elif current_customer is not None:
            if occlusion_active:
                duration = round((datetime.now() - occlusion_start_time).total_seconds(), 1)
                log_event(event='face_occlusion_end',
                          customer=last_known_customer,
                          timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                          duration_sec=duration)
            occlusion_frame_count = 0
            occlusion_active      = False
            person_gone_count     = 0

        if occlusion_active:
            cv2.putText(display, '! Occlusion Detected', (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

        # ── 위험 행동 감지 (몸 흔들림) ────────────────────────────
        if pose_results is not None:
            sway_status = sway_detector.update(pose_results, frame, frame.shape)

            # 캡처 완료 → 저장 + openai API 비동기 호출
            if sway_status == 'ready' and not danger_analysis_pending:
                captured    = sway_detector.get_captured_frames()
                sway_detector.reset_capture()
                saved_paths = save_danger_screenshots(captured)
                if saved_paths:
                    danger_analysis_pending = True
                    t = threading.Thread(
                        target=analyze_danger_with_ai_api,
                        args=(saved_paths, on_danger_analysis_complete),
                        daemon=True,
                    )
                    t.start()
                    print("[DANGER] OpenAI API 분석 요청 중...")

            # 어깨 중심점 시각화
            if pose_results.pose_landmarks:
                lm            = pose_results.pose_landmarks[0]
                h_img, w_img  = frame.shape[:2]
                ls            = lm[_POSE_LEFT_SHOULDER]
                rs            = lm[_POSE_RIGHT_SHOULDER]
                if ls.visibility >= 0.5 and rs.visibility >= 0.5:
                    cx = int((ls.x + rs.x) * w_img / 2)
                    cy = int((ls.y + rs.y) * h_img / 2)
                    cv2.circle(display, (cx, cy), 8, (255, 100, 0), -1)

            # 흔들림 수치 HUD
            x_range, y_range = sway_detector.get_sway_range()
            sway_color = (
                (0, 0, 255) if sway_detector.sway_consecutive >= SWAY_TRIGGER_FRAMES
                else (0, 200, 200)
            )
            cv2.putText(display,
                        f'Sway X:{x_range} Y:{y_range}  [{sway_detector.sway_consecutive}f]',
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sway_color, 1)

            # 캡처 중 표시
            if sway_detector.capturing:
                n_captured = len(sway_detector.capture_frames)
                cv2.putText(display,
                            f'! DANGER CAPTURING  {n_captured}/{SWAY_CAPTURE_COUNT}',
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


        # ── 기본 HUD ──────────────────────────────────────────────
        cv2.putText(display, f'Faces: {len(faces)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display, "'s': save  'r': retrain  'q': exit",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Face Detection', display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            consent_face_img      = frame[y:y + h, x:x + w].copy()
            consent_frame_display = display.copy()
            waiting_consent       = True

        elif key == ord('r'):
            recognizer, label_map = load_training_data()
            print(f"재학습 완료: {list(label_map.values())}")

        elif key == ord('q'):
            break

    # ── 동의 화면 (기존 로직 유지) ──────────────────────────────
    else:
        overlay          = consent_frame_display.copy()
        h_frame, w_frame = overlay.shape[:2]
        dark             = overlay.copy()
        cv2.rectangle(dark, (0, 0), (w_frame, h_frame), (0, 0, 0), -1)
        cv2.addWeighted(dark, 0.5, overlay, 0.5, 0, overlay)

        font  = cv2.FONT_HERSHEY_SIMPLEX
        lines = [
            ('[Personal Information Collection Notice]', 0.6, (0, 255, 255)),
            ('Collector: PA',                            0.5, (200, 200, 200)),
            ('Purpose: Customer service',                0.5, (200, 200, 200)),
            ('Items: Photo, Name, Order history',        0.5, (200, 200, 200)),
            ('Retention: 7 days, then deleted',          0.5, (200, 200, 200)),
            ('Third-party sharing: None',                0.5, (200, 200, 200)),
            ('Refusal: No disadvantage',                 0.5, (200, 200, 200)),
            ('Do you agree?  [y] Yes   [n] No',          0.65, (255, 255, 255)),
        ]
        start_y = h_frame // 2 - 130
        for i, (text, scale, color) in enumerate(lines):
            put_text_centered(overlay, text, start_y + i * 38, font, scale, color, 1)

        cv2.imshow('Face Detection', overlay)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('y'):
            waiting_consent = False
            customer_name   = ''
            typing          = True
            while typing:
                input_overlay = consent_frame_display.copy()
                dark          = input_overlay.copy()
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
                        ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = os.path.join(SAVE_DIR, f'face_{customer_name.strip()}_{ts}.jpg')
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
if pose_detector is not None:
    pose_detector.__exit__(None, None, None)
cv2.destroyAllWindows()