"""
face_detection.py의 occlusion_active 신호와 BodySwayDetector를
state_machine 큐에 연결하는 어댑터.

USB 웹캠(int) 또는 Tapo RTSP URL(str)을 지원한다.
"""
import os
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import asyncio
import base64
import threading
import time
from typing import Union

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from openai import OpenAI

from llm_module.state_machine import TriggerSignals

OPENCV_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(OPENCV_DIR, 'hand_landmarker.task')
POSE_MODEL_PATH  = os.path.join(OPENCV_DIR, 'pose_landmarker_lite.task')
FACE_MODEL_PATH  = os.path.join(OPENCV_DIR, 'blaze_face_short_range.tflite')
DANGER_SCREENSHOT_DIR = 'danger_screenshots'

FRAME_SIZE                 = (640, 360)
OCCLUSION_TRIGGER_FRAMES   = 10
PERSON_GONE_TRIGGER_FRAMES = 30
SAMPLE_INTERVAL_SEC        = 0.4

# 몸 흔들림 감지
SWAY_WINDOW_FRAMES      = 15
SWAY_X_THRESHOLD        = 30
SWAY_Y_THRESHOLD        = 25
SWAY_TRIGGER_FRAMES     = 5
SWAY_CAPTURE_COUNT      = 10
SWAY_CAPTURE_INTERVAL   = 3
OPENAI_API_COOLDOWN_SEC = 30
_POSE_LEFT_SHOULDER     = 11
_POSE_RIGHT_SHOULDER    = 12

os.makedirs(DANGER_SCREENSHOT_DIR, exist_ok=True)

CameraSource = Union[int, str]


# ═══════════════════════════════════════════════════════════════
#  몸 흔들림 감지
# ═══════════════════════════════════════════════════════════════

class BodySwayDetector:
    """어깨 중심점을 추적해 과도한 몸 흔들림을 감지한다."""

    def __init__(self):
        self.position_history: list[tuple] = []
        self.sway_consecutive   = 0
        self.capturing          = False
        self.capture_frames: list = []
        self.capture_countdown  = 0
        self.last_api_call_time = 0.0

    def _shoulder_center(self, pose_results, frame_shape) -> tuple | None:
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
        if len(self.position_history) < SWAY_WINDOW_FRAMES:
            return False
        xs = [p[0] for p in self.position_history]
        ys = [p[1] for p in self.position_history]
        return (max(xs) - min(xs) > SWAY_X_THRESHOLD or
                max(ys) - min(ys) > SWAY_Y_THRESHOLD)

    def get_sway_range(self) -> tuple[int, int]:
        if len(self.position_history) < 2:
            return 0, 0
        xs = [p[0] for p in self.position_history]
        ys = [p[1] for p in self.position_history]
        return max(xs) - min(xs), max(ys) - min(ys)

    def update(self, pose_results, frame, frame_shape) -> str | None:
        """
        Returns: 'trigger' | 'capturing' | 'ready' | None
        """
        center = self._shoulder_center(pose_results, frame_shape)
        if center is not None:
            self.position_history.append(center)
            if len(self.position_history) > SWAY_WINDOW_FRAMES:
                self.position_history.pop(0)
        else:
            self.position_history.clear()
            self.sway_consecutive = 0

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

        if self._is_swaying():
            self.sway_consecutive += 1
        else:
            self.sway_consecutive = max(0, self.sway_consecutive - 1)

        now = time.time()
        if (self.sway_consecutive >= SWAY_TRIGGER_FRAMES
                and now - self.last_api_call_time > OPENAI_API_COOLDOWN_SEC):
            self.capturing = True
            self.capture_frames = [frame.copy()]
            self.capture_countdown = SWAY_CAPTURE_INTERVAL
            print(f"[DANGER] 몸 흔들림 감지! 캡처 시작 (연속={self.sway_consecutive}프레임)")
            return 'trigger'

        return None

    def get_captured_frames(self) -> list:
        return list(self.capture_frames)

    def reset_capture(self):
        self.capture_frames.clear()
        self.last_api_call_time = time.time()


# ═══════════════════════════════════════════════════════════════
#  얼굴 모자이크 + 위험 스크린샷 저장
# ═══════════════════════════════════════════════════════════════

def apply_face_mosaic(frame, face_detector, mosaic_scale=0.05):
    """프레임의 모든 얼굴 영역에 픽셀 모자이크를 적용한 복사본 반환."""
    if face_detector is None:
        return frame.copy()
    fh_img, fw_img = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res    = face_detector.detect(mp_img)
    result = frame.copy()
    if res.detections:
        for det in res.detections:
            bb = det.bounding_box
            x1 = max(0, bb.origin_x)
            y1 = max(0, bb.origin_y)
            x2 = min(fw_img, bb.origin_x + bb.width)
            y2 = min(fh_img, bb.origin_y + bb.height)
            if x2 <= x1 or y2 <= y1:
                continue
            roi     = result[y1:y2, x1:x2]
            rh, rw  = roi.shape[:2]
            small_w = max(1, int(rw * mosaic_scale))
            small_h = max(1, int(rh * mosaic_scale))
            small   = cv2.resize(roi, (small_w, small_h))
            mosaic  = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
            result[y1:y2, x1:x2] = mosaic
    return result


def save_danger_screenshots(frames, face_detector) -> list[str]:
    """위험 감지 프레임들을 얼굴 모자이크 처리 후 JPEG로 저장. 경로 목록 반환."""
    from datetime import datetime
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_paths = []
    for i, frame in enumerate(frames):
        mosaiced = apply_face_mosaic(frame, face_detector)
        path     = os.path.join(DANGER_SCREENSHOT_DIR, f'danger_{timestamp}_{i + 1}.jpg')
        cv2.imwrite(path, mosaiced)
        saved_paths.append(path)
        print(f"[DANGER] 저장: {path}")
    return saved_paths


def analyze_danger_with_ai_api(image_paths: list[str], result_callback):
    """모자이크 처리된 이미지를 GPT-4o로 분석. 백그라운드 스레드에서 실행."""
    # 파일 읽기 + 즉시 삭제 (API 성공/실패와 무관하게 삭제 보장)
    content = []
    for path in image_paths:
        img = cv2.imread(path)
        try:
            os.remove(path)
        except OSError:
            pass
        if img is None:
            continue
        _, buf  = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf).decode('utf-8')
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    if not content:
        result_callback("Failed to load images.")
        return

    try:
        client = OpenAI()

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
#  카메라 유틸
# ═══════════════════════════════════════════════════════════════

def _open_capture(camera_source: CameraSource) -> cv2.VideoCapture:
    if isinstance(camera_source, str):
        cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(camera_source)
    return cap


def _camera_label(camera_source: CameraSource) -> str:
    if isinstance(camera_source, str):
        return "RTSP stream"
    return f"USB camera ({camera_source})"


def _hand_near_face(hand_results, face_region, frame_shape, margin_ratio=0.6) -> bool:
    """얼굴 영역 근처에 손 랜드마크가 있는지 확인."""
    if not hand_results.hand_landmarks or face_region is None:
        return False
    h_img, w_img = frame_shape[:2]
    fx, fy, fw, fh = face_region
    margin = int(max(fw, fh) * margin_ratio)
    rx1, ry1 = max(0, fx - margin), max(0, fy - margin)
    rx2, ry2 = min(w_img, fx + fw + margin), min(h_img, fy + fh + margin)
    for hand_lms in hand_results.hand_landmarks:
        for lm in hand_lms:
            hx, hy = int(lm.x * w_img), int(lm.y * h_img)
            if rx1 <= hx <= rx2 and ry1 <= hy <= ry2:
                return True
    return False


# ═══════════════════════════════════════════════════════════════
#  메인 감지 루프
# ═══════════════════════════════════════════════════════════════

def _run_detection_loop(
    zone_id: str,
    frame_queue: asyncio.Queue,
    signal_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    camera_source: CameraSource = 0,
) -> None:
    """동기 OpenCV 루프. 별도 스레드에서 실행된다."""
    if not os.path.exists(MODEL_PATH):
        print(f"[{zone_id}][BRIDGE] MediaPipe model not found: {MODEL_PATH}")
        print(f"[{zone_id}][BRIDGE] Download hand_landmarker.task to the opencv/ directory.")
        return

    # HandLandmarker
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    hands_detector = mp_vision.HandLandmarker.create_from_options(hand_options)

    # PoseLandmarker (VIDEO 모드 — 흔들림 감지)
    pose_detector = None
    if os.path.exists(POSE_MODEL_PATH):
        pose_detector = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
    else:
        print(f"[{zone_id}][BRIDGE] {POSE_MODEL_PATH} 없음 → 몸 흔들림 감지 비활성화")

    # FaceDetector VIDEO 모드 — 실시간 얼굴 감지 (face_detection.py와 동일 엔진)
    face_detector_video = None
    # FaceDetector IMAGE 모드 — 모자이크 저장 전용
    face_detector_image = None
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
    else:
        print(f"[{zone_id}][BRIDGE] {FACE_MODEL_PATH} 없음 → Haar Cascade 폴백, 모자이크 비활성화")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = _open_capture(camera_source)
    if not cap.isOpened():
        print(f"[{zone_id}][BRIDGE] Failed to open {_camera_label(camera_source)}")
        if isinstance(camera_source, str):
            print(f"[{zone_id}][BRIDGE] Check TAPO_RTSP_URL in .env (username, password, IP, stream2).")
        return

    print(f"[{zone_id}][BRIDGE] Camera started ({_camera_label(camera_source)})")

    last_face_region      = None
    occlusion_frame_count = 0
    occlusion_active      = False
    person_gone_count     = 0

    frame_buffer       = []
    last_sample_time   = 0.0
    last_data_log_time = 0.0
    _loop_start_ms     = int(time.time() * 1000)

    sway_detector  = BodySwayDetector()
    danger_state   = {'pending': False}

    def _on_danger_complete(result: str):
        print(f"\n{'=' * 52}")
        print(f"  [{zone_id}] 위험 상황 분석 결과")
        print('=' * 52)
        print(result)
        print('=' * 52 + '\n')
        danger_state['pending'] = False

    debug      = os.getenv("DEBUG_CAMERA", "1") == "1"
    trace_data = os.getenv("TRACE_CAMERA_DATA", "1") == "1"
    use_rtsp   = isinstance(camera_source, str)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[{zone_id}][BRIDGE] Failed to read frame, retrying...")
                time.sleep(0.1)
                continue

            if use_rtsp or frame.shape[1] != FRAME_SIZE[0] or frame.shape[0] != FRAME_SIZE[1]:
                frame = cv2.resize(frame, FRAME_SIZE)

            now     = time.time()
            display = frame.copy()
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
            timestamp_ms = int(time.time() * 1000) - _loop_start_ms

            if face_detector_video is not None:
                fd_res = face_detector_video.detect_for_video(mp_image, timestamp_ms)
                faces  = [(d.bounding_box.origin_x, d.bounding_box.origin_y,
                           d.bounding_box.width, d.bounding_box.height)
                          for d in fd_res.detections] if fd_res.detections else []
            else:
                faces = list(face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)))

            hand_results = hands_detector.detect_for_video(mp_image, timestamp_ms)

            # Pose
            pose_results = None
            if pose_detector is not None:
                pose_results = pose_detector.detect_for_video(mp_image, timestamp_ms)

            # 얼굴/손 가림 감지
            # bridge.py는 LBPH 인식기가 없으므로 "얼굴이 감지된 상태에서 손이 근처에 있으면" occlusion으로 판정
            if len(faces) > 0:
                fx, fy, fw, fh   = max(faces, key=lambda f: f[2] * f[3])
                last_face_region = (fx, fy, fw, fh)
                person_gone_count = 0

                hand_detected = _hand_near_face(hand_results, last_face_region, frame.shape)
                if hand_detected:
                    occlusion_frame_count += 1
                    if occlusion_frame_count >= OCCLUSION_TRIGGER_FRAMES and not occlusion_active:
                        occlusion_active = True
                else:
                    occlusion_frame_count = max(0, occlusion_frame_count - 1)
                    if occlusion_active and occlusion_frame_count == 0:
                        occlusion_active = False

            elif last_face_region is not None:
                # 얼굴이 사라진 경우 — 일정 시간 후 상태 초기화
                person_gone_count += 1
                if person_gone_count >= PERSON_GONE_TRIGGER_FRAMES:
                    occlusion_active      = False
                    last_face_region      = None
                    occlusion_frame_count = 0
                    person_gone_count     = 0

            # 몸 흔들림 감지
            body_sway_signal = False
            if pose_results is not None:
                try:
                    sway_status = sway_detector.update(pose_results, frame, frame.shape)

                    if sway_status == 'ready' and not danger_state['pending']:
                        captured    = sway_detector.get_captured_frames()
                        sway_detector.reset_capture()
                        saved_paths = save_danger_screenshots(captured, face_detector_image)
                        if saved_paths:
                            danger_state['pending'] = True
                            threading.Thread(
                                target=analyze_danger_with_ai_api,
                                args=(saved_paths, _on_danger_complete),
                                daemon=True,
                            ).start()
                            print(f"[{zone_id}][DANGER] OpenAI API 분석 요청 중...")

                    body_sway_signal = sway_detector.sway_consecutive >= SWAY_TRIGGER_FRAMES
                except Exception as e:
                    print(f"[{zone_id}][DANGER] 처리 오류 (카메라 유지): {e}")

            # debug 시각화
            if debug:
                h_img, w_img = frame.shape[:2]
                for (x, y, w, h) in faces:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if hand_results.hand_landmarks:
                    for hand_lms in hand_results.hand_landmarks:
                        for lm in hand_lms:
                            cx, cy = int(lm.x * w_img), int(lm.y * h_img)
                            cv2.circle(display, (cx, cy), 4, (0, 215, 255), -1)
                if pose_results is not None and pose_results.pose_landmarks:
                    lm = pose_results.pose_landmarks[0]
                    ls = lm[_POSE_LEFT_SHOULDER]
                    rs = lm[_POSE_RIGHT_SHOULDER]
                    if ls.visibility >= 0.5 and rs.visibility >= 0.5:
                        cx = int((ls.x + rs.x) * w_img / 2)
                        cy = int((ls.y + rs.y) * h_img / 2)
                        cv2.circle(display, (cx, cy), 8, (255, 100, 0), -1)

                alert = occlusion_active or body_sway_signal
                status = ("SWEAT!" if occlusion_active else "") + (" SWAY!" if body_sway_signal else "") or "monitoring..."
                cv2.putText(display, status.strip(), (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if alert else (0, 255, 0), 2)
                cv2.putText(display, f"occlusion: {occlusion_frame_count}/{OCCLUSION_TRIGGER_FRAMES}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display, f"faces: {len(faces)}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                if pose_results is not None:
                    x_range, y_range = sway_detector.get_sway_range()
                    cv2.putText(display,
                                f"Sway X:{x_range} Y:{y_range} [{sway_detector.sway_consecutive}f]",
                                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 1)
                if sway_detector.capturing:
                    n = len(sway_detector.capture_frames)
                    cv2.putText(display, f"! CAPTURING {n}/{SWAY_CAPTURE_COUNT}",
                                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow(f"Bridge - {zone_id}", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 프레임 버퍼 (VLM용)
            if now - last_sample_time >= SAMPLE_INTERVAL_SEC:
                frame_buffer.append(frame.copy())
                if len(frame_buffer) > 4:
                    frame_buffer.pop(0)
                last_sample_time = now

            signals = TriggerSignals(
                sweat_wiping=occlusion_active,
                body_sway=body_sway_signal,
                person_count=len(faces),
            )

            if trace_data and now - last_data_log_time >= 1.0:
                hand_count = len(hand_results.hand_landmarks or [])
                print(
                    f"[{zone_id}][DATA][BRIDGE->STATE] "
                    f"faces={len(faces)} hands={hand_count} "
                    f"occlusion_count={occlusion_frame_count}/{OCCLUSION_TRIGGER_FRAMES} "
                    f"sweat_wiping={signals.sweat_wiping} "
                    f"body_sway={signals.body_sway} "
                    f"person_count={signals.person_count} "
                    f"frame_buffer={len(frame_buffer)}/4"
                )
                last_data_log_time = now

            asyncio.run_coroutine_threadsafe(
                _put_nowait(signal_queue, signals), loop
            )

            # 이상 감지 시 프레임 배치를 VLM 파이프라인으로 전송 (얼굴 모자이크 처리 후)
            if len(frame_buffer) == 4 and (occlusion_active or body_sway_signal):
                if trace_data:
                    print(
                        f"[{zone_id}][DATA][BRIDGE->STATE] "
                        "sending 4-frame batch for VLM confirmation"
                    )
                mosaiced_batch = [apply_face_mosaic(f, face_detector_image) for f in frame_buffer]
                asyncio.run_coroutine_threadsafe(
                    _put_nowait(frame_queue, mosaiced_batch), loop
                )

    finally:
        cap.release()
        hands_detector.__exit__(None, None, None)
        if pose_detector is not None:
            pose_detector.__exit__(None, None, None)
        if face_detector_video is not None:
            face_detector_video.__exit__(None, None, None)
        if face_detector_image is not None:
            face_detector_image.__exit__(None, None, None)
        if debug:
            cv2.destroyAllWindows()
        print(f"[{zone_id}][BRIDGE] Camera stopped")


async def _put_nowait(queue: asyncio.Queue, item) -> None:
    """큐가 꽉 찼으면 오래된 것을 버리고 최신 값으로 교체."""
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    await queue.put(item)


async def run(
    zone_id: str,
    frame_queue: asyncio.Queue,
    signal_queue: asyncio.Queue,
    camera_source: CameraSource = 0,
) -> None:
    """main.py에서 asyncio.create_task()로 호출하는 진입점."""
    loop = asyncio.get_running_loop()
    await asyncio.to_thread(
        _run_detection_loop,
        zone_id, frame_queue, signal_queue, loop, camera_source,
    )
