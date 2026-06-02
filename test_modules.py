"""
카메라 없이 각 모듈을 독립적으로 테스트하는 스크립트.
실행: python test_modules.py
"""
import asyncio
from dotenv import load_dotenv
load_dotenv()


async def test_customer_bot():
    print("\n" + "="*50)
    print("[ 고객 응대 챗봇 테스트 ]")
    print("="*50)
    from llm_module.customer_bot import respond, closing_notice, extension_offer

    context = {
        "customer_name": "김철수",
        "visit_count": 3,
        "current_temp": 26.0,
        "remaining_min": 10,
        "reserved_min": 60,
    }

    tests = [
        "좀 더 춥게 해줘",
        "남은 시간 알려줘",
        "시간 연장하고 싶어",
        "너무 더워요",
    ]

    for msg in tests:
        print(f"\n고객: {msg}")
        result = await respond(msg, "1번 구역", context, tts=False)
        print(f"봇:   {result['message']}")
        if result["action"]:
            print(f"액션: {result['action']}")


async def test_customer_bot_mic():
    """
    마이크 인터랙티브 모드.
    Enter를 누르면 N초간 녹음 → Whisper로 전사 → customer_bot에 전달 → TTS 응답 재생.
    Ctrl+C로 종료.
    """
    print("\n" + "="*50)
    print("[ 고객 응대 챗봇 — 마이크 입력 모드 ]")
    print("="*50)
    print("준비물: 마이크. (sounddevice/soundfile 미설치 시 pip install 필요)")

    try:
        from llm_module.stt import record_and_transcribe, play_audio, list_input_devices
    except ImportError as e:
        print(f"[ERROR] STT 모듈 import 실패: {e}")
        print("→ pip install sounddevice soundfile")
        return

    from llm_module.customer_bot import respond

    zone_id = input("구역 ID (기본 '1번 구역'): ").strip() or "1번 구역"
    try:
        seconds = float(input("녹음 길이 초 (기본 5): ").strip() or "5")
    except ValueError:
        seconds = 5.0
    show_devices = input("마이크 장치 목록을 볼까요? (y/N): ").strip().lower() == "y"
    if show_devices:
        list_input_devices()
    device_text = input("사용할 입력 장치 ID/이름 (기본 장치 사용 시 Enter): ").strip()
    device = device_text if device_text else None
    if device_text.isdigit():
        device = int(device_text)

    context = {
        "customer_name": "테스터",
        "visit_count": 1,
        "current_temp": 25.0,
        "remaining_min": 30,
        "reserved_min": 60,
    }

    print(f"\n[준비] zone={zone_id} · 녹음 {seconds:.0f}초. Ctrl+C로 종료.\n")

    while True:
        try:
            input("→ Enter를 눌러 녹음 시작...")
        except (EOFError, KeyboardInterrupt):
            print("\n[종료]")
            return

        try:
            text = await record_and_transcribe(seconds=seconds, device=device)
        except KeyboardInterrupt:
            print("\n[종료]")
            return
        except Exception as e:
            print(f"[STT ERROR] {e}")
            continue

        if not text:
            print("[STT] 인식 결과 없음 (다시 시도)")
            continue

        print(f"[STT] 인식: {text}")
        result = await respond(text, zone_id, context, tts=True, all_zone_ids=["1번 구역","2번 구역","3번 구역"])

        print(f"\n봇: {result.get('message','')}")
        if result.get("action"):
            print(f"액션: {result['action']}")
        audio_path = result.get("audio_path")
        if audio_path:
            print(f"[TTS] 재생: {audio_path}")
            play_audio(audio_path)
        print()


async def test_stt_mic_only():
    """마이크 → Whisper STT 단독 테스트."""
    print("\n" + "="*50)
    print("[ STT 마이크 단독 테스트 ]")
    print("="*50)
    print("목적: 마이크 입력이 stt.py를 통해 텍스트로 변환되는지만 확인")

    try:
        from llm_module.stt import record_and_transcribe, list_input_devices
    except ImportError as e:
        print(f"[ERROR] STT 모듈 import 실패: {e}")
        print("→ pip install sounddevice soundfile")
        return

    try:
        seconds = float(input("녹음 길이 초 (기본 5): ").strip() or "5")
    except ValueError:
        seconds = 5.0

    if input("마이크 장치 목록을 볼까요? (y/N): ").strip().lower() == "y":
        list_input_devices()
    device_text = input("사용할 입력 장치 ID/이름 (기본 장치 사용 시 Enter): ").strip()
    device = device_text if device_text else None
    if device_text.isdigit():
        device = int(device_text)

    while True:
        try:
            input("→ Enter를 눌러 녹음 시작 (Ctrl+C 종료)...")
            text = await record_and_transcribe(seconds=seconds, device=device)
        except (EOFError, KeyboardInterrupt):
            print("\n[종료]")
            return
        except Exception as e:
            print(f"[STT ERROR] {e}")
            continue

        print(f"[STT RESULT] {text or '(인식 결과 없음)'}")


async def test_stt_video_file():
    """저장된 영상/오디오 파일 → Whisper STT 테스트."""
    print("\n" + "="*50)
    print("[ STT 영상/오디오 파일 테스트 ]")
    print("="*50)
    print("영상 파일에서 음성을 가져오는 경우 ffmpeg가 필요합니다.")

    try:
        from llm_module.stt import transcribe_audio_file, transcribe_video_file
    except ImportError as e:
        print(f"[ERROR] STT 모듈 import 실패: {e}")
        return

    path = input("영상/오디오 파일 경로 입력: ").strip().strip('"')
    if not path:
        print("[STT] 파일 경로가 비어 있습니다.")
        return

    lower = path.lower()
    try:
        if lower.endswith((".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg")):
            text = await transcribe_audio_file(path)
        else:
            text = await transcribe_video_file(path)
    except Exception as e:
        print(f"[STT ERROR] {e}")
        return

    print(f"[STT RESULT] {text or '(인식 결과 없음)'}")


async def test_report_generator():
    print("\n" + "="*50)
    print("[ 일일 리포트 테스트 ]")
    print("="*50)
    from llm_module.report_generator import generate_daily_report

    report = await generate_daily_report(["1번 구역", "2번 구역", "3번 구역"])
    print(report)


async def test_alert_manager():
    print("\n" + "="*50)
    print("[ 관리자 알림 테스트 (SMS/푸시 미설정 시 콘솔 출력) ]")
    print("="*50)
    from llm_module.vlm_analyzer import AnalysisResult, DetectionType, Severity
    from llm_module.alert_manager import handle

    # 도난 감지 시뮬레이션
    result = AnalysisResult(
        detection_type=DetectionType.THEFT,
        detected=True,
        confidence=0.88,
        severity=Severity.HIGH,
        evidence="Person moving merchandise toward exit",
        action_required="Notify manager immediately",
    )
    print("\n도난 감지 알림 테스트:")
    await handle(result, "2번 구역")

    # 땀 감지 시뮬레이션
    result2 = AnalysisResult(
        detection_type=DetectionType.SWEAT_WIPING,
        detected=True,
        confidence=0.91,
        severity=Severity.LOW,
        evidence="Hand touching forehead repeatedly",
        action_required="Lower temperature",
    )
    print("\n땀 감지 알림 테스트:")
    await handle(result2, "1번 구역")


async def test_multi_agent():
    """
    멀티에이전트 협력 시나리오 테스트.
    [AGENT: xxx] 로그로 어떤 에이전트가 실행됐는지 확인 가능.
    """
    print("\n" + "="*50)
    print("[ 멀티에이전트 시나리오 테스트 ]")
    print("="*50)
    from llm_module.graph import facility_graph
    from llm_module.state import make_customer_state, make_safety_state

    ZONE_IDS = ["1번 구역", "2번 구역", "3번 구역"]

    # ── 시나리오 1: 단일 구역 요청 ────────────────────────────────────────────
    print("\n[시나리오 1] 단일 구역 요청 (reconcile no-op 예상)")
    print("  기대 경로: orchestrator_dispatch → customer → orchestrator_reconcile → actuator → END")
    state = make_customer_state("1번 구역", ZONE_IDS, "온도 좀 낮춰줘", {"current_temp": 27.0}, tts_enabled=False)
    result = await facility_graph.ainvoke(state)
    print(f"  응답: {result.get('bot_response', {}).get('message', '')}")
    print(f"  오케스트레이터 결정: {result.get('orchestrator_decision', '(미실행)')}")

    # ── 시나리오 2: 전체 구역 요청 ────────────────────────────────────────────
    print("\n[시나리오 2] 전체 구역 요청 (reconcile LLM 판단 예상)")
    print("  기대 경로: orchestrator_dispatch → customer → orchestrator_reconcile → actuator → END")
    state2 = make_customer_state("1번 구역", ZONE_IDS, "전체 구역 온도 22도로 맞춰줘", {"current_temp": 27.0}, tts_enabled=False)
    result2 = await facility_graph.ainvoke(state2)
    print(f"  응답: {result2.get('bot_response', {}).get('message', '')}")
    print(f"  오케스트레이터 결정: {result2.get('orchestrator_decision', '(미실행)')}")

    # ── 시나리오 3: 고위험 불확실 안전 감지 ─────────────────────────────────────
    print("\n[시나리오 3] 고위험 불확실 감지 (reconcile 알림 발송 예상)")
    print("  기대 경로: orchestrator_dispatch → safety → orchestrator_reconcile → actuator → END")
    state3 = make_safety_state(
        zone_id="2번 구역",
        all_zone_ids=ZONE_IDS,
        analysis_result={
            "detection_type": "fall_emergency",
            "detected": True,
            "confidence": 0.72,   # 0.80 미만 → 오케스트레이터 위임
            "severity": "high",
            "evidence": "Person lying on floor, unclear if intentional",
            "action_required": "Verify with customer or manager",
        },
        signals={"temperature": 25.0, "humidity": 60.0},
    )
    result3 = await facility_graph.ainvoke(state3)
    print(f"  충돌 감지됨: {result3.get('conflict_detected', False)}")
    print(f"  오케스트레이터 결정: {result3.get('orchestrator_decision', '(미실행)')}")


# ── 메뉴 ──────────────────────────────────────────────────────────────────────

TESTS = {
    "1": ("고객 응대 챗봇 (자동 텍스트)",        test_customer_bot),
    "2": ("일일 리포트",                          test_report_generator),
    "3": ("관리자 알림",                          test_alert_manager),
    "4": ("멀티에이전트 시나리오",                test_multi_agent),
    "5": ("전체 실행 (1~4, 마이크 모드 제외)",    None),
    "6": ("고객 응대 챗봇 (마이크 입력, 인터랙티브)", test_customer_bot_mic),
    "7": ("STT 마이크 단독 테스트",               test_stt_mic_only),
    "8": ("STT 영상/오디오 파일 테스트",          test_stt_video_file),
}

async def main():
    from db.models import init_db
    await init_db()

    print("\n테스트할 모듈을 선택하세요:")
    for k, (name, _) in TESTS.items():
        print(f"  [{k}] {name}")

    choice = input("\n선택: ").strip()

    if choice == "5":
        for k, (name, fn) in TESTS.items():
            # 마이크 모드는 인터랙티브라 자동 실행 제외
            if fn and k not in {"6", "7", "8"}:
                await fn()
    elif choice in TESTS:
        _, fn = TESTS[choice]
        await fn()
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    asyncio.run(main())
