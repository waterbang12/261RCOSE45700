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
        "current_hole": 15,
        "remaining_min": 10,
        "reserved_min": 60,
    }

    tests = [
        "좀 더 춥게 해줘",
        "지금 몇 홀이야?",
        "시간 연장하고 싶어",
        "너무 더워요",
    ]

    for msg in tests:
        print(f"\n고객: {msg}")
        result = await respond(msg, "1번 타석", context, tts=False)
        print(f"봇:   {result['message']}")
        if result["action"]:
            print(f"액션: {result['action']}")


async def test_coaching_engine():
    print("\n" + "="*50)
    print("[ 자세 피드백 테스트 ]")
    print("="*50)
    from llm_module.coaching_engine import posture_feedback, beginner_guide

    # 실제 스윙 관절 좌표 (임의값)
    pose_keypoints = {
        "left_shoulder":  [310, 150],
        "right_shoulder": [420, 148],
        "left_elbow":     [280, 230],
        "right_elbow":    [460, 220],
        "left_wrist":     [260, 310],
        "right_wrist":    [490, 300],
        "left_hip":       [320, 340],
        "right_hip":      [410, 338],
    }

    print("\n스윙 자세 피드백 요청 중...")
    audio_path = await posture_feedback(pose_keypoints, "1번 타석")
    print(f"음성 파일 저장: {audio_path}")

    print("\n입문자 1단계 가이드 요청 중...")
    audio_path = await beginner_guide(step=1, bay_id="1번 타석")
    print(f"음성 파일 저장: {audio_path}")


async def test_report_generator():
    print("\n" + "="*50)
    print("[ 일일 리포트 테스트 ]")
    print("="*50)
    from db.models import init_db
    from llm_module.report_generator import generate_daily_report

    await init_db()
    report = await generate_daily_report(["1번 타석", "2번 타석", "3번 타석"])
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
        evidence="Person moving equipment toward exit",
        action_required="Notify manager immediately",
    )
    print("\n도난 감지 알림 테스트:")
    await handle(result, "2번 타석")

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
    await handle(result2, "1번 타석")


# ── 메뉴 ──────────────────────────────────────────────────────────────────────

TESTS = {
    "1": ("고객 응대 챗봇",   test_customer_bot),
    "2": ("자세 피드백 + TTS", test_coaching_engine),
    "3": ("일일 리포트",      test_report_generator),
    "4": ("관리자 알림",      test_alert_manager),
    "5": ("전체 실행",        None),
}

async def main():
    print("\n테스트할 모듈을 선택하세요:")
    for k, (name, _) in TESTS.items():
        print(f"  [{k}] {name}")

    choice = input("\n선택: ").strip()

    if choice == "5":
        for k, (name, fn) in TESTS.items():
            if fn:
                await fn()
    elif choice in TESTS:
        _, fn = TESTS[choice]
        await fn()
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    asyncio.run(main())
