import operator
from typing import TypedDict, Optional, List, Annotated


class FacilityState(TypedDict, total=False):
    # 공통 (필수)
    zone_id: str
    trigger_type: str        # "safety" | "customer" | "report" | "insight"
    all_zone_ids: List[str]

    # 결정 노드가 발행하는 실행 의도. actuator 노드가 일괄 실행한다.
    # 여러 노드의 발행이 누적되도록 add 리듀서를 사용한다.
    pending_actions: Annotated[List[dict], operator.add]

    # 안전 에이전트 입력
    analysis_result: Optional[dict]  # AnalysisResult 직렬화 dict
    signals: Optional[dict]          # TriggerSignals dict (temperature, humidity 등)

    # 고객봇 에이전트 입력/출력
    user_message: Optional[str]
    customer_context: Optional[dict]
    customer_persona: Optional[dict]  # Nemotron-Personas-Korea 페르소나 dict
    tts_enabled: bool                # False면 TTS 생성 생략
    bot_response: Optional[dict]     # {message, audio_path, action}

    # 보고서 에이전트 출력
    report_text: Optional[str]
    anomaly_detected: bool

    # 인사이트 엔진 입력/출력
    district_profile: Optional[dict]  # DistrictProfile dict (상권 인구통계)
    insight_result: Optional[dict]    # insight_node 출력 (요약/인사이트/피크시간/적합업종)
    recommendations: Optional[dict]   # recommendation_node 출력 (상품 구성)

    # 오케스트레이터
    cross_zone_request: Optional[dict]  # {action: {...}} 시설 전체 영향 요청
    conflict_detected: bool             # 에이전트 간 판단 충돌
    escalate_to_human: bool             # 관리자 에스컬레이션 필요
    orchestrator_decision: Optional[str]


def make_safety_state(
    zone_id: str,
    all_zone_ids: List[str],
    analysis_result: dict,
    signals: dict,
) -> FacilityState:
    return FacilityState(
        zone_id=zone_id,
        trigger_type="safety",
        all_zone_ids=all_zone_ids,
        analysis_result=analysis_result,
        signals=signals,
        pending_actions=[],
        anomaly_detected=False,
        conflict_detected=False,
        escalate_to_human=False,
    )


def make_customer_state(
    zone_id: str,
    all_zone_ids: List[str],
    user_message: str,
    customer_context: dict,
    tts_enabled: bool = True,
    customer_persona: Optional[dict] = None,
) -> FacilityState:
    return FacilityState(
        zone_id=zone_id,
        trigger_type="customer",
        all_zone_ids=all_zone_ids,
        user_message=user_message,
        customer_context=customer_context,
        customer_persona=customer_persona,
        tts_enabled=tts_enabled,
        pending_actions=[],
        anomaly_detected=False,
        conflict_detected=False,
        escalate_to_human=False,
    )


def make_report_state(
    zone_id: str,
    all_zone_ids: List[str],
) -> FacilityState:
    return FacilityState(
        zone_id=zone_id,
        trigger_type="report",
        all_zone_ids=all_zone_ids,
        pending_actions=[],
        anomaly_detected=False,
        conflict_detected=False,
        escalate_to_human=False,
    )


def make_insight_state(
    district_profile: dict,
    all_zone_ids: Optional[List[str]] = None,
) -> FacilityState:
    """상권 인구통계 프로필 → 인사이트 엔진 진입 상태.

    인사이트 경로는 구역 개념이 없으므로 zone_id에 district 이름을 넣어
    reconcile/actuator의 zone_id 참조를 안전하게 만족시킨다 (부수효과는 없음).
    """
    district = district_profile.get("district", "unknown")
    return FacilityState(
        zone_id=district,
        trigger_type="insight",
        all_zone_ids=all_zone_ids or [district],
        district_profile=district_profile,
        pending_actions=[],
        anomaly_detected=False,
        conflict_detected=False,
        escalate_to_human=False,
    )
