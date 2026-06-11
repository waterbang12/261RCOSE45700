"""
Nemotron-Personas-Korea 데이터셋 기반 한국인 페르소나 관리 모듈.
HuggingFace 스트리밍으로 첫 CACHE_SIZE개를 메모리에 캐시하고,
고객봇 응대 시 페르소나 컨텍스트를 제공한다.

데이터셋: nvidia/Nemotron-Personas-Korea (CC BY 4.0)
  - 100만 레코드, 700만 페르소나, 26개 필드
  - 직업/취미/음식/가족/요약 등 7가지 페르소나 내러티브
  - 한국 실제 인구통계(KOSIS) 기반 연령·지역·이름 분포
"""
import random
from typing import Optional

CACHE_SIZE = 200
_PERSONA_CACHE: list[dict] = []

_FIELD_MAP = {
    "name":                 "name",
    "age":                  "age",
    "sex":                  "sex",
    "occupation":           "occupation",
    "education_level":      "education",
    "province":             "province",
    "marital_status":       "marital_status",
    "professional_persona": "professional_persona",
    "culinary_persona":     "culinary_persona",
    "family_persona":       "family_persona",
    "summary":              "summary",
    "hobbies_and_interests":"hobbies",
    "skills_and_expertise": "skills",
    "career_goals_and_ambitions": "career_goals",
    "cultural_background":  "cultural_background",
}


def _load_personas(n: int = CACHE_SIZE) -> list[dict]:
    """HuggingFace streaming으로 첫 n개 페르소나를 로드한다."""
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "nvidia/Nemotron-Personas-Korea",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        personas = []
        for row in ds.take(n):
            entry = {dst: row.get(src, "") for src, dst in _FIELD_MAP.items()}
            personas.append(entry)
        print(f"[PERSONA] {len(personas)}개 페르소나 로드 완료")
        return personas
    except Exception as e:
        print(f"[PERSONA] 데이터셋 로드 실패 (오프라인 모드 동작): {e}")
        return []


def _ensure_cache() -> list[dict]:
    global _PERSONA_CACHE
    if not _PERSONA_CACHE:
        _PERSONA_CACHE = _load_personas()
    return _PERSONA_CACHE


def get_random_persona() -> Optional[dict]:
    """캐시에서 임의 페르소나를 반환한다."""
    cache = _ensure_cache()
    return random.choice(cache) if cache else None


def get_persona_by_age(min_age: int, max_age: int) -> Optional[dict]:
    """연령대에 맞는 페르소나를 반환한다. 없으면 임의 페르소나 반환."""
    cache = _ensure_cache()
    matches = [p for p in cache if isinstance(p.get("age"), int) and min_age <= p["age"] <= max_age]
    return random.choice(matches) if matches else get_random_persona()


def get_persona_by_occupation(keyword: str) -> Optional[dict]:
    """직업 키워드가 포함된 페르소나를 반환한다."""
    cache = _ensure_cache()
    keyword_lower = keyword.lower()
    matches = [p for p in cache if keyword_lower in (p.get("occupation") or "").lower()]
    return random.choice(matches) if matches else get_random_persona()


def format_persona_for_prompt(persona: dict) -> str:
    """
    페르소나 dict를 고객봇 시스템 프롬프트에 삽입할 텍스트로 변환한다.
    토큰 절약을 위해 핵심 필드만 포함한다.
    """
    lines = []
    if persona.get("name"):
        lines.append(f"이름: {persona['name']}")
    if persona.get("age"):
        lines.append(f"나이: {persona['age']}세")
    if persona.get("sex"):
        lines.append(f"성별: {persona['sex']}")
    if persona.get("occupation"):
        lines.append(f"직업: {persona['occupation']}")
    if persona.get("province"):
        lines.append(f"거주 지역: {persona['province']}")
    if persona.get("marital_status"):
        lines.append(f"혼인 상태: {persona['marital_status']}")
    if persona.get("education"):
        lines.append(f"학력: {persona['education']}")
    if persona.get("hobbies"):
        lines.append(f"관심사/취미: {persona['hobbies'][:120]}")
    if persona.get("culinary_persona"):
        lines.append(f"음식 성향: {persona['culinary_persona'][:120]}")
    if persona.get("professional_persona"):
        lines.append(f"직업 성향: {persona['professional_persona'][:120]}")
    if persona.get("summary"):
        lines.append(f"요약: {persona['summary'][:200]}")
    return "\n".join(lines)
