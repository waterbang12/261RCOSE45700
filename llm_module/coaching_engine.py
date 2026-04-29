import os
import asyncio
import pathlib
from openai import OpenAI

client = OpenAI()

TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("TTS_VOICE", "nova")
AUDIO_DIR = pathlib.Path("audio_cache")
AUDIO_DIR.mkdir(exist_ok=True)

COACHING_PROMPT = """
당신은 무인 스크린골프장의 친절한 스윙 코치입니다.
골퍼의 스윙 직후 관절 좌표 데이터를 받아 핵심 문제점 한 가지만 짧고 명확하게 한국어로 말해주세요.
두 문장 이내로 작성하고, 전문 용어보다 쉬운 표현을 사용하세요.
"""


async def _tts(text: str, filename: str) -> pathlib.Path:
    """텍스트 → MP3 파일 생성"""
    path = AUDIO_DIR / filename
    response = await asyncio.to_thread(
        client.audio.speech.create,
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
    )
    response.stream_to_file(str(path))
    return path


async def posture_feedback(pose_keypoints: dict, bay_id: str) -> str:
    """
    OpenCV/MediaPipe에서 추출한 관절 좌표를 받아
    LLM 피드백 텍스트 생성 + TTS 파일 저장.

    pose_keypoints 예시:
    {
        "left_shoulder": [x, y],
        "right_shoulder": [x, y],
        "left_elbow": [x, y],
        ...
    }
    반환값: 음성 파일 경로 (str)
    """
    user_content = f"스윙 직후 관절 좌표:\n{pose_keypoints}"

    response = await asyncio.to_thread(
        client.chat.completions.create,
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": COACHING_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=100,
    )
    feedback_text = response.choices[0].message.content.strip()
    print(f"[{bay_id}][COACHING] {feedback_text}")

    filename = f"coaching_{bay_id}_{id(feedback_text)}.mp3"
    audio_path = await _tts(feedback_text, filename)
    return str(audio_path)


async def beginner_guide(step: int, bay_id: str) -> str:
    """
    입문자용 단계별 안내 음성 생성.
    step: 1(그립), 2(어드레스), 3(백스윙), 4(임팩트), 5(팔로우스루)
    """
    steps = {
        1: "클럽 그립을 잡는 방법을 안내해드릴게요. 양손으로 클럽을 가볍게 쥐고, 엄지손가락은 샤프트 위를 향하게 해주세요.",
        2: "어드레스 자세입니다. 발을 어깨너비로 벌리고, 무릎을 살짝 구부린 뒤 허리를 앞으로 숙여 공을 향해 정렬해주세요.",
        3: "백스윙 시작합니다. 클럽을 천천히 오른쪽으로 올리면서 체중을 오른발로 이동시켜주세요.",
        4: "임팩트 구간입니다. 체중을 왼발로 옮기며 클럽 페이스가 공에 정면으로 맞도록 회전해주세요.",
        5: "팔로우스루입니다. 스윙 후 클럽이 왼쪽 어깨까지 자연스럽게 따라오도록 마무리해주세요.",
    }
    text = steps.get(step, "좋은 스윙이었습니다! 계속 연습해보세요.")
    filename = f"guide_{bay_id}_step{step}.mp3"
    audio_path = await _tts(text, filename)
    return str(audio_path)
