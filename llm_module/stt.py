"""
공용 STT 유틸리티 — 마이크 녹음 + OpenAI Whisper 전사.

테스트 모듈에서 사용자가 직접 마이크에 말하면 한국어 텍스트로 변환해
customer_bot에 그대로 전달할 수 있게 한다.

의존성: sounddevice, soundfile (requirements.txt 참고)
"""
import asyncio
import os
import pathlib
import shutil
import subprocess
import tempfile

from openai import OpenAI

STT_MODEL    = os.getenv("STT_MODEL", "whisper-1")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "ko")
SAMPLE_RATE  = 16000  # Whisper 권장 샘플레이트


client = OpenAI()


def _record_blocking(seconds: float, samplerate: int, device: int | str | None = None) -> tuple[pathlib.Path, "numpy.ndarray"]:
    """sounddevice로 동기 녹음 → 임시 WAV 파일 저장."""
    import sounddevice as sd  # 지연 import (마이크 미사용 모드에서는 로드 안 되도록)
    import soundfile as sf

    frames = sd.rec(
        int(seconds * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="int16",
        device=device,
    )
    sd.wait()

    fd, name = tempfile.mkstemp(suffix=".wav", prefix="stt_")
    os.close(fd)
    path = pathlib.Path(name)
    sf.write(str(path), frames, samplerate)
    return path, frames


async def transcribe_audio_file(path: str | pathlib.Path) -> str:
    """이미 존재하는 오디오 파일을 Whisper API로 전사."""
    audio_path = pathlib.Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"[STT] Whisper로 전사 중: {audio_path}")
    with audio_path.open("rb") as f:
        result = await asyncio.to_thread(
            client.audio.transcriptions.create,
            model=STT_MODEL,
            file=f,
            language=STT_LANGUAGE,
        )
    return (result.text or "").strip()


async def extract_audio_from_video(video_path: str | pathlib.Path) -> pathlib.Path:
    """
    영상 파일에서 오디오를 WAV로 추출.
    ffmpeg가 PATH에 있어야 한다. RTSP live stream이 아닌 저장된 mp4/mov/avi 테스트용.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to extract audio from video files.")

    src = pathlib.Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"Video file not found: {src}")

    fd, name = tempfile.mkstemp(suffix=".wav", prefix="video_audio_")
    os.close(fd)
    out = pathlib.Path(name)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        str(out),
    ]
    print(f"[STT] 영상에서 오디오 추출 중: {src}")
    await asyncio.to_thread(
        subprocess.run,
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out


async def transcribe_video_file(video_path: str | pathlib.Path) -> str:
    """영상 파일의 오디오를 추출한 뒤 Whisper API로 전사."""
    audio_path = await extract_audio_from_video(video_path)
    try:
        return await transcribe_audio_file(audio_path)
    finally:
        try:
            audio_path.unlink()
        except OSError:
            pass


def list_input_devices() -> None:
    """마이크 입력 장치 목록 출력."""
    import sounddevice as sd

    print("[STT] 사용 가능한 오디오 장치:")
    print(sd.query_devices())


async def record_and_transcribe(seconds: float = 5.0, device: int | str | None = None) -> str:
    """
    `seconds`초 동안 마이크에서 녹음 → Whisper API로 한국어 전사.
    반환: 전사된 텍스트(strip). 인식 실패/무음이면 빈 문자열.
    """
    print(f"[STT] {seconds:.0f}초간 녹음합니다. 지금 말씀하세요...")
    path, _ = await asyncio.to_thread(_record_blocking, seconds, SAMPLE_RATE, device)
    print("[STT] 녹음 완료. Whisper로 전사 중...")

    try:
        text = await transcribe_audio_file(path)
    finally:
        try:
            path.unlink()
        except OSError:
            pass

    return text


def play_audio(path: str) -> None:
    """결과 음성을 스피커로 재생 (테스트 편의용). 실패해도 조용히 패스."""
    try:
        import soundfile as sf
        import sounddevice as sd
        data, sr = sf.read(path)
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        print(f"[STT] 재생 실패(무시): {e}")
