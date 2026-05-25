import os
import asyncio
import edge_tts
from pydub import AudioSegment

DATA_DIR = "data"
AUDIO_DIR = "audio"

LYRICS_PATH = os.path.join(DATA_DIR, "final_lyrics.txt")
BEAT_PATH = os.path.join(AUDIO_DIR, "beat.mp3")
VOICE_PATH = os.path.join(AUDIO_DIR, "voice.mp3")
FINAL_PATH = os.path.join(AUDIO_DIR, "final.mp3")

# 如果你的 ffmpeg 不在 PATH，取消註解並改成你的實際路徑
AudioSegment.converter = r"D:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"D:\ffmpeg\bin\ffprobe.exe"


def ensure_dirs():
    os.makedirs(AUDIO_DIR, exist_ok=True)


def read_lyrics():
    if not os.path.exists(LYRICS_PATH):
        raise FileNotFoundError(f"Cannot find lyrics file: {LYRICS_PATH}")

    with open(LYRICS_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


async def text_to_speech_with_pitch(text, output_file):
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if not lines:
        raise ValueError("Lyrics are empty.")

    temp_files = []
    pitch_pattern = ["+4Hz", "+0Hz", "+6Hz", "-2Hz"]

    combined = AudioSegment.empty()

    for i, line in enumerate(lines):
        temp_file = os.path.join(AUDIO_DIR, f"temp_voice_{i}.mp3")
        temp_files.append(temp_file)

        communicate = edge_tts.Communicate(
            text=line,
            voice="zh-TW-HsiaoChenNeural",
            rate="+18%",
            pitch=pitch_pattern[i % len(pitch_pattern)],
        )

        await communicate.save(temp_file)

        seg = AudioSegment.from_file(temp_file)

        # 每句之間短暫停頓
        combined += seg + AudioSegment.silent(duration=150)

    combined.export(output_file, format="mp3")

    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)


def mix_with_beat(
    voice_file,
    beat_file,
    output_file,
    vocal_start_ms=6000,
    voice_gain_db=3,
    beat_gain_db=-6,
):
    if not os.path.exists(voice_file):
        raise FileNotFoundError(f"Cannot find voice file: {voice_file}")

    if not os.path.exists(beat_file):
        raise FileNotFoundError(f"Cannot find beat file: {beat_file}")

    voice = AudioSegment.from_file(voice_file)
    beat = AudioSegment.from_file(beat_file)

    voice = voice + voice_gain_db
    beat = beat + beat_gain_db

    while len(beat) < len(voice) + vocal_start_ms + 2000:
        beat += beat

    beat = beat[: len(voice) + vocal_start_ms + 2000]

    final = beat.overlay(voice, position=vocal_start_ms)
    final.export(output_file, format="mp3")


def main():
    ensure_dirs()

    lyrics = read_lyrics()

    print("\n=== Generating Voice ===")
    asyncio.run(text_to_speech_with_pitch(lyrics, VOICE_PATH))
    print(f"Voice saved to: {VOICE_PATH}")

    print("\n=== Mixing With Beat ===")
    mix_with_beat(
        voice_file=VOICE_PATH,
        beat_file=BEAT_PATH,
        output_file=FINAL_PATH,
        vocal_start_ms=6000,
        voice_gain_db=3,
        beat_gain_db=-6,
    )

    print(f"\nDone: {FINAL_PATH}")


if __name__ == "__main__":
    main()