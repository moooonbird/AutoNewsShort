import asyncio
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import edge_tts
import os
from pydub import AudioSegment
from pydub.utils import which
import os
import asyncio
import librosa
import edge_tts
from pydub import AudioSegment

ffmpeg_dir = r"D:\ffmpeg\bin"
ffmpeg_path = r"D:\ffmpeg\bin\ffmpeg.exe"
ffprobe_path = r"D:\ffmpeg\bin\ffprobe.exe"

os.environ["PATH"] += os.pathsep + ffmpeg_dir

AudioSegment.converter = ffmpeg_path
AudioSegment.ffmpeg = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

print("ffmpeg exists:", os.path.exists(ffmpeg_path))
print("ffprobe exists:", os.path.exists(ffprobe_path))
print("pydub ffmpeg:", which("ffmpeg"))
print("pydub ffprobe:", which("ffprobe"))

voice = AudioSegment.from_file("newsflow_voice.mp3")
print("voice length:", len(voice), "ms")

# =========================
# OpenAI API Key
# =========================
with open("sk.txt", "r", encoding="utf-8") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

# =========================
# Config
# =========================
urls = [
    "https://technews.tw/2026/05/22/optical-interconnects-emerge-in-gpu-hbm-packaging/",
    "https://technews.tw/2026/05/24/fermi-telescope-captures-energy-source-of-slsn/",
    "https://technews.tw/2026/05/24/southwest-airlines-bans-taking-human-animal-robots-on-board/",
    "https://technews.tw/2026/05/23/jensen-huang-vera-rubin-mass-production-taiwan-supply-chain-h2-busy/",
    "https://finance.technews.tw/2026/05/23/jensen-huang-ai/",
    "https://technews.tw/2026/05/23/mitchell-institute-calls-for-us-space-force-to-deploy-on-the-moon-asap/",
    # add more news URLs here
]

beat_path = "beat.mp3"          # 你自己準備的背景音樂
voice_path = "newsflow_voice.mp3"
output_path = "autonewsong.mp3"

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "zh-TW,zh;q=0.9",
}

# =========================
# Fetch Article
# =========================
def fetch_article(url):
    res = requests.get(url, headers=headers, timeout=10)
    print(f"Status {res.status_code}: {url}")

    if res.status_code != 200:
        return ""

    soup = BeautifulSoup(res.text, "html.parser")

    article = (
        soup.select_one("article")
        or soup.select_one("div.indent")
        or soup.select_one("div.entry-content")
    )

    if article:
        text = article.get_text("\n", strip=True)
    else:
        text = soup.get_text("\n", strip=True)

    return text[:8000]


# =========================
# Generate Summary
# =========================
def generate_summary(articles):
    combined_text = "\n\n--- ARTICLE SPLIT ---\n\n".join(articles)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": """
你是一位科技新聞編輯，負責把多篇新聞整理成適合生成「每日新聞 Flow」的主題摘要。

你的任務不是逐篇摘要，而是做「新聞主題群整理」。

整理原則：
- 使用繁體中文
- 先辨識每篇新聞中的關鍵實體，例如人名、公司、技術名稱、國家、機構、產業主題
- 如果多篇新聞提到相同人物、公司、技術、產業主題，即使不是完全同一事件，也可以合併成同一組
- 如果是完全不同主題，請分開
- 不要重複描述相同事件
- 不要加入原文沒有的資訊
- 最終整理成 3~5 個「新聞主題群」
- 每個主題群都要適合後續改寫成短影音 news flow

輸出格式：
1. 主題：
   涵蓋重點：
   可唱式重點句：

2. 主題：
   涵蓋重點：
   可唱式重點句：
"""
            },
            {
                "role": "user",
                "content": combined_text
            }
        ]
    )

    return response.choices[0].message.content.strip()

# =========================
# Generate News Flow Lyrics
# =========================
def generate_news_flow(summary):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": """
你是一位「短影音新聞 Flow」創作者。
你的任務是根據新聞主題群，生成 20~35 秒內可以搭配 beat 的中文 news rap / news flow。

要求：
- 使用繁體中文
- 不要逐篇唱新聞，而是依照主題群組織內容
- 相同人物、公司、技術、產業主題請放在同一段
- 聽完要知道今天有哪些新聞重點
- 每句 6~12 個中文字左右
- 整體長度約 12~16 行
- 每個主題群最多 2~4 行
- 要有短影音感與記憶點
- 可以加入一點諧音梗、單押、雙押
- 資訊清楚優先，押韻其次
- 不要太文藝，不要抽象隱喻
- 不要加入摘要沒有的資訊
- 不需要 Verse / Chorus 標籤
"""
            },
            {
                "role": "user",
                "content": summary
            }
        ]
    )

    return response.choices[0].message.content.strip()


# =========================
# Text to Speech
# =========================
async def text_to_speech(text, output_file):
    communicate = edge_tts.Communicate(
        text=text,
        voice="zh-TW-HsiaoChenNeural",
        rate="+15%",
        pitch="+0Hz"
    )
    await communicate.save(output_file)


# =========================
# Mix Voice with Beat
# =========================
def mix_with_beat(voice_file, beat_file, output_file):
    voice = AudioSegment.from_file(voice_file)
    beat = AudioSegment.from_file(beat_file)

    # 調整音量
    voice = voice + 3
    beat = beat - 14

    # 如果 beat 太短，就循環
    while len(beat) < len(voice) + 2000:
        beat += beat

    # 裁切 beat 長度，比人聲多 2 秒
    beat = beat[:len(voice) + 2000]

    # 人聲延遲 500ms 進來
    final = beat.overlay(voice, position=500)

    final.export(output_file, format="mp3")


# 自動偵測 beat，讓人聲從第 N 拍後開始
def detect_vocal_start_ms(beat_file, wait_beats=4):
    y, sr = librosa.load(beat_file, sr=None, mono=True)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    if len(beat_frames) == 0:
        print("[WARN] No beat detected. Use default 3000ms.")
        return 3000

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) > wait_beats:
        start_time = beat_times[wait_beats]
    else:
        start_time = beat_times[0] + 2.0

    return int(start_time * 1000)


# 分行產生 TTS，每行用不同 pitch，讓它不要像平念
async def text_to_speech_with_pitch(text, output_file):
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    temp_files = []
    pitch_pattern = ["+4Hz", "+0Hz", "+6Hz", "-2Hz"]

    for i, line in enumerate(lines):
        temp_file = f"temp_voice_{i}.mp3"
        temp_files.append(temp_file)

        communicate = edge_tts.Communicate(
            text=line,
            voice="zh-TW-HsiaoChenNeural",
            rate="+18%",
            pitch=pitch_pattern[i % len(pitch_pattern)]
        )

        await communicate.save(temp_file)

    combined = AudioSegment.empty()

    for f in temp_files:
        seg = AudioSegment.from_file(f)

        # 每句之間加短暫停頓，讓 flow 比較像卡拍
        combined += seg + AudioSegment.silent(duration=180)

    combined.export(output_file, format="mp3")

    for f in temp_files:
        os.remove(f)


def mix_with_beat_auto_start(voice_file, beat_file, output_file):
    voice = AudioSegment.from_file(voice_file)
    beat = AudioSegment.from_file(beat_file)

    voice = voice + 3
    beat = beat - 14

    # vocal_start_ms = detect_vocal_start_ms(beat_file, wait_beats=4)
    vocal_start_ms = 21000

    print(f"Vocal starts at: {vocal_start_ms} ms")

    while len(beat) < len(voice) + vocal_start_ms + 2000:
        beat += beat

    beat = beat[:len(voice) + vocal_start_ms + 2000]

    final = beat.overlay(voice, position=vocal_start_ms)

    final.export(output_file, format="mp3")

# =========================
# Main
# =========================
def main():
    articles = []

    for url in urls:
        text = fetch_article(url)
        if text:
            articles.append(text)

    if not articles:
        print("No articles fetched.")
        return

    print("\n=== Generating Summary ===")
    summary = generate_summary(articles)
    print(summary)
    print("\n=== Generating News Flow ===")
    lyrics = generate_news_flow(summary)
    print(lyrics)

    print("\n=== Generating Voice ===")
    asyncio.run(text_to_speech_with_pitch(lyrics, voice_path))

    print("\n=== Mixing with Beat ===")
    mix_with_beat_auto_start(voice_path, beat_path, output_path)

    print(f"\nDone: {output_path}")


if __name__ == "__main__":
    main()