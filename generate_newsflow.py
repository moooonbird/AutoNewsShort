import os
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from data.story_memory import update_story_memory
import json

DATA_DIR = "data"
LYRICS_V1_PATH = os.path.join(DATA_DIR, "lyrics_v1.txt")
LYRICS_V2_PATH = os.path.join(DATA_DIR, "lyrics_v2.txt")
BEST_LYRICS_PATH = os.path.join(DATA_DIR, "best_lyrics.txt")
FACTS_PATH = os.path.join(DATA_DIR, "facts.txt")
STORY_PATH = os.path.join(DATA_DIR, "story_summary.txt")
LYRICS_PATH = os.path.join(DATA_DIR, "lyrics.txt")
FLOW_PLAN_PATH = os.path.join(DATA_DIR, "flow_plan.txt")
FACT_CHECK_PATH = os.path.join(DATA_DIR, "fact_check.txt")
EVAL_PATH = os.path.join(DATA_DIR, "evaluation.txt")
FINAL_LYRICS_PATH = os.path.join(DATA_DIR, "final_lyrics.txt")

# URLS = [
#     "https://technews.tw/2026/05/22/optical-interconnects-emerge-in-gpu-hbm-packaging/",
#     "https://technews.tw/2026/05/24/fermi-telescope-captures-energy-source-of-slsn/",
#     "https://technews.tw/2026/05/24/southwest-airlines-bans-taking-human-animal-robots-on-board/",
#     "https://technews.tw/2026/05/23/jensen-huang-vera-rubin-mass-production-taiwan-supply-chain-h2-busy/",
#     "https://finance.technews.tw/2026/05/23/jensen-huang-ai/",
#     "https://technews.tw/2026/05/23/mitchell-institute-calls-for-us-space-force-to-deploy-on-the-moon-asap/",
# ]
URLS = [
    "https://finance.technews.tw/2026/05/24/general-strike/",
    "https://technews.tw/2026/05/24/china-students-stressed-by-dozens-of-surveillance-cameras-in-university-class-told-devices-are-digital-media-teaching-aids/"
    "https://technews.tw/2026/05/24/wemo-seeks-vehicle-supply-partnership-with-gogoro/",
    "https://finance.technews.tw/2026/05/24/regulations-should-be-strengthened/",
    "https://finance.technews.tw/2026/05/24/3coins-coming-soon/",
    "https://finance.technews.tw/2026/05/24/2x-leveraged-etfs/",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "zh-TW,zh;q=0.9",
}


def load_api_key(path="sk.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


client = OpenAI(api_key=load_api_key())


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)


def fetch_article(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        print(f"Status {res.status_code}: {url}")

        if res.status_code != 200:
            return ""

        soup = BeautifulSoup(res.text, "html.parser")

        article = (
            soup.select_one("article")
            or soup.select_one("div.indent")
            or soup.select_one("div.entry-content")
        )

        text = article.get_text("\n", strip=True) if article else soup.get_text("\n", strip=True)
        return text[:8000]

    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return ""


def chat(system_prompt, user_prompt, model="gpt-4.1-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def create_flow_plan(facts, compressed_story):
    system_prompt = """
你是一位短影音 news flow 內容企劃。

你的任務不是直接寫歌詞，而是規劃 news flow 的內容結構。

請根據壓縮後故事摘要，安排：
1. 開場 hook
2. 主線段落
3. 支線段落
4. Comic Relief / Fun News Slot
5. 收尾方向

要求：
- 使用繁體中文
- 不要加入可用事實清單沒有的資訊
- 主線新聞要負責主要資訊量
- Comic Relief 只能簡短帶過，用來增加趣味或轉場
- 如果有生活、消費、趣味新聞，可以放在 Comic Relief 或 Outro
- punchline 可以口語、有梗，但不能改變事實
- 目標是避免最後歌詞變成新聞稿

輸出格式：

主 hook：
...

段落規劃：
1. 角色：Main Story
   主題：
   要講的事實：
   語氣：
   punchline：

2. 角色：Supporting Story
   主題：
   要講的事實：
   語氣：
   punchline：

3. 角色：Comic Relief
   主題：
   要講的事實：
   語氣：
   punchline：

結尾方向：
...
"""

    user_prompt = f"""
可用事實清單：
{facts}

壓縮後故事摘要：
{compressed_story}
"""

    return chat(system_prompt, user_prompt, model="gpt-4.1-mini")

def extract_facts(articles):
    combined_text = "\n\n--- ARTICLE SPLIT ---\n\n".join(articles)

    system_prompt = """
你是一位嚴謹的科技新聞事實抽取器。

請從多篇新聞中抽取「後續生成歌詞可以使用的事實」。

要求：
- 使用繁體中文
- 只列出原文明確提到的事實
- 不要加入背景知識
- 不要推論不存在的因果關係
- 保留人名、公司、技術、機構、事件重點
- 相似事實可以合併
- 每點一句話
"""

    return chat(system_prompt, combined_text)


def build_story_summary_from_memory(memory):
    system_prompt = """
你是一位科技新聞編輯。

請根據 story memory 產生今日故事摘要。

要求：
- 使用繁體中文
- 不要逐篇摘要
- 根據 story memory 整理出 3~5 個主要故事群
- 每個故事群包含：
  1. 主題
  2. 核心事實
  3. 故事主線
  4. 可唱式重點句
- 不要加入 story memory 沒有的資訊
"""

    user_prompt = json.dumps(memory, ensure_ascii=False, indent=2)

    return chat(system_prompt, user_prompt)

def rank_and_compress_stories(memory):
    system_prompt = """
你是一位短影音新聞內容主編。

請根據 story memory，整理今天最適合生成 news flow 的故事結構。

重要：
不要只挑最重要的新聞。
請同時考慮每則新聞在短影音內容中的角色。

可用角色：
- Main Story：今天最重要、最有主線感的新聞
- Supporting Story：補充主線或延伸世界狀態的新聞
- Comic Relief：較輕鬆、有生活感、可作為轉場或趣味插曲的新聞
- Outro Topic：適合作為最後一句收尾的新聞

要求：
- 使用繁體中文
- 不要逐篇摘要
- 不要加入 story memory 沒有的資訊
- Main Story 最多 2 個
- Supporting Story 最多 1 個
- Comic Relief 最多 1 個
- 如果有生活、消費、趣味新聞，即使重要性較低，也可以放進 Comic Relief
- 每個故事只保留 1~2 句重點
- 最後給一個整體主題句

輸出格式：

整體主題：
...

Main Story：
1. 主題：
   重點：
   建議歌詞比重：高/中/低

Supporting Story：
1. 主題：
   重點：
   建議歌詞比重：中/低

Comic Relief：
1. 主題：
   重點：
   建議歌詞比重：低

Outro Topic：
1. 主題：
   重點：
   建議歌詞比重：低
"""

    user_prompt = json.dumps(memory, ensure_ascii=False, indent=2)

    return chat(system_prompt, user_prompt, model="gpt-4.1-mini")

def generate_lyrics(facts, compressed_story, flow_plan, style_hint=""):
    system_prompt = f"""
你是一位「短影音新聞 Flow」創作者。

請根據 flow plan 生成 20~30 秒內可搭配 beat 的中文 news flow。

風格方向：
{style_hint}

重要要求：
- 使用繁體中文
- 長度 8~10 行
- 每行 5~9 個中文字左右
- 要像短影音 flow，不要像新聞稿
- 句子要短、有 punch、有節奏
- 可以使用口語化比喻
- 可以有單押、雙押、諧音梗
- 但不能新增可用事實清單沒有的資訊
- Main Story 佔主要篇幅
- Supporting Story 簡短帶過
- Comic Relief 最多 1~2 行，用來增加趣味或收尾
- 不要讓 Comic Relief 搶走主線
- 不需要 Verse / Chorus 標籤
"""

    user_prompt = f"""
可用事實清單：
{facts}

壓縮後故事摘要：
{compressed_story}

Flow Plan：
{flow_plan}

請生成最終 news flow 歌詞。
"""

    return chat(system_prompt, user_prompt, model="gpt-4.1")


def fact_check(facts, lyrics):
    system_prompt = """
你是一位新聞事實檢查員。

請檢查歌詞是否違反可用事實清單。

輸出格式：
是否有事實錯誤：是/否
錯誤句子：
修改建議：

如果沒有錯誤，請簡短回答沒有明顯事實錯誤。
"""

    user_prompt = f"""
可用事實清單：
{facts}

歌詞：
{lyrics}
"""

    return chat(system_prompt, user_prompt)

def rank_lyrics_candidates(
    facts,
    compressed_story,
    lyrics_a,
    lyrics_b
):
    system_prompt = """
    你是一位短影音內容總編輯。

    請比較兩組 meme-style news flow 歌詞，
    選出更適合短影音內容的一版。

    評估重點：
    - factual correctness
    - punchline strength
    - hook memorability
    - rhythm / flow
    - 是否口語自然
    - 是否有短影音感
    - 是否保留新聞主線
    - 是否避免新聞稿感

    請不要偏好資訊較完整的版本。
    如果一版資訊較完整但比較無聊，應該扣分。
    請選出「更有記憶點、但仍然事實正確」的一版。

    請輸出：

    最佳版本：A / B

    原因：
    ...

    A優點：
    ...

    A缺點：
    ...

    B優點：
    ...

    B缺點：
    ...
    """

    user_prompt = f"""
可用事實清單：
{facts}

壓縮後故事摘要：
{compressed_story}

Lyrics A：
{lyrics_a}

Lyrics B：
{lyrics_b}
"""

    return chat(system_prompt, user_prompt, model="gpt-4.1-mini")


def evaluate_lyrics(facts, compressed_story, lyrics):
    system_prompt = """
    你是一位短影音 news flow 品質評估器。

    請評估以下歌詞是否：
    - 有 short-form flow 感
    - 有記憶點
    - 不像新聞稿
    - 有清楚主線
    - 不會太冗長
    - 保持事實正確

    請評分（1~5）：

    1. factual_correctness
    2. clarity
    3. flow_quality
    4. length_control
    5. punchline_strength

    請輸出：

    - 五項分數
    - 總分
    - 是否 acceptable：是/否
    - improvement potential：low / medium / high
    - 最弱的部分
    - 最值得修改的兩句
    - 修改方向

    注意：
    acceptable 不代表最佳版本。
    如果 flow 還有明顯優化空間，
    請將 improvement potential 設為 medium 或 high。
    """

    user_prompt = f"""
可用事實清單：
{facts}

壓縮後故事摘要：
{compressed_story}

歌詞：
{lyrics}
"""

    return chat(system_prompt, user_prompt, model="gpt-4.1-mini")

def revise_lyrics(facts, compressed_story, flow_plan, lyrics, evaluation):
    system_prompt = """
你是一位短影音新聞 flow 編輯。

請根據評估結果，修正原本歌詞。

修正目標：
- 保留事實正確性
- 縮短歌詞
- 讓句子更短、更有 punch
- 不要像新聞稿
- 不要加入可用事實清單沒有的資訊
- 長度控制在 8~10 行
- 每行 5~9 個中文字左右
- 可以有口語感、單押、雙押或諧音梗
- 不需要 Verse / Chorus 標籤

請只輸出修改後的最終歌詞。
"""

    user_prompt = f"""
可用事實清單：
{facts}

壓縮後故事摘要：
{compressed_story}

Flow Plan：
{flow_plan}

原本歌詞：
{lyrics}

評估結果：
{evaluation}
"""

    return chat(system_prompt, user_prompt, model="gpt-4.1")


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    ensure_dirs()

    articles = []
    for url in URLS:
        text = fetch_article(url)
        if text:
            articles.append(text)

    if not articles:
        print("No articles fetched.")
        return

    print("\n=== Extracting Facts ===")
    facts = extract_facts(articles)
    print(facts)
    save_text(FACTS_PATH, facts)

    print("\n=== Updating Story Memory ===")
    decision, memory = update_story_memory(client, facts)
    print(decision)

    print("\n=== Ranking and Compressing Stories ===")
    compressed_story = rank_and_compress_stories(memory)
    print(compressed_story)
    save_text(STORY_PATH, compressed_story)

    print("\n=== Creating Flow Plan ===")
    flow_plan = create_flow_plan(facts, compressed_story)
    print(flow_plan)
    save_text(FLOW_PLAN_PATH, flow_plan)

    print("\n=== Generating Lyrics Candidate A ===")

    lyrics_v1 = generate_lyrics(
        facts,
        compressed_story,
        flow_plan,
        style_hint="""
    偏短影音 meme flow。
    特色：
    - 口語自然
    - punchline 清楚
    - 主線明確
    - Comic Relief 簡短但有記憶點
    - 不要資訊導向，不要新聞稿感
    """
    )

    print(lyrics_v1)
    save_text(LYRICS_V1_PATH, lyrics_v1)

    print("\n=== Generating Lyrics Candidate B ===")
    
    lyrics_v2 = generate_lyrics(
        facts,
        compressed_story,
        flow_plan,
        style_hint="""
    偏洗腦型 meme flow。
    特色：
    - hook 更強
    - 節奏更短
    - 更像短影音 BGM 可以搭的詞
    - Comic Relief 可以作為趣味收尾
    - 不要資訊導向，不要新聞稿感
    """
    )

    print(lyrics_v2)
    save_text(LYRICS_V2_PATH, lyrics_v2)

    print("\n=== Ranking Lyrics Candidates ===")

    ranking_result = rank_lyrics_candidates(
        facts,
        compressed_story,
        lyrics_v1,
        lyrics_v2
    )

    print(ranking_result)

    if "最佳版本：B" in ranking_result:
        best_lyrics = lyrics_v2
    else:
        best_lyrics = lyrics_v1

    save_text(BEST_LYRICS_PATH, best_lyrics)

    print("\n=== Evaluating Lyrics ===")
    evaluation = evaluate_lyrics(
        facts,
        compressed_story,
        best_lyrics
    )
    print(evaluation)
    save_text(EVAL_PATH, evaluation)

    # 不再只是 pass / fail
    # 而是判斷是否還有明顯 improvement potential

    need_rewrite = False

    if (
        "是否需要重寫：是" in evaluation
        or "需要重寫：是" in evaluation
    ):
        need_rewrite = True

    # 即使 acceptable，只要 evaluator 認為還有明顯優化空間，也進行 revise
    if (
        "flow_quality：4" in evaluation
        or "length_control：4" in evaluation
        or "improvement potential：high" in evaluation.lower()
    ):
        need_rewrite = True

    if need_rewrite:
        print("\n=== Revising Lyrics ===")

        revised_lyrics = revise_lyrics(
            facts,
            compressed_story,
            flow_plan,
            best_lyrics,
            evaluation
        )

        print(revised_lyrics)

        print("\n=== Re-evaluating Revised Lyrics ===")

        revised_evaluation = evaluate_lyrics(
            facts,
            compressed_story,
            revised_lyrics
        )

        print(revised_evaluation)

        # 可以再加入 ranking
        # 選擇比較好的版本

        final_lyrics = revised_lyrics
        save_text(EVAL_PATH, revised_evaluation)

    else:
        final_lyrics = best_lyrics

    print("\n=== Final Lyrics ===")
    print(final_lyrics)

    save_text(FINAL_LYRICS_PATH, final_lyrics)

    print("\n=== Fact Checking Final Lyrics ===")
    check_result = fact_check(facts, final_lyrics)
    print(check_result)
    save_text(FACT_CHECK_PATH, check_result)

    print("\nDone.")


if __name__ == "__main__":
    main()