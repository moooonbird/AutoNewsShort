import os
import json
from datetime import datetime
from openai import OpenAI
import numpy as np

DATA_DIR = "data"
MEMORY_PATH = os.path.join(DATA_DIR, "story_memory.json")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_story_memory():
    ensure_data_dir()

    if not os.path.exists(MEMORY_PATH):
        return []

    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_story_memory(memory):
    ensure_data_dir()

    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def get_embedding(client, text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:8000]
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def build_story_text(story):
    return f"""
            Theme: {story.get("theme", "")}
            Narrative: {story.get("narrative", "")}
            Entities: {", ".join(story.get("entities", []))}
            Facts:
            {chr(10).join(story.get("facts", []))}
            """

def retrieve_similar_stories(client, new_facts, memory, top_k=3):
    if not memory:
        return []

    new_embedding = get_embedding(client, new_facts)

    scored = []

    for story in memory:
        story_embedding = story.get("embedding")

        if story_embedding is None:
            story_text = build_story_text(story)
            story_embedding = get_embedding(client, story_text)
            story["embedding"] = story_embedding

        score = cosine_similarity(new_embedding, story_embedding)
        scored.append((score, story))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {
            "similarity": score,
            "story": story
        }
        for score, story in scored[:top_k]
    ]

def build_story_matching_prompt(new_facts, similar_stories):
    return f"""
你是一個新聞故事記憶系統。

請判斷「新事實」是否應該併入相似故事，或建立新故事。

判斷規則：
- 如果新事實和相似故事有相同人物、公司、技術、產業主題，且能形成同一個長期主線，可以合併。
- 如果只是完全不同主題，請建立新故事。
- similarity 高不代表一定要合併，仍需根據語意判斷。
- 不要為了合併而硬合併。
- 請只根據提供資料判斷，不要加入外部知識。

相似故事候選：
{json.dumps(similar_stories, ensure_ascii=False, indent=2)}

新事實：
{new_facts}

請輸出 JSON：
{{
  "action": "merge" 或 "create",
  "target_story_id": "若 merge，填入 story_id；若 create，填 null",
  "reason": "判斷原因",
  "updated_theme": "更新後故事主題",
  "updated_narrative": "更新後故事主線",
  "entities": ["相關人物/公司/技術"],
  "facts_to_add": ["要加入故事的新事實"]
}}
"""


def decide_story_update(client, new_facts, similar_stories):
    prompt = build_story_matching_prompt(new_facts, similar_stories)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "你是一位嚴謹的新聞故事記憶管理器，只輸出合法 JSON。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

def update_story_memory(client, new_facts):
    memory = load_story_memory()

    similar_stories = retrieve_similar_stories(
        client=client,
        new_facts=new_facts,
        memory=memory,
        top_k=3
    )

    decision = decide_story_update(
        client=client,
        new_facts=new_facts,
        similar_stories=similar_stories
    )

    new_embedding = get_embedding(client, new_facts)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if decision["action"] == "merge":
        target_id = decision["target_story_id"]

        for story in memory:
            if story["story_id"] == target_id:
                story["theme"] = decision["updated_theme"]
                story["narrative"] = decision["updated_narrative"]

                story["entities"] = list(
                    set(story.get("entities", [])) | set(decision.get("entities", []))
                )

                story.setdefault("facts", [])
                story["facts"].extend(decision.get("facts_to_add", []))

                # 更新 embedding：用更新後 story 重新算
                story["embedding"] = get_embedding(client, build_story_text(story))
                story["updated_at"] = now
                break

    else:
        new_story = {
            "story_id": f"story_{len(memory) + 1:03d}",
            "theme": decision["updated_theme"],
            "narrative": decision["updated_narrative"],
            "entities": decision.get("entities", []),
            "facts": decision.get("facts_to_add", []),
            "embedding": new_embedding,
            "created_at": now,
            "updated_at": now
        }

        memory.append(new_story)

    save_story_memory(memory)
    return decision, memory