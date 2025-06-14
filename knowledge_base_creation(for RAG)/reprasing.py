import os
import json
import time
import random
import re
import google.generativeai as genai

# Set Gemini API key
model = genai.GenerativeModel("gemini-1.5-flash")
os.environ["GOOGLE_API_KEY"] = "AIzaSyCoFnXOxt8Hau3kQZuVTX3llO7R_bQh5ss"

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

INPUT_FILE = "classification_algorithms_knowledge_base1.json"
OUTPUT_FILE = "rephrased_output.json"
PROGRESS_FILE = "progress.json"

# Load input
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# Load or create output and progress trackers
rephrased_data = json.load(open(OUTPUT_FILE, "r", encoding="utf-8")) if os.path.exists(OUTPUT_FILE) else {}
progress = json.load(open(PROGRESS_FILE, "r", encoding="utf-8")) if os.path.exists(PROGRESS_FILE) else {}

# All valid topics and subtopics
TOPIC_SUBTOPIC_MAP = {
    "Logistic Regression": ["Introduction", "Mathematical Foundation", "Cost Function and Optimization", "Regularization", "Multiclass Logistic Regression", "Implementation"],
    "K-Nearest Neighbors": ["Introduction", "Distance Metrics", "Choosing k", "Algorithm Mechanics", "Scaling and Normalization", "Implementation"],
    "Random Forest": ["Introduction", "Bootstrap Aggregation", "Feature Randomness", "Voting and Averaging", "Hyperparameters", "Implementation"],
    "Decision Trees": ["Introduction", "Splitting Criteria", "Tree Construction", "Pruning", "Prediction", "Implementation"],
    "Naïve Bayes": ["Introduction", "Probability Basics", "Gaussian Naïve Bayes", "Multinomial Naïve Bayes", "Bernoulli Naïve Bayes", "Implementation"],
    "Support Vector Machines": ["Introduction", "Linear SVM", "Kernel Trick", "Mathematical Foundation", "Regularization and Cost", "Implementation"]
}

# Difficulty level mapping
LEVEL_MAP = {
    "level 1": "chunks_level1",
    "level 2": "chunks_level2",
    "level 3": "chunks_level3"
}

def parse_gemini_response(text):
    blocks = []
    paragraphs = text.strip().split("Rephrased:")

    for para in paragraphs:
        lines = para.strip().splitlines()
        if not lines or lines[0].strip().lower().startswith("paragraph"):
            continue

        rephrased_text = lines[0].strip()
        if len(rephrased_text) < 25:
            continue

        content = {"text": rephrased_text, "topics": []}
        current = {}

        for line in lines[1:]:
            line = line.strip()
            if line.lower().startswith("- topic:"):
                if current:
                    content["topics"].append(current)
                    current = {}
                current["topic"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("subtopic:"):
                current["subtopic"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("difficulty level:"):
                current["level"] = line.split(":", 1)[1].strip()

        if current:
            content["topics"].append(current)

        # Only keep this paragraph if valid topics exist
        if content["topics"]:
            blocks.append(content)

    return blocks


# --- Gemini API Call ---
def rephrase_and_classify_chunks(chunks):
    topic_prompt = "\n".join(
        f"- {topic}: {', '.join(subs)}" for topic, subs in TOPIC_SUBTOPIC_MAP.items()
    )

    prompt = (
        "You are given technical paragraphs related to supervised learning.\n"
        "For each paragraph:\n"
        "1. Rephrase it clearly in natural language (no headings).\n"
        "2. Classify it under multiple Main Topics (if applicable).\n"
        "3. For each topic, assign one valid Subtopic and a conceptual Difficulty Level.\n\n"
        "VALID Main Topics and their Subtopics:\n"
        f"{topic_prompt}\n\n"
        "Format your output as follows for EACH paragraph:\n"
        "Rephrased: [your rephrased paragraph]\n"
        "Classifications:\n"
        "- Topic: [main topic 1]\n  Subtopic: [subtopic 1]\n  Difficulty Level: Level X\n"
        "- Topic: [main topic 2]\n  Subtopic: [subtopic 2]\n  Difficulty Level: Level X\n\n"
        "Now, here are the paragraphs:\n"
    )

    for i, chunk in enumerate(chunks):
        prompt += f"Paragraph {i+1}:\n{chunk['text']}\n\n"

    try:
        response = model.generate_content(prompt)
        print("response",response.text)
        return parse_gemini_response(response.text)
    except Exception as e:
        print("❌ Gemini error:", e)
        return None

# --- Main Processing ---
for topic_key, content in knowledge_base.items():
    if topic_key not in progress:
        progress[topic_key] = 0

    chunks = content.get("chunks", [])
    total_chunks = len(chunks)

    while progress[topic_key] < total_chunks:
        batch = chunks[progress[topic_key]:progress[topic_key] + 3]
        print(f"🔁 Chunks {progress[topic_key]} to {progress[topic_key] + len(batch) - 1}")

        result = rephrase_and_classify_chunks(batch)
        print(result)
        if not result:
            print("⚠️ Skipping batch due to error.")
            continue

        for para in result:
            raw_text = para.get("text", "").strip()
            # if not raw_text or raw_text.lower().startswith("paragraph") or len(raw_text) < 25:
            #     continue
            if not raw_text:
                print("❌ Skipped due to empty text.")
                continue

            for item in para.get("topics", []):
                topic = item.get("topic", "").strip()
                subtopic = item.get("subtopic", "").strip()
                level_raw = item.get("level", "").strip().lower()

                if topic not in TOPIC_SUBTOPIC_MAP:
                    print(f"❌ Invalid topic: {topic}")
                    continue
                if subtopic not in TOPIC_SUBTOPIC_MAP[topic]:
                    print(f"❌ Invalid subtopic: {subtopic} under {topic}")
                    continue

                level = LEVEL_MAP.get(level_raw, "chunks_level2")

                # Initialize structure
                if topic not in rephrased_data:
                    rephrased_data[topic] = {}
                if subtopic not in rephrased_data[topic]:
                    rephrased_data[topic][subtopic] = {
                        "chunks_level1": [],
                        "chunks_level2": [],
                        "chunks_level3": []
                    }

                rephrased_data[topic][subtopic][level].append({"text": raw_text})

        # Update progress
        progress[topic_key] += len(batch)

        # Save progress and output
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(rephrased_data, f, indent=2)
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)

        print("✅ Batch saved. Waiting 45s...\n")
        time.sleep(45 + random.uniform(1, 5))

print("🎯 All paragraphs processed successfully.")
