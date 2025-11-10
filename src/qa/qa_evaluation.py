import os
import re
import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

REPORTS_DIR = "output/reports"
VISUALS_DIR = "output/visuals"
PROCESSED_DIR = "output/processed"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)

QA_PATHS = [
    os.path.join(REPORTS_DIR, "simulated_qa.txt"),
    os.path.join(REPORTS_DIR, "simulated_qa.py"),
]

TOPIC_FILE = os.path.join(REPORTS_DIR, "topics_lsa.txt")
REVIEW_SUMMARY_FILE = os.path.join(REPORTS_DIR, "review_summary.txt")
SENTIMENT_CSV = os.path.join(PROCESSED_DIR, "review_sentiment_classes.csv")

def read_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    raise FileNotFoundError(f"None of files exist: {paths}")

def parse_qa(text):
    pairs = re.findall(r"Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|$)", text, flags=re.DOTALL)
    return [(q.strip(), a.strip()) for q, a in pairs]

def parse_topics(topic_txt):
    topics = {}
    current = None
    for line in topic_txt.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"Topic\s*([0-9]+)\s*[:\-]?", line, flags=re.IGNORECASE)
        if m:
            current = int(m.group(1))
            topics[current] = set()
            continue
        if current is not None:
            toks = [t.strip().lower() for t in re.split(r"[,\t]+", line) if t.strip()]
            topics[current].update(toks)
    return topics

def extract_related_topics_from_answer(answer):
    m = re.search(r"Related topics\s*:\s*(.*)$", answer, flags=re.IGNORECASE)
    if m:
        raw = m.group(1)
    else:
        raw = answer
    toks = [t.strip().lower() for t in re.split(r"[,\n;]+", raw) if t.strip()]
    clean = []
    for t in toks:

        t2 = re.sub(r"[^\w\s\-]+", "", t).strip()
        if t2:
            words = [w for w in re.split(r"\s+", t2) if len(w) > 1]
            if len(words) > 1:
                clean += words
            else:
                clean.append(t2)
    return list(dict.fromkeys(clean)) 

def detect_sentiment_claim(answer):
    """
    Simple detection: returns 'positive', 'negative', 'neutral', or None
    """
    a = answer.lower()
    if "mostly positive" in a or "positive" in a and "negative" not in a:
        return "positive"
    if "mostly negative" in a or "negative" in a and "positive" not in a:
        return "negative"
    if "mixed" in a or "neutral" in a:
        return "neutral"
    return None

qa_text = read_first_existing(QA_PATHS)
qa_pairs = parse_qa(qa_text)
if not qa_pairs:
    raise RuntimeError("No Q/A pairs parsed from the QA file.")

topic_text = ""
if os.path.exists(TOPIC_FILE):
    with open(TOPIC_FILE, "r", encoding="utf-8") as f:
        topic_text = f.read()
else:
    raise FileNotFoundError(f"Missing topic file: {TOPIC_FILE}")

topics = parse_topics(topic_text)
if not topics:
    raise RuntimeError("No topics parsed from topic file.")

if os.path.exists(SENTIMENT_CSV):
    sentiment_df = pd.read_csv(SENTIMENT_CSV)
else:
    raise FileNotFoundError(f"Missing sentiment CSV: {SENTIMENT_CSV}")

pos_mask = sentiment_df["sentiment_class"].str.contains("Positive", case=False, na=False)
neg_mask = sentiment_df["sentiment_class"].str.contains("Negative", case=False, na=False)
neutral_mask = sentiment_df["sentiment_class"].str.contains("Neutral", case=False, na=False)

overall_pos_ratio = pos_mask.sum() / max(1, len(sentiment_df))
overall_neg_ratio = neg_mask.sum() / max(1, len(sentiment_df))
overall_neu_ratio = neutral_mask.sum() / max(1, len(sentiment_df))

# evaluation 
rows = []
topic_ids = sorted(topics.keys())
heat = np.zeros((len(qa_pairs), len(topic_ids)))  

for qi, (question, answer) in enumerate(qa_pairs):
    qa_keywords = extract_related_topics_from_answer(answer)
    qa_keywords_set = set(qa_keywords)

    # Coverage: how many QA keywords appear in any topic keywords
    all_topic_keywords = set().union(*topics.values())
    coverage_count = sum(1 for k in qa_keywords_set if k in all_topic_keywords)
    coverage_score = coverage_count / max(1, len(qa_keywords_set))

    # Topic match: for each topic, compute overlap ratio ( |qa_keywords ∩ topic_keywords| / |topic_keywords| )
    topic_match_scores = []
    for tidx, t in enumerate(topic_ids):
        topk = topics[t]
        inter = qa_keywords_set.intersection(topk)
        heat[qi, tidx] = len(inter)
        match_ratio = (len(inter) / max(1, len(topk)) + len(inter) / max(1, len(qa_keywords_set))) / 2.0
        topic_match_scores.append(match_ratio)

    # Best matching topic
    best_idx = int(np.argmax(topic_match_scores))
    best_topic = topic_ids[best_idx]
    best_match_score = float(topic_match_scores[best_idx])

    # Sentiment claim
    claimed = detect_sentiment_claim(answer)
    # Compute alignment: compare claimed vs overall sentiment distribution
    sas = None
    verdict = "Undetermined"
    if claimed == "positive":
        sas = overall_pos_ratio
        verdict = "Aligned" if sas >= 0.6 else "Partially aligned" if sas >= 0.4 else "Not aligned"
    elif claimed == "negative":
        sas = overall_neg_ratio
        verdict = "Aligned" if sas >= 0.6 else "Partially aligned" if sas >= 0.4 else "Not aligned"
    elif claimed == "neutral":
        sas = overall_neu_ratio
        verdict = "Aligned" if sas >= 0.4 else "Partially aligned" if sas >= 0.25 else "Not aligned"
    else:
        # if no explicit claim, use best topic sentiment heuristic (not available), fallback to global
        sas = overall_pos_ratio
        verdict = "No explicit claim; compared to global positive ratio"

    # Compose short explanation
    matching_keywords = list({k for k in qa_keywords_set if k in topics[best_topic]})
    explanation = (
        f"Best topic: {best_topic} (match {best_match_score:.2f}). "
        f"QA keywords matched in topic: {matching_keywords}. "
        f"Coverage: {coverage_score:.2f}. "
        f"Sentiment claim: {claimed or 'none'}; global pos_ratio={overall_pos_ratio:.2f}."
    )

    rows.append({
        "question": question,
        "answer_summary": (answer[:300] + "...") if len(answer) > 300 else answer,
        "qa_keywords": ",".join(qa_keywords),
        "best_topic": best_topic,
        "best_topic_match_score": round(best_match_score, 3),
        "coverage_score": round(coverage_score, 3),
        "sentiment_claim": claimed or "",
        "sentiment_alignment_value": round(sas, 3),
        "verdict": verdict,
        "explanation": explanation
    })

# saving to CSV
csv_out = os.path.join(REPORTS_DIR, "qa_evidence_scores.csv")
pd.DataFrame(rows).to_csv(csv_out, index=False, encoding="utf-8")
print("Saved numeric QA evidence scores to", csv_out)

# save human readable report 
txt_out = os.path.join(REPORTS_DIR, "qa_evidence_evaluation.txt")
with open(txt_out, "w", encoding="utf-8") as f:
    f.write("QA Evidence Evaluation Report\n")
    f.write("______________________________\n\n")
    f.write(f"Overall sentiment ratios: positive={overall_pos_ratio:.2f}, negative={overall_neg_ratio:.2f}, neutral={overall_neu_ratio:.2f}\n\n")
    for r in rows:
        f.write(f"Q: {r['question']}\n")
        f.write(f"A (summary): {r['answer_summary']}\n")
        f.write(f"- QA keywords: {r['qa_keywords']}\n")
        f.write(f"- Best matching topic: {r['best_topic']} (score={r['best_topic_match_score']})\n")
        f.write(f"- Coverage score: {r['coverage_score']}\n")
        f.write(f"- Sentiment claim: {r['sentiment_claim']} (alignment value={r['sentiment_alignment_value']})\n")
        f.write(f"- Verdict: {r['verdict']}\n")
        f.write(f"- Explanation: {r['explanation']}\n\n")

print("Saved human-readable evaluation to", txt_out)

# heatmap 
plt.figure(figsize=(max(6, len(qa_pairs)*1.2), max(4, len(topic_ids)*0.6)))
sns.heatmap(heat, annot=True, fmt=".0f", cmap="Blues",
            yticklabels=[f"Q{i+1}" for i in range(len(qa_pairs))],
            xticklabels=[f"T{t}" for t in topic_ids])
plt.xlabel("Topics")
plt.ylabel("QA (rows)")
plt.title("QA keywords matched (count) per Topic")
plt.tight_layout()
heat_out = os.path.join(VISUALS_DIR, "qa_topic_match_heatmap.png")
plt.savefig(heat_out)
plt.close()
print("Saved heatmap to", heat_out)

print("\nAll outputs written to the 'output' folder:")
print(" -", csv_out)
print(" -", txt_out)
print(" -", heat_out)
