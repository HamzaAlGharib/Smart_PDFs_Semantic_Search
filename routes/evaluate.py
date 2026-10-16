import os
import re
import json

from fastapi import APIRouter
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)

from utils.embeddings import generate_embeddings
from utils.keywords import extract_keywords_per_file
import utils.vector_store as vs  # module import so we always see current state

router = APIRouter()

KEYWORDS_PATH = os.path.join(vs.STORE_FOLDER, "keywords.json")


@router.get("/evaluate")
async def evaluate():
    # Nothing to evaluate if no PDFs are indexed.
    if vs.total_vectors() == 0:
        return {"message": "No PDFs indexed yet. Please upload PDFs first."}

    # 1. Rebuild each file's full text from its chunks in metadata.
    file_texts = {}
    for item in vs.metadata:
        file_texts.setdefault(item["filename"], []).append(item["text"])
    file_texts = {name: " ".join(parts) for name, parts in file_texts.items()}

    # 2. Extract the top keywords per file (TF-IDF) and save them to disk.
    keywords_per_file = extract_keywords_per_file(file_texts, top_n=3)
    with open(KEYWORDS_PATH, "w", encoding="utf-8") as f:
        json.dump(keywords_per_file, f, indent=2)

    # Use each unique keyword as a test query (dedup so none is counted twice).
    queries = sorted({kw for kws in keywords_per_file.values() for kw in kws})
    if not queries:
        return {"message": "Could not extract keywords (documents too short)."}

    # 3. Build the three aligned lists over every (keyword, chunk) pair.
    #    y_true : does the chunk actually contain the keyword? (ground truth)
    #    y_score: the similarity our search gives that chunk (prediction score)
    #    y_pred : did the score pass the threshold? (yes/no prediction)
    y_true, y_pred, y_score = [], [], []
    threshold = vs.SIMILARITY_THRESHOLD

    for keyword in queries:
        scores = vs.score_all_chunks(generate_embeddings([keyword]))
        pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)

        for i, item in enumerate(vs.metadata):
            y_true.append(1 if pattern.search(item["text"]) else 0)
            y_score.append(scores[i])
            y_pred.append(1 if scores[i] >= threshold else 0)

    # 4. Let scikit-learn compute everything from the two lists.
    #    labels=[1, 0] forces a 2x2 matrix laid out as:
    #        [[TP, FN],
    #         [FP, TN]]
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn = int(cm[0][0]), int(cm[0][1])
    fp, tn = int(cm[1][0]), int(cm[1][1])

    accuracy = accuracy_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUC needs both classes present in y_true; guard against the edge case.
    if len(set(y_true)) > 1:
        auc = round(float(roc_auc_score(y_true, y_score)), 3)
    else:
        auc = None

    return {
        "keywords": keywords_per_file,
        "num_queries": len(queries),
        "num_chunks": vs.total_vectors(),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "metrics": {
            "accuracy": round(float(accuracy), 3),
            "specificity": round(float(specificity), 3),
            "auc": auc,
        },
    }
