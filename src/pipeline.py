import json
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config           import DATA_PATH, JD_PATH, OUTPUT_PATH, RETRIEVAL_K, WEIGHTS
from src.text_builder import build_candidate_text
from src.embedder     import (load_model, build_embeddings,
                               build_faiss_index, load_faiss_index)
from src.retriever    import read_jd, retrieve_top_k
from src.scorer       import score_candidate

EMBEDDINGS_PATH = "data/processed/embeddings.npy"
IDS_PATH        = "data/processed/candidate_ids.json"
INDEX_PATH      = "indexes/faiss_hnsw.index"


def load_candidates():
    print("Loading candidates...")
    candidates = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Total: {len(candidates)} candidates loaded")
    return candidates


def run_offline(candidates, model):
    """
    Run once to precompute embeddings and build the FAISS index.
    Results are saved to disk and reused on every subsequent run.
    """
    print("\n--- OFFLINE STEP START ---")

    texts = [build_candidate_text(c) for c in candidates]
    embeddings = build_embeddings(texts, model, EMBEDDINGS_PATH)

    os.makedirs(os.path.dirname(IDS_PATH), exist_ok=True)
    ids = [c["candidate_id"] for c in candidates]
    with open(IDS_PATH, "w") as f:
        json.dump(ids, f)
    print(f"Candidate IDs saved: {IDS_PATH}")

    build_faiss_index(embeddings, INDEX_PATH)
    print("--- OFFLINE STEP COMPLETE ---\n")


def run_online(candidates, model):
    """
    Timed ranking step — must complete within 5 minutes.
    Loads precomputed index, retrieves top-K, scores, saves top 100.
    """
    print("\n--- ONLINE STEP START ---")

    index = load_faiss_index(INDEX_PATH)
    with open(IDS_PATH) as f:
        candidate_ids = json.load(f)

    cand_map = {c["candidate_id"]: c for c in candidates}

    jd_text = read_jd(JD_PATH)
    top_ids, top_scores = retrieve_top_k(
        jd_text, model, index, candidate_ids, k=RETRIEVAL_K
    )

    print("Scoring candidates...")
    results = []
    for cid, sem_score in zip(top_ids, top_scores):
        c = cand_map.get(cid)
        if c:
            results.append({
                "candidate_id": cid,
                "score": score_candidate(c, sem_score, WEIGHTS)
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    top_100 = results[:100]

    os.makedirs("submission", exist_ok=True)
    df = pd.DataFrame(top_100)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Submission saved: {OUTPUT_PATH}")
    print(f"Scored {len(results)} candidates — top 100 written")
    print("--- ONLINE STEP COMPLETE ---\n")

    return top_100


def print_top_10(top_100, cand_map):
    print("\n========== TOP 10 CANDIDATES ==========\n")
    for i, r in enumerate(top_100[:10], 1):
        c = cand_map[r["candidate_id"]]
        p = c["profile"]
        signals = c.get("redrob_signals", {})
        print(f"Rank {i}")
        print(f"  Name     : {p.get('anonymized_name', 'N/A')}")
        print(f"  Title    : {p.get('current_title', 'N/A')} @ {p.get('current_company', 'N/A')}")
        print(f"  Exp      : {p.get('years_of_experience', 'N/A')} yrs")
        print(f"  Active   : {signals.get('last_active_date', 'N/A')}")
        print(f"  Open     : {signals.get('open_to_work_flag', False)}")
        print(f"  Score    : {r['score']}")
        print()


def main():
    start = time.time()

    candidates = load_candidates()
    model      = load_model()

    if not os.path.exists(EMBEDDINGS_PATH):
        run_offline(candidates, model)
    else:
        print("Embeddings already exist — skipping offline step")

    top_100  = run_online(candidates, model)
    cand_map = {c["candidate_id"]: c for c in candidates}
    print_top_10(top_100, cand_map)

    elapsed = time.time() - start
    print(f"Total runtime: {elapsed:.1f}s")
    if elapsed > 300:
        print("WARNING: 5-minute limit exceeded!")
    else:
        print(f"Within time limit — {300 - elapsed:.0f}s remaining")


if __name__ == "__main__":
    main()