# Pipeline Architecture

This document describes the design and flow of the two-stage candidate retrieval and ranking pipeline developed by team Data Dames.

## Pipeline Architecture Diagram

```
+-------------------------------------------------------------------+
|                           OFFLINE STAGE                           |
|                                                                   |
| 1. Profile Representation:                                        |
|    Build concise text representation from current_title, headline,|
|    and top 5 skills.                                              |
| 2. Dense Vector Encoding:                                         |
|    Generate 384-d embeddings via all-MiniLM-L6-v2 on CPU.         |
| 3. HNSW Flat Indexing:                                            |
|    Construct and save FAISS IndexHNSWFlat to disk.                 |
+-------------------------------------------------------------------+
                                  |
                                  | Reused during online run
                                  v
+-------------------------------------------------------------------+
|                        STAGE 1: RETRIEVAL                         |
|                                                                   |
| 1. Embed Job Description:                                         |
|    Load query and embed JD text using the same model.             |
| 2. FAISS HNSW Semantic Search:                                    |
|    Query index to retrieve top 500 semantically matching candidate|
|    IDs and their similarity scores.                               |
+-------------------------------------------------------------------+
                                  |
                                  | 500 candidate IDs + similarity scores
                                  v
+-------------------------------------------------------------------+
|                        STAGE 2: RERANKING                         |
|                                                                   |
| 1. Hard Logic Filter:                                             |
|    - Remove non-technical BAD_TITLES.                             |
|    - Remove 70 mathematically proven honeypots.                   |
| 2. Multi-Signal Scoring:                                          |
|    - Compute weights (Semantic, Location, Notice Period, Startup  |
|      Fit, Production ML depth, Platform Engagement).              |
|    - Apply soft penalties (Consulting, Research, YoE).            |
| 3. Deterministic Sorting:                                         |
|    - Sort by Score Descending, then by Candidate ID Ascending.    |
| 4. Reasoning Generation:                                          |
|    - Generate factual 1-2 sentence justifications.                 |
| 5. Output Shortlist:                                              |
|    - Save top 100 rows to team_submission.csv.                    |
+-------------------------------------------------------------------+
```

## Detailed Stage Flow

### Stage 1: Dense Retrieval
The dense retrieval stage is executed once offline to build the search space. It operates as follows:
*   **Concise Tokenization:** Candidate profile strings are kept under 60-100 words. This ensures batch sequence lengths are short, minimizing CPU batch padding and optimizing encoding throughput.
*   **HNSW Index Construction:** Embeddings are L2-normalized and indexed in a FAISS HNSW (Hierarchical Navigable Small World) index. During online execution, this structure allows sub-linear, high-recall similarity searches.
*   **JD Embedding:** The job description text is read, embedded using the same SentenceTransformer instance, L2-normalized, and queried against the HNSW index to retrieve the top 500 candidate matches.

### Stage 2: Heuristic Reranking and Verification
The top 500 candidates undergo detailed python-level heuristic scoring:
*   **Honeypot Validation:** We check for Logical Contradictions:
    1.  `yoe_diff = abs(profile_yoe - calculated_yoe) >= 1.0`
    2.  `expert_zero = (expert skills with 0 months experience) >= 1`
    If a candidate matches either rule, their score is set to `0.0` and they are excluded.
*   **Location, Availability, and Experience Alignment:** Reranking scores are compiled based on candidate location, notice period length, startup experience, and hands-on production ML keywords.
*   **Sorting & Formatting:** Scored candidates are sorted. Ranks 1 to 100 are assigned. Standard string reasoning templates are constructed dynamically, substituting the candidate's exact years of experience, current title, named skills, relocation statuses, and notice period values.
