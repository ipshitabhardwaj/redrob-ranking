# Architectural Decisions Log

This document logs all key architectural decisions made by team Data Dames.

## Decisions Log

### 1. Two-Stage Retrieval and Reranking Architecture
*   **Context:** The candidate pool contains 100,000 profiles. The online ranking task must finish within a 5-minute wall-clock limit on a CPU-only environment. Simple keyword matching fails due to honeypots and keyword-stuffers, while full-profile LLM parsing is too slow for CPU inference.
*   **Decision:** Implement a two-stage search architecture. Stage 1 utilizes a lightweight, dense retrieval index (FAISS HNSW) to quickly extract the top 500 candidate IDs. Stage 2 runs a Python-based multi-signal scoring and heuristic ranking engine over the retrieved subset.
*   **Consequence:** High recall and blazing-fast online latency. Retrieval finishes in less than 30 seconds, and Stage 2 reranking completes in less than 0.1 seconds, easily satisfying the 5-minute compute constraints.

### 2. Concise Text Representation for Embedding Generation
*   **Context:** Early benchmarks of embedding generation for 100K profiles using full summaries and complete career histories took over 4 hours on a 2-core CPU. This was due to sequence length padding overhead within batches.
*   **Decision:** Build an ultra-concise text representation containing only the current title, professional headline, and top 5 skills. Move all other details (detailed job descriptions, dates, locations, notice periods) to the Stage 2 python scorer.
*   **Consequence:** Reduced sequence lengths to under 60-100 words, eliminating padding overhead. The offline embedding step time dropped from 4 hours to under 45 minutes on a 2-core CPU, while preserving Stage 1 retrieval recall.

### 3. Strict Boolean Filtering for Honeypot Candidates
*   **Context:** The starter code used a fuzzy keyword-density threshold to penalize honeypots. This erroneously penalized 456 valid candidates who legitimately used ML keywords in their summaries. The dataset has exactly 70 true honeypot candidates.
*   **Decision:** Replace the keyword penalty with a strict Boolean check detecting logical contradictions:
    1. Years of experience discrepancy (profile experience differs from career history sum by >= 1.0 year).
    2. Skill inflation (expert proficiency listed for skills with 0 months duration).
*   **Consequence:** Identifies and filters out exactly the 70 true honeypot candidates. Guarantees a 0% honeypot rate in the final top 100 list, preventing auto-disqualification.

### 4. PyTorch Thread Optimization
*   **Context:** Testing on a 2-core CPU showed PyTorch's default thread pool setup led to OMP/OpenMP thread contention and thrashing, reducing candidate processing speeds.
*   **Decision:** Force PyTorch thread pool size to 1 using `torch.set_num_threads(1)` and use a smaller batch size of 32 in the model configuration.
*   **Consequence:** Removed thread synchronization overhead, doubling candidate embedding speed from 19.1 to 35.5 candidates/sec.