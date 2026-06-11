# Decisions Log

## Day 1 — Architecture Choice
**Decision:** Two-stage pipeline (FAISS retrieval → multi-signal reranker)
**Why:** JD explicitly says keyword matching is a trap. Sample submission confirms
HR Managers ranked #1 with keyword matching — that's the wrong approach.
Behavioral signals + career trajectory required.