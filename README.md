# Intelligent Candidate Discovery and Ranking Engine

This repository contains the implementation of the candidate discovery and ranking system developed by team Data Dames for the Intelligent Candidate Discovery & Ranking Challenge.

## Team Identity

Team Name: Data Dames
Team Members:
*   PLACEHOLDER_MEMBER_1_NAME (ML Engineer)
*   PLACEHOLDER_MEMBER_2_NAME (Data Lead)
*   PLACEHOLDER_MEMBER_3_NAME (Evaluation Lead)
*   PLACEHOLDER_MEMBER_4_NAME (Systems Lead)

## System Architecture

The ranking engine is designed as a high-performance, two-stage retrieval and ranking pipeline optimized for CPU execution.

1. Stage 1: Retrieval (FAISS Dense HNSW Index)
   * A candidate profile representation is constructed from the candidate's current title, headline, summary, most recent job description, and top skills.
   * Profiles are encoded into 384-dimensional dense vectors using the all-MiniLM-L6-v2 SentenceTransformer.
   * A FAISS IndexHNSWFlat structure is built offline to retrieve the top 500 semantically matching candidates under 30 seconds during online execution.

2. Stage 2: Reranking (Multi-Signal Scorer)
   * Non-technical titles are immediately filtered out.
   * Discovered honeypot candidates (70 profiles with logical timeline discrepancies or expert skill duration inflation) are scrubbed.
   * The remaining candidates are scored using a weighted combination of:
     * Semantic relevance score (30%)
     * Platform response rate and open-to-work flag (20%)
     * Active recency (10%)
     * GitHub activity (8%)
     * Profile completeness (5%)
     * Notice period length (7%)
     * Production ML experience depth (10%)
     * Noida/Pune location fit (5%)
     * Startup and founding experience (5%)

3. Deterministic Sorting:
   * Candidates are sorted by final score in descending order.
   * In case of score ties, candidate IDs are sorted alphabetically in ascending order.
   * Factual, custom reasonings are generated for the top 100 candidates.

## Installation and Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place candidate data files under the data directory:
   * data/candidates.jsonl
   * data/job_description.docx

## How to Run

1. Run the pipeline:
   ```bash
   python -m src.pipeline
   ```
   * The pipeline will automatically run the offline step (generating embeddings and the FAISS HNSW index) on the first run, and then perform the online retrieval and ranking step.
   * The final output is written to: submission/team_submission.csv

2. Validate the output format:
   ```bash
   python validate_submission.py submission/team_submission.csv
   ```