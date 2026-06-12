# Scoring Philosophy

This document outlines the candidate evaluation heuristics and scoring philosophy applied by team Data Dames.

## Target Profile Justification

The goal of our ranking system is to align with the "Senior AI Engineer — Founding Team" job description. The ideal candidate profile must have:
1. Deep technical expertise in production ML, NLP, ranking, and search systems.
2. An early-stage product development mindset ("shipper" over "academic researcher").
3. Immediate availability and local/geographic alignment.

We implemented these requirements by weighting the following factors:

### 1. Semantic Relevance (30%)
*   **Source:** Dense retrieval cosine similarity between the job description and the candidate's core profile (current title, headline, and top 5 skills).
*   **Justification:** Verifies basic alignment on terminology and core capabilities (NLP, search, retrieval, ML engineering).

### 2. Location Fit (5%)
*   **Source:** Candidate `location` and `willing_to_relocate` flag.
*   **Justification:** Noida and Pune are target office cities. Candidates based in Pune/Noida receive +1.0. Candidates in other Tier-1 Indian relocation cities (such as Bangalore, Delhi NCR, Hyderabad, Mumbai, Chennai) receive +0.7 if they are willing to relocate, and +0.3 if they are not. Others receive 0.0.

### 3. Notice Period (7%)
*   **Source:** Stated `notice_period_days`.
*   **Justification:** Urgent founding hires require short notice. Notice periods of <= 30 days receive full score (+1.0). Notice periods of 30-60 days receive partial score (+0.6). Notice periods of 60-90 days receive +0.3. Notice periods > 90 days receive 0.0.

### 4. Startup and Founding Fit (5%)
*   **Source:** Company sizes in `career_history` (e.g., "1-10", "11-50", "51-200") and founding keywords ("founding", "first", "co-founder", "lead engineer").
*   **Justification:** Early-stage startups need candidates comfortable with ownership and execution. Candidates with startup size history get up to +0.6. Those with founding titles get an additional +0.4 boost.

### 5. Production ML Experience Depth (10%)
*   **Source:** Heavy scanning for engineering keywords ("serving", "deployment", "mlops", "ndcg", "reranking", "faiss", "vector database", "elasticsearch") in candidate career descriptions.
*   **Justification:** The JD explicitly filters out candidates whose "AI experience" consists solely of LangChain tutorials or API calling. We search for systemic, operations-level experience.

### 6. Platform Engagement (33%)
*   **Source:** `recruiter_response_rate` (10%), `open_to_work_flag` (10%), `last_active_date` recency (10%), and `profile_completeness_score` (5%).
*   **Justification:** A perfect candidate who is inactive on the platform and has a low response rate is practically unreachable. Engagement signals act as a crucial multiplier.

---

## Penalties and Hard Filters

To ensure the top candidates are genuine fits, we implement the following filters:

### 1. Honeypots (Hard Filter)
*   Candidates matching the YoE discrepancy or the expert skill inflation checks are immediately disqualified (score set to `0.0`).

### 2. Bad Titles (Hard Filter)
*   Candidates holding current non-technical titles (e.g. HR manager, accountant, content writer) are immediately disqualified (score set to `0.0`).

### 3. Consulting-Heavy Background (-10% Penalty)
*   Candidates who spent more than 60% of their career at services/consulting firms (e.g., TCS, Wipro, Infosys, Accenture, Cognizant) receive a score deduction.

### 4. Pure Research Background (-8% Penalty)
*   Candidates with academic-only publications and thesis experience without production deployment evidence receive a score deduction.

### 5. Experience Mismatch (-5% to -15% Penalty)
*   The target range is 5-9 years. Candidates with < 4 years receive a heavy penalty (-15%). Candidates with > 12 years receive a soft penalty (-5%).
