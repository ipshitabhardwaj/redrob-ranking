from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONSULTING_FIRMS = [
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "mindtree", "mphasis", "hexaware", "tech mahindra",
    "hcl", "l&t infotech", "niit technologies"
]

PRODUCTION_ML_KEYWORDS = [
    "production", "deployed", "serving", "vector", "embedding",
    "retrieval", "ranking", "recommendation", "search", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "fine-tun", "mlops", "a/b test", "reranking", "llm", "rag",
    "real-time", "low latency", "scale", "pipeline"
]

RESEARCH_ONLY_KEYWORDS = [
    "phd", "research lab", "academic", "publication", "professor",
    "thesis", "arxiv", "iit research", "research intern"
]

HONEYPOT_JD_KEYWORDS = [
    "senior ai engineer", "founding team", "production ml",
    "vector database", "ranking", "retrieval", "recommendation",
    "llm", "rag", "mlops", "search", "reranking", "embedding"
]

BAD_TITLES = [
    "hr manager", "content writer", "accountant",
    "graphic designer", "sales executive", "marketing manager",
    "mechanical engineer", "civil engineer", "customer support"
]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def days_since(date_str):
    """Return number of days since a given date string."""
    if not date_str:
        return 999
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return (datetime.utcnow() - d).days
    except:
        return 999


def is_consulting_heavy(c):
    """
    Returns True if more than 60% of the candidate's career
    was spent at known consulting firms.
    """
    career = c.get("career_history", [])
    if not career:
        return False
    consulting_count = sum(
        1 for job in career
        if any(firm in (job.get("company") or "").lower()
               for firm in CONSULTING_FIRMS)
    )
    return consulting_count / len(career) > 0.6


def production_ml_score(c):
    """
    Score based on evidence of production ML experience
    found in job descriptions, titles, and summary.
    """
    text = ""
    for job in c.get("career_history", []):
        text += (job.get("description") or "").lower() + " "
        text += (job.get("title") or "").lower() + " "
    text += (c.get("profile", {}).get("summary") or "").lower()

    hits = sum(1 for kw in PRODUCTION_ML_KEYWORDS if kw in text)
    return min(hits / 6.0, 1.0)  # 6+ keyword hits = full score


def is_research_only(c):
    """
    Returns True if the candidate appears to be purely academic
    with no industry deployment experience.
    """
    text = ""
    for job in c.get("career_history", []):
        text += (job.get("description") or "").lower() + " "
        text += (job.get("title") or "").lower() + " "
    hits = sum(1 for kw in RESEARCH_ONLY_KEYWORDS if kw in text)
    return hits >= 2


def is_honeypot(c):
    """
    Detect honeypot candidates — profiles that are suspiciously
    stuffed with exact JD keywords but may not be genuine fits.
    """
    text = ""
    text += (c.get("profile", {}).get("summary") or "").lower() + " "
    text += (c.get("profile", {}).get("headline") or "").lower() + " "
    for job in c.get("career_history", []):
        text += (job.get("description") or "").lower() + " "

    hits = sum(1 for kw in HONEYPOT_JD_KEYWORDS if kw in text)
    return hits >= 6


# ---------------------------------------------------------------------------
# Main Scoring Function
# ---------------------------------------------------------------------------

def score_candidate(c, semantic_score, weights):
    """
    Compute a final relevance score for a candidate
    using a weighted combination of signals.
    """
    score = 0.0
    signals = c.get("redrob_signals", {})
    profile  = c.get("profile", {})

    # 1. Semantic similarity from FAISS retrieval
    score += semantic_score * weights["semantic"]

    # 2. Actively open to work
    if signals.get("open_to_work_flag"):
        score += weights["open_to_work"]

    # 3. Recency of last platform activity
    days_ago = days_since(signals.get("last_active_date"))
    if days_ago < 30:
        recency = 1.0
    elif days_ago < 60:
        recency = 0.7
    elif days_ago < 90:
        recency = 0.4
    elif days_ago < 180:
        recency = 0.2
    else:
        recency = 0.0
    score += recency * weights["recency"]

    # 4. Recruiter response rate — measures candidate responsiveness
    rr = signals.get("recruiter_response_rate", 0) or 0
    score += rr * weights["response_rate"]

    # 5. GitHub activity — proxy for hands-on technical work
    github = signals.get("github_activity_score", -1)
    if github is not None and github >= 0:
        score += (github / 100.0) * weights["github_activity"]

    # 6. Profile completeness
    completeness = signals.get("profile_completeness_score", 0) or 0
    score += (completeness / 100.0) * weights["profile_completeness"]

    # 7. Notice period — shorter is better for an urgent founding hire
    notice = signals.get("notice_period_days", 90) or 90
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.6
    elif notice <= 90:
        notice_score = 0.3
    else:
        notice_score = 0.0
    score += notice_score * weights["notice_period"]

    # 8. Production ML experience depth
    score += production_ml_score(c) * weights["production_ml"]

    # -----------------------------------------------------------------------
    # Penalties
    # -----------------------------------------------------------------------

    # Consulting-heavy background — poor fit for a founding product role
    if is_consulting_heavy(c):
        score -= 0.10

    # Pure research background — no production deployment evidence
    if is_research_only(c):
        score -= 0.08

    # Completely wrong domain (non-technical roles)
    current_title = (profile.get("current_title") or "").lower()
    if any(t in current_title for t in BAD_TITLES):
        score -= 0.20

    # Honeypot — abnormally high exact JD keyword density
    if is_honeypot(c):
        score -= 0.20

    return round(max(score, 0.0), 4)