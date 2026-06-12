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
        # Reference date set to June 12, 2026 to align with dataset timeline
        d = datetime.strptime(date_str[:10], "%Y-%m-%d")
        ref = datetime(2026, 6, 12)
        return (ref - d).days
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
    Detect honeypot candidates — synthetic profiles with logical timeline
    contradictions or impossible skill duration claims.
    """
    prof = c.get("profile", {})
    career = c.get("career_history", [])
    skills = c.get("skills", [])
    
    # 1. Total YoE mismatch
    yoe = prof.get("years_of_experience", 0)
    calculated_yoe = sum(job.get("duration_months", 0) for job in career) / 12.0
    if abs(yoe - calculated_yoe) >= 1.0:
        return True
        
    # 2. Expert skills with 0 months duration
    expert_zero = sum(1 for s in skills if s.get("proficiency") == "expert" and s.get("duration_months") == 0)
    if expert_zero >= 1:
        return True
        
    return False


def is_bad_title(c):
    """Filter out candidates in completely wrong domains (non-technical)."""
    current_title = (c.get("profile", {}).get("current_title") or "").lower()
    return any(t in current_title for t in BAD_TITLES)


def location_score(c):
    """ Noida/Pune target locations get 1.0. Tier-1 Indian relocation cities get 0.7 if willing to relocate. """
    prof = c.get("profile", {})
    loc = (prof.get("location") or "").lower()
    willing_to_relocate = c.get("redrob_signals", {}).get("willing_to_relocate", False)
    
    if "noida" in loc or "pune" in loc:
        return 1.0
        
    tier1_cities = ["bangalore", "bengaluru", "hyderabad", "mumbai", "delhi", "ncr", "gurgaon", "chennai"]
    is_tier1 = any(city in loc for city in tier1_cities)
    
    if is_tier1 and willing_to_relocate:
        return 0.7
    elif is_tier1:
        return 0.3
    return 0.0


def startup_fit_score(c):
    """ Found founding or early stage experience in product companies. """
    score = 0.0
    career = c.get("career_history", [])
    if not career:
        return 0.0
        
    startup_sizes = ["1-10", "11-50", "51-200"]
    startup_jobs = 0
    founding_title = False
    
    for job in career:
        if job.get("company_size") in startup_sizes:
            startup_jobs += 1
        title = (job.get("title") or "").lower()
        if any(kw in title for kw in ["founding", "first", "co-founder", "lead engineer", "founding engineer"]):
            founding_title = True
            
    score += min(startup_jobs / 2.0, 1.0) * 0.6
    if founding_title:
        score += 0.4
        
    return score

# ---------------------------------------------------------------------------
# Main Scoring Function
# ---------------------------------------------------------------------------

def score_candidate(c, semantic_score, weights):
    """
    Compute a final relevance score for a candidate
    using a weighted combination of signals.
    """
    # Quick filters
    if is_honeypot(c) or is_bad_title(c):
        return 0.0
        
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

    # 9. Location fit
    score += location_score(c) * weights["location"]

    # 10. Startup fit
    score += startup_fit_score(c) * weights["startup_fit"]

    # -----------------------------------------------------------------------
    # Penalties
    # -----------------------------------------------------------------------

    # Consulting-heavy background — poor fit for a founding product role
    if is_consulting_heavy(c):
        score -= 0.10

    # Pure research background — no production deployment evidence
    if is_research_only(c):
        score -= 0.08

    # Years of experience constraints (should ideally be 5-9 years as per JD)
    yoe = profile.get("years_of_experience", 0)
    if yoe < 4.0:
        score -= 0.15
    elif yoe > 12.0:
        score -= 0.05

    return round(max(score, 0.0), 4)