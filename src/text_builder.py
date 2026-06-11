def build_candidate_text(c):
    """
    Build a single text string per candidate for embedding.
    Order matters — most recent and relevant experience comes first.
    """
    p = c.get("profile", {})
    parts = []

    if p.get("current_title"):
        parts.append(p["current_title"])
    if p.get("headline"):
        parts.append(p["headline"])
    if p.get("summary"):
        parts.append(p["summary"])

    # Include descriptions from the 2 most recent jobs
    career = c.get("career_history", [])
    for job in career[:2]:
        if job.get("title"):
            parts.append(job["title"])
        if job.get("description"):
            parts.append(job["description"])

    # Append all skill names
    skills = c.get("skills", [])
    skill_names = [s["name"] for s in skills if s.get("name")]
    if skill_names:
        parts.append(" ".join(skill_names))

    return " ".join([x for x in parts if x]).strip()