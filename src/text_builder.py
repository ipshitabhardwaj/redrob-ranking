def build_candidate_text(c):
    """
    Build an ultra-concise text representation for fast CPU embedding.
    Fits well within the 256-token limit of all-MiniLM-L6-v2, allowing fast indexing on CPU.
    """
    p = c.get("profile", {})
    parts = []

    if p.get("current_title"):
        parts.append(p["current_title"])
    if p.get("headline"):
        parts.append(p["headline"])
        
    # Append top 5 skill names
    skills = c.get("skills", [])
    skill_names = [s["name"] for s in skills if s.get("name")]
    if skill_names:
        parts.append(", ".join(skill_names[:5]))
        
    return " ".join([x for x in parts if x]).strip()