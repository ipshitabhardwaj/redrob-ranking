def build_candidate_text(c):
    """
    Build a concise single text string per candidate for embedding.
    Fits well within the 256-token limit of all-MiniLM-L6-v2 to speed up CPU encoding.
    """
    p = c.get("profile", {})
    parts = []

    if p.get("current_title"):
        parts.append(p["current_title"])
    if p.get("headline"):
        parts.append(p["headline"])
    if p.get("summary"):
        # Keep summary short
        parts.append(p["summary"][:200])

    # Career history (titles and companies of last 3 jobs, descriptions truncated)
    career = c.get("career_history", [])
    for job in career[:3]:
        job_parts = []
        if job.get("title"):
            job_parts.append(job["title"])
        if job.get("company"):
            job_parts.append(f"at {job['company']}")
        if job.get("description"):
            # Only include the first 80 characters of the description for keywords
            job_parts.append(job["description"][:80])
        parts.append(" ".join(job_parts))

    # All skill names (just names to keep length short)
    skills = c.get("skills", [])
    skill_names = [s["name"] for s in skills if s.get("name")]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))

    # Education degree and field of study (no institution name to save space/noise)
    education = c.get("education", [])
    edu_strings = []
    for edu in education:
        deg = edu.get("degree")
        field = edu.get("field_of_study")
        if deg and field:
            edu_strings.append(f"{deg} in {field}")
    if edu_strings:
        parts.append("Education: " + ", ".join(edu_strings))

    return " ".join([x for x in parts if x]).strip()