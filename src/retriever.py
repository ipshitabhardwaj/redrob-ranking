import faiss
import numpy as np
from docx import Document


def read_jd(jd_path):
    """Extract full text from the job description .docx file."""
    doc = Document(jd_path)
    text = "\n".join([
        para.text for para in doc.paragraphs
        if para.text.strip()
    ])
    print(f"JD loaded — {len(text)} characters")
    return text


def retrieve_top_k(jd_text, model, index, candidate_ids, k=500):
    """
    Embed the JD and retrieve the top-k most similar
    candidates from the FAISS index.
    """
    print(f"Retrieving top {k} candidates...")

    jd_vec = model.encode(
        [jd_text],
        convert_to_numpy=True
    ).astype("float32")
    faiss.normalize_L2(jd_vec)

    distances, indices = index.search(jd_vec, k=k)

    top_ids    = [candidate_ids[i] for i in indices[0]]
    top_scores = distances[0].tolist()

    print(f"Retrieved {len(top_ids)} candidates")
    return top_ids, top_scores