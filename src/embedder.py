from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

MODEL_NAME = "all-MiniLM-L6-v2"


def load_model():
    print("Loading embedding model...")
    return SentenceTransformer(MODEL_NAME)


def build_embeddings(texts, model,
                     save_path="data/processed/embeddings.npy"):
    print(f"Building embeddings for {len(texts)} candidates...")
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, embeddings)
    print(f"Embeddings saved: {save_path}")
    return embeddings


def load_embeddings(path="data/processed/embeddings.npy"):
    print(f"Loading embeddings from: {path}")
    return np.load(path).astype("float32")


def build_faiss_index(embeddings,
                      save_path="indexes/faiss_hnsw.index"):
    print("Building FAISS HNSW index...")
    emb = embeddings.copy()
    faiss.normalize_L2(emb)

    dimension = emb.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.add(emb)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    faiss.write_index(index, save_path)
    print(f"FAISS index saved: {save_path}")
    return index


def load_faiss_index(path="indexes/faiss_hnsw.index"):
    print(f"Loading FAISS index from: {path}")
    return faiss.read_index(path)