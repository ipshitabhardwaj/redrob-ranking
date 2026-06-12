import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os

# Monkeypatch requests to bypass SSL verify
try:
    import requests
    original_send = requests.Session.send
    def patched_send(self, request, **kwargs):
        kwargs['verify'] = False
        return original_send(self, request, **kwargs)
    requests.Session.send = patched_send
except ImportError:
    pass

# Monkeypatch httpx to bypass SSL verify
try:
    import httpx
    original_init = httpx.Client.__init__
    def patched_init(self, *args, **kwargs):
        kwargs['verify'] = False
        original_init(self, *args, **kwargs)
    httpx.Client.__init__ = patched_init
    
    original_async_init = httpx.AsyncClient.__init__
    def patched_async_init(self, *args, **kwargs):
        kwargs['verify'] = False
        original_async_init(self, *args, **kwargs)
    httpx.AsyncClient.__init__ = patched_async_init
except ImportError:
    pass

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

MODEL_NAME = "all-MiniLM-L6-v2"


def load_model():
    print("Loading embedding model...")
    import torch
    torch.set_num_threads(1)
    return SentenceTransformer(MODEL_NAME)


def build_embeddings(texts, model,
                     save_path="data/processed/embeddings.npy"):
    print(f"Building embeddings for {len(texts)} candidates...")
    embeddings = model.encode(
        texts,
        batch_size=32,
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
    print("Building FAISS Flat IP index...")
    emb = embeddings.copy()
    faiss.normalize_L2(emb)

    dimension = emb.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(emb)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    faiss.write_index(index, save_path)
    print(f"FAISS index saved: {save_path}")
    return index


def load_faiss_index(path="indexes/faiss_hnsw.index"):
    print(f"Loading FAISS index from: {path}")
    return faiss.read_index(path)