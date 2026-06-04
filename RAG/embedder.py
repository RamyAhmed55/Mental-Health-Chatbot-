import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDINGS_FILE = os.path.join(os.path.dirname(__file__), "embeddings_q_only.pkl")
MODEL_NAME      = "all-MiniLM-L6-v2"

# =====================================================================================================

def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)

# =====================================================================================================

def load_embeddings(records: list[dict], embedder: SentenceTransformer) -> np.ndarray:

    if os.path.exists(EMBEDDINGS_FILE):
        print("Loading embeddings from file...")
        with open(EMBEDDINGS_FILE, "rb") as f:
            embeddings = pickle.load(f)
        print(f"Loaded: {embeddings.shape}")
        return embeddings

    print("Embedding questions only (one-time)...")
    questions  = [r["context"] for r in records]
    embeddings = embedder.encode(
        questions,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Embedded & saved: {embeddings.shape}")
    return embeddings

# =====================================================================================================

def embed_query(query: str, embedder: SentenceTransformer) -> list:
    return embedder.encode( query, normalize_embeddings=True ).tolist()