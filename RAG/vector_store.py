from qdrant_client        import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
import numpy as np

COLLECTION_NAME = "mental_health_kb"
VECTOR_DIM      = 384
BATCH_SIZE      = 200

# =====================================================================================================

def get_client(url: str, api_key: str) -> QdrantClient:
    return QdrantClient(url=url, api_key=api_key)

# =====================================================================================================

def create_collection(client: QdrantClient):
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        print("Collection created")
    except Exception as e:
        print(f"Already exists or error: {e}")

# =====================================================================================================

def upload_records(client: QdrantClient, records: list[dict], embeddings: np.ndarray):

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "id"      : records[i]["id"],
                "context" : records[i]["context"],
                "response": records[i]["response"],
            }
        )
        for i in range(len(records))
    ]

    for i in tqdm(range(0, len(points), BATCH_SIZE), desc="Uploading"):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i : i + BATCH_SIZE])

    print(f"Uploaded {len(points)} records to Qdrant")

# =====================================================================================================

def retrieve(query_vec: list, client: QdrantClient, top_k: int = 5) -> list[dict]:

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        with_payload=True
    )

    retrieved = []
    for point in results.points:
        retrieved.append({
            "score"           : round(point.score, 4),
            "similar_question": point.payload["context"],
            "counselor_answer": point.payload["response"],
        })

    return retrieved