from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client import models
from pymongo import MongoClient
import uuid
from qdrant_client.models import VectorParams, Distance


# Source data: raw scraped/ingested documents live in MongoDB
mongo_client = MongoClient('mongodb://llm_engineering:llm_engineering@127.0.0.1:27018/?authSource=admin')
db = mongo_client['rag_system']
collection = db['raw_data']

# Destination: vector embeddings are stored in Qdrant for similarity search
qdrant = qdrant_client.QdrantClient("http://127.0.0.1:6333")


# Embedding model used to turn document text into vectors
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

collection_name = "raw_data"
collections_response = qdrant.get_collections()

# Create the Qdrant collection on first run only
if not any(col.name == collection_name for col in collections_response.collections):

    qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=768,   # output dimension of all-mpnet-base-v2
        distance=Distance.COSINE
    )
)
# Split long documents into overlapping word windows so each chunk fits the
# embedding model's input limit, while the overlap preserves context across chunk boundaries
def chunk_text(text, chunk_size=512, overlap=64):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # overlap with previous chunk
    return chunks

def extract_features(text):
    """Encode text into an embedding vector using the SentenceTransformer model."""
    
    return model.encode(text, convert_to_numpy=True)

def push_features_to_qdrant():
    """Embed every document from MongoDB and upsert it into Qdrant."""
    for doc in collection.find():
        content = doc['content']
        chunks = chunk_text(content)
        # Embed and store each chunk as its own point, so retrieval can return
        # the specific passage relevant to a query rather than the whole document
        for chunk_idx, chunk in enumerate(chunks):
            embedding = extract_features(chunk)
            # Derive a stable point ID from the Mongo document _id + chunk index so re-runs upsert in place
            point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, str(doc['_id']) + str(chunk_idx)))

            payload = {
                "text": chunk,
                "url": doc.get("url", ""),
                "chunk_idx": chunk_idx
            }

            model_point = models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload
            )
            qdrant.upsert(collection_name=collection_name, points=[model_point])
        print(f"Features for {doc['url']} pushed to Qdrant ({len(chunks)} chunks)")



push_features_to_qdrant()
