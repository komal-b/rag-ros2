from transformers import BertTokenizer, BertModel
import torch
import qdrant_client
from qdrant_client import models
from pymongo import MongoClient
import uuid
from bson import ObjectId
from qdrant_client.models import VectorParams, Distance  

mongo_client = MongoClient('mongodb://llm_engineering:llm_engineering@127.0.0.1:27017')
db = mongo_client['rag_system']
collection = db['raw_data']

qdrant = qdrant_client.QdrantClient("http://127.0.0.1:6333")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

collection_name = "raw_data"
collections_response = qdrant.get_collections()

if not any(col.name == collection_name for col in collections_response.collections):
    
    qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=768,  
        distance=Distance.COSINE  
    )
)

def extract_features(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def push_features_to_qdrant():
    for doc in collection.find():
        content = doc['content']
        embedding = extract_features(content)

        point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, str(doc['_id'])))

        payload = {
            "embedding": embedding.tolist(),
            "metadata": {"source": doc['source'], "url": doc['url']}
        }

        point = models.PointStruct(id=point_id, vector=embedding.tolist(), payload=payload)
        qdrant.upsert(collection_name=collection_name, points=[point])
        print(f"Features for {doc['url']} pushed to Qdrant.")

push_features_to_qdrant()
