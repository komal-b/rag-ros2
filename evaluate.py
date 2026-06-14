from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from dotenv import load_dotenv

from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from ragas.run_config import RunConfig


load_dotenv()

# reuse your existing pipeline components
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# configure RAGAS to use Groq instead of OpenAI
groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

ragas_llm = LangchainLLMWrapper(groq_llm)

# Groq's API rejects n > 1, but AnswerRelevancy defaults to strictness=3 (i.e. n=3
# self-consistency samples). Lower it to 1 so it requests a single completion.
answer_relevancy.strictness = 1
ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)
qdrant_client = QdrantClient(url="http://localhost:6333")
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# golden dataset
golden_dataset = [
    {"question": "What is Nav2?", "ground_truth": "Nav2 is the second generation ROS navigation stack for autonomous robot navigation in ROS2"},
    {"question": "What is a costmap in Nav2?", "ground_truth": "A costmap represents the environment as a grid where each cell has a cost value indicating obstacles"},
    {"question": "What is MoveIt2?", "ground_truth": "MoveIt2 is a motion planning framework for robot manipulation in ROS2"},
    {"question": "What is Gazebo?", "ground_truth": "Gazebo is a robot simulation environment that works with ROS2"},
    {"question": "What is SLAM in ROS2?", "ground_truth": "SLAM stands for Simultaneous Localization and Mapping allowing robots to build maps while navigating"}
    ]

def get_embedding(text):
    return embedding_model.encode(text, convert_to_numpy=True).tolist()

def retrieve_context(question, top_k=3):
    query_vector = get_embedding(question)
    results = qdrant_client.query_points(
        collection_name="raw_data",
        query=query_vector,
        limit=top_k,
    ).points
    return [r.payload.get("text", "") for r in results if r.payload.get("text")]

def generate_answer(question, contexts):
    context_str = "\n".join([f"Context {i+1}: {c[:300]}" for i, c in enumerate(contexts)])
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful ROS2 robotics assistant."},
            {"role": "user", "content": f"Answer based on this context:\n{context_str}\n\nQuestion: {question}"}
        ],
        max_tokens=200
    )
    return response.choices[0].message.content

# run pipeline on all questions
questions = []
answers = []
contexts = []
ground_truths = []

print("Running RAG pipeline on golden dataset...")
for item in golden_dataset:
    q = item["question"]
    gt = item["ground_truth"]
    
    ctx = retrieve_context(q)
    ans = generate_answer(q, ctx)
    
    questions.append(q)
    answers.append(ans)
    contexts.append(ctx)
    ground_truths.append(gt)
    
    print(f"✅ {q}")

# create RAGAS dataset
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}

dataset = Dataset.from_dict(data)

# run RAGAS evaluation
print("\nRunning RAGAS evaluation...")
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=RunConfig(max_workers=1, timeout=300)
)

print("\n=== RAGAS Evaluation Results ===")
print(results)