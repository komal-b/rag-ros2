from dotenv import load_dotenv
import os
import gradio as gr
from qdrant_client import QdrantClient
from loguru import logger
from sentence_transformers import SentenceTransformer
from groq import Groq



# Load GROQ_API_KEY from .env so the key isn't hardcoded/committed
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=api_key)

QDRANT_HOST = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "raw_data"  # populated by feature_engineering.py

# Must match the model used in feature_engineering.py so query and document vectors share the same space
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')



qdrant_client = QdrantClient(url=QDRANT_HOST)

def generate_query_embedding(query: str):
    """
    Generate a query embedding using a suitable transformer model.
    This function can be customized for your specific embedding model.
    """
    # Use a pre-trained sentence-transformer or other embedding model
    
    return embedding_model.encode(query, convert_to_numpy=True)

def fetch_similar_data(query: str, top_k: int = 3) -> list:
    """
    Fetch similar data from Qdrant based on the query.
    """
    logger.info("Connecting to Qdrant...")

    # Generate the query embedding
    query_embedding = generate_query_embedding(query)

    # Perform a vector similarity search in Qdrant (cosine distance, per the collection's config)
    # returning the top_k nearest document chunks to the query embedding
    results = qdrant_client.query_points(
    collection_name=QDRANT_COLLECTION_NAME,
    query=query_embedding.tolist(),
    limit=top_k,
    ).points

    # Collect the retrieved documents and their scores
    similar_data = []
    for result in results:
        # Check if 'text' is in the payload
        text = result.payload.get("text", None)
        if text:  # Only add results with a valid 'text' field
            similar_data.append({"text": text, "score": result.score})
        else:
            logger.warning(f"Result does not contain 'text' field: {result.payload}")
    if not similar_data:
                logger.warning("No similar documents found in Qdrant")
                return []
    logger.info(f"Retrieved {len(similar_data)} similar documents from Qdrant.")
    return similar_data


def generate_response(query: str, similar_data: list, history: list) -> str:
    """Generate a response using Groq API with Llama3."""
    # Concatenate the retrieved chunks into a single numbered context block for the prompt
    context = "\n".join([f"Context {idx+1}: {doc['text']}" for idx, doc in enumerate(similar_data)])

    # Truncate to the first 500 words to avoid exceeding the model's token limit
    # (combined with chat history + system prompt, the full request must fit the context window)
    context_words = context.split()[:500]
    context = ' '.join(context_words)
    messages = [
        {"role": "system", "content": "You are a helpful ROS2 robotics assistant."}
    ]

     # Replay prior turns so the model retains conversational context across messages
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})

    # Final user turn: ground the model's answer in the retrieved ROS2 documentation
    messages.append({
        "role": "user",
        "content": f"Answer based on this context:\n{context}\n\nQuestion: {query}"
    })
    # call Groq API - llama-3.1-8b-instant is small/fast, suited for low-latency chat responses;
    # max_tokens caps the response length (and cost/latency)
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=300
    )
    return response.choices[0].message.content

def rag_pipeline(query: str, history: list):
    """
    Perform the full RAG pipeline: query -> knowledge base -> model -> response.
    """
    logger.info(f"Received query: {query}")

    # Retrieval step: find document chunks relevant to the query
    similar_data = fetch_similar_data(query)
    if not similar_data:
        # Bail out early rather than asking the LLM to answer with no grounding context
        return "I could not find relevant ROS2 documentation for this query."
    logger.info(f"Similar data fetched: {similar_data}")

    # Generation step: ask the LLM to answer using the retrieved context
    response = generate_response(query, similar_data, history)

    logger.info(f"Final response: {response}")
    return response

def gradio_interface(message: str, history: list):
    """
    Chat interface function — history maintained automatically by Gradio.
    """
    # Gradio passes `history` as a list of (user_message, assistant_message) tuples,
    # which matches the format generate_response expects
    return rag_pipeline(message, history)

# Initialize fancy Chat Interface
gradio_ui = gr.ChatInterface(
    fn=gradio_interface,  # called on every user message; receives (message, history)
    title="🤖 ROS2 RAG Assistant",
    description="""
    Ask questions about ROS2, Nav2, MoveIt2, and Gazebo.
    Answers are grounded in real ROS2 documentation retrieved from a vector database.
    """,
    # Example prompts shown to the user as clickable suggestions
    examples=[
        "What is Nav2 and how does it work?",
        "How do I configure Nav2 costmap parameters?",
        "What is MoveIt2 used for?",
        "How do I set up SLAM in ROS2?",
        "What is Gazebo simulation?"
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    # share=True asks Gradio to create a temporary public URL (tunnels through Gradio's servers)
    # so the app is reachable outside localhost
    gradio_ui.launch(share=True)
