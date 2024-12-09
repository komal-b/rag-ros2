import gradio as gr
from qdrant_client import QdrantClient
from transformers import pipeline as hf_pipeline
from loguru import logger
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


QDRANT_HOST = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "raw_data"

MODEL_NAME = "gpt2"  # Replace with your model if needed
# hf_model = hf_pipeline("text-generation", model=MODEL_NAME)
fine_tuned_model = AutoModelForCausalLM.from_pretrained("./fine_tuned_llama2")
fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_llama2")

hf_model = pipeline("text-generation", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)



qdrant_client = QdrantClient(url=QDRANT_HOST)

def generate_query_embedding(query: str):
    """
    Generate a query embedding using a suitable transformer model.
    This function can be customized for your specific embedding model.
    """
    # Use a pre-trained sentence-transformer or other embedding model
    embedding_model = hf_pipeline("feature-extraction", model="sentence-transformers/all-distilroberta-v1")
    embedding = embedding_model(query)[0][0]  # Extract the first embedding result
    assert len(embedding) == 768
    return embedding

def fetch_similar_data(query: str, top_k: int = 3) -> list:
    """
    Fetch similar data from Qdrant based on the query.
    """
    logger.info("Connecting to Qdrant...")
    
    # Generate the query embedding
    query_embedding = generate_query_embedding(query)
    
    # Perform a vector similarity search in Qdrant
    results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
    )

    # Collect the retrieved documents and their scores
    similar_data = []
    for result in results:
        # Check if 'text' is in the payload
        text = result.payload.get("text", None)
        if text:  # Only add results with a valid 'text' field
            similar_data.append({"text": text, "score": result.score})
        else:
            logger.warning(f"Result does not contain 'text' field: {result.payload}")

    logger.info(f"Retrieved {len(similar_data)} similar documents from Qdrant.")
    return similar_data

def generate_response(query: str, similar_data: list) -> str:
    """
    Generate a response using the Hugging Face model.
    """
    context = "\n".join([f"Context {idx+1}: {doc['text']}" for idx, doc in enumerate(similar_data)])
    prompt = f"Answer the following question based on the context:\n\n{context}\n\nQuestion: {query}"

    logger.info(f"Sending prompt to model: {prompt[:200]}...")  # Log part of the prompt for debugging
    
    response = hf_model(prompt, max_length=200, num_return_sequences=1)
    return response[0]["generated_text"]

def rag_pipeline(query: str):
    """
    Perform the full RAG pipeline: query -> knowledge base -> model -> response.
    """
    logger.info(f"Received query: {query}")

    similar_data = fetch_similar_data(query)
    logger.info(f"Similar data fetched: {similar_data}")

    response = generate_response(query, similar_data)

    logger.info(f"Final response: {response}")
    return response

def gradio_interface(query: str):
    """
    Gradio interface function to interact with the RAG pipeline.
    """
    return rag_pipeline(query)

# Initialize the Gradio Interface
gradio_ui = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Enter your query", placeholder="Ask a question..."),
    outputs=gr.Textbox(label="Generated Response"),
    live=False,  # Optional: To get real-time updates
    title="RAG-based Query System",
    description="This system uses Qdrant to fetch context based on the query and generate a response using GPT-2 or another language model."
)

if __name__ == "__main__":
    gradio_ui.launch(share=True)  # Set share=True to create a public link
