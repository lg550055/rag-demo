import os
from dotenv import load_dotenv
from openai import OpenAI
from retrieve_faiss import load_faiss_index, load_metadata, retrieve_similar_chunks

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

def generate_answer_api(query, top_k=3) -> None:
    """
    Retrieves relevant chunks and generates a final answer using OpenAI API.
    """
    # Load FAISS index and metadata
    index = load_faiss_index()
    text_chunks = load_metadata()

    # Retrieve top relevant chunks
    context_chunks = retrieve_similar_chunks(query, index, text_chunks, top_k=top_k)
    context = "\n\n".join(context_chunks)

    # Initialize OpenAI client
    client = OpenAI(api_key=API_KEY)
    
    # Build the prompt with retrieved context
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context."},
        {"role": "user", "content": f"""Context:
{context}

Question: {query}

Please provide a clear and concise answer based on the context above."""}
    ]

    # Generate answer using OpenAI API
    print("Generating answer with OpenAI...")
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=messages,  # type: ignore
        # max_tokens=300,  # gpt-5-nano supports max_completion_tokens
        # temperature=0.7  # gpt-5-nano does not support temperature
    )
    
    answer = response.choices[0].message.content
    
    print("Final Answer:")
    print(answer)
