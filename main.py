import os
from load_data import prepare_docs
from split_text import split_docs
from create_embeddings import get_embeddings
from store_faiss import build_faiss_index, save_metadata
from generate_answer import generate_answer

def run_pipeline():
    """
    Runs end-to-end RAG workflow
    """
    # if faiss index and metadata already exist, skip to answer generation
    if not os.path.exists("faiss_index.index") and not os.path.exists("faiss_metadata.pkl"):
        print("Load and clean data:")
        documents = prepare_docs("data/")
        print(f"Loaded {len(documents)} clean documents.\n")

        print("Split text into chunks:")
        # documents is a list of strings, but split_docs expects a list of documents
        # For this simple example where documents are small, we pass them as strings
        chunks_as_text = split_docs(documents, chunk_size=500, chunk_overlap=100)
        # In this case, chunks_as_text is a list of LangChain Document objects

        # Extract text content from LangChain Document objects
        texts = [c.page_content for c in chunks_as_text]
        print(f"Created {len(texts)} text chunks.\n")

        print("Generate Embeddings:")
        embeddings = get_embeddings(texts)

        print("Store Embeddings in FAISS:")
        build_faiss_index(embeddings)
        save_metadata(texts)
        print("Stored embeddings and metadata successfully.\n")

    print("Retrieve faiss index & Generate Answer:")
    query = "Does unsupervised ML cover regression tasks?"
    generate_answer(query)

if __name__ == "__main__":
    run_pipeline()