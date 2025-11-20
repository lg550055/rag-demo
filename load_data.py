import re
import os


def load_documents(folder_path) -> list[str]:
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                docs.append(f.read())
    return docs


def clean_text(text) -> str:
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    return text.strip()


def prepare_docs(folder_path="data/") -> list[str]:
    """
    Loads and cleans all text documents from the given folder.
    """
    raw_docs = load_documents(folder_path)

    cleaned_docs = [clean_text(doc) for doc in raw_docs]

    print(f"Prepared {len(cleaned_docs)} documents.")
    return cleaned_docs
