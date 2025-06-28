#David Shableski 7/2/2025
#dbshableski@gmail.com
#Creates a local Chroma vector database from movie reviews CSV using Ollama embeddings.

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Configuration
DATA_FILE = "movie_reviews.csv"
DB_PATH = "./chroma_movie_db"
COLLECTION_NAME = "movie_reviews"
EMBEDDING_MODEL = "mxbai-embed-large"

# Load data
df = pd.read_csv(DATA_FILE)

# Set up embeddings
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Check if we need to build the database
needs_initialization = not os.path.exists(DB_PATH)

# Prepare documents
documents_to_add = []
if needs_initialization:
    for idx, row in df.iterrows():
        content = f"{row['Movie']} {row['Review']}"
        doc = Document(
            page_content=content,
            metadata={
                "movie": row["Movie"],
                "rating": row["Rating"],
                "date": row["Date"]
            },
            id=str(idx)
        )
        documents_to_add.append(doc)

# Create or load the Chroma DB
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

# Populate DB if first run
if needs_initialization:
    vector_store.add_documents(documents=documents_to_add)

# Make a retriever for main.py to import
retriever = vector_store.as_retriever()
