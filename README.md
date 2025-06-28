# LLM Movie Review Q&A

## YouTube Demo

[![Watch on YouTube](https://img.shields.io/badge/Watch-YouTube-red)]((https://youtu.be/i1WWWHY6XCM))


A fully local retrieval-augmented generation (RAG) pipeline that answers natural language questions about movie reviews.  
Runs entirely offline using:

- 🧠 **Ollama** for local embeddings & LLM inference
- 🗃️ **Chroma** for the vector database
- 🔗 **LangChain** for retrieval + prompt orchestration
- ✨ **Rich** for clean terminal output
- 🔍 **NLTK** for simple keyword analysis

---

## 🚀 Features

✅ Loads movie reviews from a CSV file into a local Chroma vector store, using Ollama to create embeddings.

✅ Dynamically decides **how many reviews to pull** based on your question:
- Small, direct questions pull just `3` documents.
- Broad or comparative questions pull up to `10`.

✅ Summarizes the retrieved reviews in bullet points, then answers your question with a local language model.

✅ Displays everything in a clean, color-coded terminal interface.

✅ Shows the most frequent words from the matched reviews to give quick analytic insights.

---

📂 How it works
vector.py
Loads movie_reviews.csv, embeds each review, and builds a Chroma vector database.
Makes a retriever you can import anywhere.

main.py
Sets up a multi-step pipeline with LangChain:

Retrieves relevant reviews

Summarizes them

Answers your question based on that summary

Highlights top keywords with nltk

rich makes it look polished in the terminal.

🛠️ Install & run
bash
Copy
Edit
pip install -r requirements.txt
python vector.py     # Builds your local vector DB
python main.py       # Run the Q&A app
💼 Why this matters
This small project shows how to build a full local RAG system from scratch, combining embedding generation, semantic retrieval, prompt chaining, and lightweight data analysis — all without relying on external APIs.

✍️ Author
David Shableski
dbshableski@gmail.com
