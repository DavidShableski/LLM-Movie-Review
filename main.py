#David Shableski 7/2/2025
#dbshableski@gmail.com
#A local Q&A app for movie reviews using LangChain, Chroma, and Ollama.
#This runs entirely offline and demonstrates a multi-step local RAG pipeline:
# first summarizing the retrieved reviews, then answering the user’s question.

import os
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever, vector_store
from rich import print
from rich.console import Console

console = Console()


#Disable telemetry
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["OLLAMA_DISABLE_TELEMETRY"] = "1"
os.environ["LLAMA_TELEMETRY"] = "off"


#Initialize LLM
model = OllamaLLM(model="llama3.2")

#Summarization prompt
summary_template = """
You are an expert movie critic. Here are some raw reviews pulled from a database:

{reviews}

Please provide a short summary of what these reviews are saying, as 3 bullet points.
"""
summary_prompt = ChatPromptTemplate.from_template(summary_template)
summary_chain = summary_prompt | model

# Question answering prompt
answer_template = """
You are a movie analysis expert.

Here is a summary of relevant reviews:
{summary}

Now, please answer the customer's question:
{question}
"""

answer_prompt = ChatPromptTemplate.from_template(answer_template)
answer_chain = answer_prompt | model

# Main loop
try:
    while True:
        print("\n--------------------")
        question = input("Ask your question about movies (q to quit): ").strip()
        print("\n--------------------")

        if question.lower() == "q":
            print("Goodbye!")
            break

         # Determine how many docs to pull
        if len(question.split()) > 8 or any(word in question.lower() for word in ["compare", "movies", "decade", "years", "versus"]):
            k_value = 10
        else:
            k_value = 3

         # Retrieve
        dynamic_retriever = vector_store.as_retriever(search_kwargs={"k": k_value})
        reviews = dynamic_retriever.invoke(question)

        # Summarize
        summary = summary_chain.invoke({"reviews": reviews})

        # Answer
        result = answer_chain.invoke({"summary": summary, "question": question})

        console.print("\n[bold blue]Summary of reviews:[/bold blue]")
        console.print(summary)

        console.print("\n[bold green]Response:[/bold green]")
        console.print(result)


except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting...")
