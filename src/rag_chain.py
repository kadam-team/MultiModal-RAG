# Functions for setting up the RAG chain and the language model.

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# The import path for config has been updated to reflect the new folder structure
from src.config import GROQ_API_KEY, GROQ_MODEL_NAME

def get_llm():
    """Initializes and returns the Groq language model."""
    llm = ChatGroq(
        model_name=GROQ_MODEL_NAME,
        api_key=GROQ_API_KEY
    )
    print(f"Groq LLM '{GROQ_MODEL_NAME}' initialized.")
    return llm

def get_rag_prompt():
    """Returns the ChatPromptTemplate for the RAG chain."""
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nContext:\n{context}"),
        ("user", "{question}")
    ])

def format_docs_for_context(docs):
    """Formats retrieved documents into a single string for the prompt context."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever, llm, prompt):
    """Defines and returns the complete RAG chain."""
    rag_chain = (
        {"context": retriever | format_docs_for_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG pipeline setup complete.")
    return rag_chain