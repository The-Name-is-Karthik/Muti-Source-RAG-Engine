"""
Defines the core Retrieval-Augmented Generation (RAG) chain.

This module connects all the pieces: the retriever from the vector store,
a prompt template to structure the query, the LLM to generate a response,
and an output parser to format the final answer. The create_rag_chain
function builds and returns this entire processing pipeline.
"""

import logging
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

import src.config as config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Main RAG Chain Function ---
def create_rag_chain(retriever: Chroma.as_retriever):
    """
    Creates and returns a RAG chain for question-answering.

    Args:
        retriever: A retriever object from a vector store (e.g., Chroma).

    Returns:
        A runnable LangChain sequence (RAG chain).
    """
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        logging.info("Gemini-2.5.flash model initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Gemini-2.5.flash model: {e}")
        raise

    prompt_template = """
    You are an expert assistant for answering questions about YouTube videos.
    Answer the user's question based ONLY on the following context from the video transcript.
    If the information is not in the context, say "I don't know, the information is not in the Knowledge base."
    Do not make up information. Be concise and helpful.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    logging.info("Prompt template created.")

    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("RAG chain created successfully.")

    return rag_chain