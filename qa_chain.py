import os
import pickle  # For saving and loading embeddings
from io import BytesIO
import fitz  # PyMuPDF for PDFs
from striprtf.striprtf import rtf_to_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.utils import filter_complex_metadata
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import spacy
import gensim
from gensim import corpora
from nltk.corpus import stopwords
import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

nlp = spacy.load("en_core_web_sm")

def calculate_similarity(embedding1, embedding2):
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    harsher_similarity = similarity ** 2
    return harsher_similarity

def extract_key_terms(query):
    doc = nlp(query)
    key_terms = [token.text.lower() for token in doc if not token.is_stop]
    return key_terms

def calculate_term_relevance(doc, key_terms):
    term_counts = Counter(doc.page_content.lower().split())
    relevance = sum(term_counts[term] for term in key_terms)
    return relevance

def adjust_similarity_scores(query, doc_similarities):
    key_terms = extract_key_terms(query)
    adjusted_similarities = []
    
    for doc, initial_score in doc_similarities:
        term_relevance = calculate_term_relevance(doc, key_terms)
        adjusted_score = initial_score + term_relevance * 0.25
        adjusted_similarities.append((doc, adjusted_score))
    
    return adjusted_similarities

def get_llm_responses(queries, doc_similarities):
    responses = []
    embedding = OpenAIEmbeddings()

    for query in queries:

        adjusted_similarities = adjust_similarity_scores(query, doc_similarities)

        top_n = 10
        adjusted_similarities = sorted(adjusted_similarities, key=lambda x: x[1], reverse=True)[:top_n]

        adjusted_docs = []
        for doc, score in adjusted_similarities:
            metadata = {k: (', '.join(v) if isinstance(v, list) else v) for k, v in doc.metadata.items()}
            metadata["weight"] = float(score)
            adjusted_docs.append(Document(page_content=doc.page_content, metadata=metadata))
        
        vectordb = Chroma.from_documents(
            documents=adjusted_docs,
            embedding=embedding,
            persist_directory="db_temp5"
        )
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(max_tokens=1024),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        
        llm_response = qa_chain(query)
        responses.append({
            "result": llm_response["result"],
            "source_documents": llm_response["source_documents"]
        })

    return responses

