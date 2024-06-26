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
from document_processing import retrieve_embeddings, calculate_similarity
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


from openai import OpenAI
import openai
os.environ["OPENAI_API_KEY"] = "sk-proj-RtDTB0Gu43mSTegX8soqT3BlbkFJelbbFftCPBNORR9dfpyp"
# Ensure the OpenAI API key is set
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

nlp = spacy.load("en_core_web_sm")

def extract_key_terms(query):
    """
    Extract key terms from a query using Spacy.
    """
    doc = nlp(query)
    key_terms = [token.text.lower() for token in doc if not token.is_stop]
    return key_terms

def calculate_term_relevance(doc, key_terms):
    """
    Calculate the relevance of key terms in a document.
    """
    term_counts = Counter(doc.page_content.lower().split())
    relevance = sum(term_counts[term] for term in key_terms)
    return relevance

def adjust_similarity_scores(query, doc_similarities, adjustment_factor=2):
    """
    Adjust similarity scores based on term relevance.
    """
    key_terms = extract_key_terms(query)
    adjusted_similarities = []
    
    for doc, initial_score in doc_similarities:
        term_relevance = calculate_term_relevance(doc, key_terms)
        adjusted_score = initial_score + term_relevance * 1.25
        adjusted_similarities.append((doc, adjusted_score))
    
    return adjusted_similarities

def update_conversation_history(history, new_entry, max_length=5):
    """
    Update conversation history with a sliding window of the most recent interactions.
    """
    history.append(new_entry)
    if len(history) > max_length:
        history.pop(0)
    return history

def calculate_chunk_relevance(query, adjusted_docs):
    """
    Calculate the relevance of query chunks.
    """
    openai_embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    query_embedding = openai_embedding_model.embed_documents([query])[0]
    doc_embeddings = [doc.metadata["embedding"] for doc in adjusted_docs]
    return calculate_query_relevance(query_embedding, doc_embeddings)

def calculate_query_relevance(query_embedding, document_embeddings):
    """
    Calculate the relevance scores between a query embedding and a list of document embeddings.
    
    Parameters:
        query_embedding (numpy array): The embedding vector of the query.
        document_embeddings (list of numpy arrays): List of embedding vectors for documents.
    
    Returns:
        list of floats: Relevance scores between the query and each document embedding.
    """
    if not document_embeddings:
        return []
    relevance_scores = cosine_similarity([query_embedding], document_embeddings)[0]
    return relevance_scores


def detect_new_topic(conversation_history, query, threshold=0.5):
    """
    Detect if the new query is a different topic from the recent conversation history.
    """
    if len(conversation_history) < 2:
        return False  # Not enough history to determine a new topic

    # Calculate semantic similarity
    previous_queries = " ".join(conversation_history[-2:])
    previous_embedding = sentence_model.encode(previous_queries, convert_to_tensor=True)
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)

    similarity = cosine_similarity([previous_embedding.cpu().numpy()], [query_embedding.cpu().numpy()])[0][0]
    
    # Check if the similarity is below the threshold
    return similarity < threshold

def detect_hallucination(response_text, adjusted_docs):
    """
    Detect if the generated response contains information not supported by the documents.
    """
    hallucination_detected = False
    checks = []

    # Combine all document contents
    combined_content = " ".join([doc.page_content for doc in adjusted_docs]).lower()
    response_text_lower = response_text.lower()

    # Split the response into sentences
    response_sentences = response_text_lower.split(".")
    
    # Check each sentence if it's supported by any document content
    for sentence in response_sentences:
        if sentence.strip() and sentence.strip() not in combined_content:
            hallucination_detected = True
            checks.append(f"Potential hallucination: '{sentence.strip()}' not found in documents.")

    return hallucination_detected, checks

def get_confidence_level(confidence_score):
    """
    Determine the confidence level based on the confidence score.
    """
    if confidence_score > 90:
        return "High"
    elif confidence_score > 70:
        return "Medium"
    else:
        return "Low"


def get_llm_responses(queries, conversation_history):
    """
    Retrieve responses from the LLM based on adjusted similarity scores.
    """
    responses = []
    openai_embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    seen_documents = set()

    for query in queries:
        step_start_time = time.time()
        topic_shifted = detect_new_topic(conversation_history, query)
        print(f"Topic shifted: {topic_shifted}, Time taken: {time.time() - step_start_time:.2f} seconds")

        if topic_shifted:
            conversation_history = []  # Reset conversation history if a new topic is detected

        step_start_time = time.time()
        documents, embeddings = retrieve_embeddings()
        print(f"Document embeddings retrieved in {time.time() - step_start_time:.2f} seconds")

        step_start_time = time.time()
        query_embedding = openai_embedding_model.embed_documents([query])[0]
        print(f"Query embedding created in {time.time() - step_start_time:.2f} seconds")

        step_start_time = time.time()
        doc_similarities = [(Document(page_content=doc.page_content, metadata=doc.metadata), calculate_similarity(query_embedding, emb)) for doc, emb in zip(documents, embeddings)]
        print(f"Document similarities calculated in {time.time() - step_start_time:.2f} seconds")

        step_start_time = time.time()
        adjusted_similarities = adjust_similarity_scores(query, doc_similarities)
        print(f"Similarity scores adjusted in {time.time() - step_start_time:.2f} seconds")

        top_n = 10
        adjusted_similarities = sorted(adjusted_similarities, key=lambda x: x[1], reverse=True)

        adjusted_docs = []
        for doc, score in adjusted_similarities:
            if doc.page_content not in seen_documents:
                seen_documents.add(doc.page_content)
                metadata = {k: (', '.join(map(str, v)) if isinstance(v, list) else str(v)) for k, v in doc.metadata.items()}
                metadata["weight"] = float(score)
                metadata["embedding"] = doc.metadata["embedding"]  # Ensure embedding is in metadata
                adjusted_docs.append(Document(page_content=doc.page_content, metadata=metadata))
                if len(adjusted_docs) >= top_n:
                    break

        # Combine the top adjusted documents into a single context for the LLM
        context = "\n\n".join([doc.page_content for doc in adjusted_docs])

        step_start_time = time.time()
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a seasoned sales representative and the questions being asked are questions by junior reps who have questions about your own company and competitors. The answers need to be detailed with specificity and not give any generic answers and should be answers that junior reps can directly tell potential prospects during a discovery call."},
                {"role": "user", "content": context + "\n\n" + query}
            ],
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.2
        )
        print(f"OpenAI completion created in {time.time() - step_start_time:.2f} seconds")

        # Access the generated text from the response
        response_text = completion.choices[0].message.content.strip()

        relevant_metadata = [doc.metadata for doc in adjusted_docs]

        step_start_time = time.time()
        query_relevance_scores = calculate_query_relevance(query_embedding, [doc.metadata["embedding"] for doc in adjusted_docs])
        print(f"Query relevance scores calculated in {time.time() - step_start_time:.2f} seconds")

        step_start_time = time.time()
        chunk_relevance_scores = calculate_chunk_relevance(query, adjusted_docs)
        print(f"Chunk relevance scores calculated in {time.time() - step_start_time:.2f} seconds")

        step_start_time = time.time()
        hallucination_detected, checks = detect_hallucination(response_text, adjusted_docs)
        print(f"Hallucination detection completed in {time.time() - step_start_time:.2f} seconds")

        step_start_time = time.time()
        confidence_score = (sum(query_relevance_scores) + sum(chunk_relevance_scores)) * 100 / (len(query_relevance_scores) + len(chunk_relevance_scores))
        if hallucination_detected:
            confidence_score *= 0.5  # Penalize for hallucination
        confidence_level = get_confidence_level(confidence_score)
        print(f"Confidence score calculated in {time.time() - step_start_time:.2f} seconds")

        responses.append({
            "result": response_text,
            "content": [doc.page_content for doc in adjusted_docs],  # Only include content
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "hallucination_checks": checks  # Add detailed check results
        })

    return responses