import os
import pickle
import fitz  # PyMuPDF
from io import BytesIO
from striprtf.striprtf import rtf_to_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings  # Updated import
from langchain_community.vectorstores import Chroma  # Updated import
from langchain.llms import OpenAI  # Ensure this is imported
from langchain.chains import RetrievalQA  # Ensure this is imported
import spacy
from nltk.corpus import stopwords
import nltk
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from keybert import KeyBERT
from textblob import TextBlob
from langdetect import detect
import textstat
import time  # Ensure this is imported

nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()

class CustomTextLoader:
    def __init__(self):
        self.handlers = {
            '.docx': self.text_from_docx,
            '.pptx': self.text_from_pptx,
            '.pdf': self.text_from_pdf,
            '.rtf': self.text_from_rtf,
            '.txt': self.text_from_txt,
        }

    def load(self, file):
        ext = os.path.splitext(file.name)[-1].lower()
        if ext in self.handlers:
            text = self.handlers[ext](file)
            title = os.path.splitext(os.path.basename(file.name))[0]
            return Document(page_content=text, metadata={"source": title})
        else:
            raise ValueError(f"Unsupported file type {ext}")

    def text_from_pdf(self, file):
        file.seek(0)
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ''
        for page in doc:
            text += page.get_text("text")
        doc.close()
        return text

    def text_from_docx(self, file):
        import docx
        file.seek(0)
        doc = docx.Document(file)
        return '\n'.join(para.text for para in doc.paragraphs)

    def text_from_pptx(self, file):
        import pptx
        file.seek(0)
        ppt = pptx.Presentation(BytesIO(file.read()))
        return '\n'.join(
            shape.text for slide in ppt.slides for shape in slide.shapes if hasattr(shape, "text")
        )

    def text_from_rtf(self, file):
        file.seek(0)
        return rtf_to_text(file.read().decode("utf-8"))

    def text_from_txt(self, file):
        file.seek(0)
        return file.read().decode("utf-8")

def load_and_process_document(file_path, loader):
    try:
        with open(file_path, 'rb') as file:
            document_instance = loader.load(file)
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents([document_instance])
            return chunks
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

def embed_documents_in_batches(documents, embedding_model, batch_size=10):
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    embeddings = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(embedding_model.embed_documents, [doc.page_content for doc in batch]) for batch in batches]
        for future in as_completed(futures):
            embeddings.extend(future.result())
    return embeddings

def perform_lda_topic_modeling(texts):
    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=10, random_state=0)
    lda.fit(count_data)
    return lda, count_vectorizer

def enhance_metadata_with_topic_modeling(doc, lda_model, count_vectorizer):
    content = doc.page_content
    count_data = count_vectorizer.transform([content])
    topics = lda_model.transform(count_data)[0]
    top_topic = topics.argmax()
    
    doc_nlp = nlp(content)
    entities = [ent.text for ent in doc_nlp.ents]
    
    keywords = kw_model.extract_keywords(content, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
    
    text_blob = TextBlob(content)
    sentiment = text_blob.sentiment.polarity
    language = detect(content)
    flesch = textstat.flesch_kincaid_grade(content)
    length = len(content.split())

    doc.metadata.update({
        "entities": ', '.join(entities),
        "keywords": ', '.join([kw[0] for kw in keywords]),
        "topic_id": int(top_topic),
        "topic_prob": float(topics[top_topic]),
        "topics": ', '.join([f"Topic {i}: {prob:.4f}" for i, prob in enumerate(topics)]),
        "sentiment": sentiment,
        "language": language,
        "readability_flesch_kincaid": flesch,
        "document_length": length
    })

    return doc

def process_documents(main_document_path, file_paths, save_path):
    start_time = time.time()
    all_chunks = []
    loader = CustomTextLoader()

    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            doc_similarities, main_doc_embedding, saved_chunks = pickle.load(f)
        print("Loaded precomputed embeddings from file.")
        
        embedding = OpenAIEmbeddings()
        # push embeddings to cromadb
        vectordb = Chroma.from_documents(
            documents=saved_chunks,
            embedding=embedding,
            persist_directory="db_optimized3"
        )
        
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(max_tokens=512),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        print(f"Process documents completed in {time.time() - start_time} seconds")
        return qa_chain, doc_similarities, main_doc_embedding, saved_chunks

    main_load_start = time.time()
    with open(main_document_path, 'rb') as main_file:
        main_doc_instance = loader.load(main_file)
        main_doc_embedding = OpenAIEmbeddings().embed_documents([main_doc_instance.page_content])[0]
    print(f"Main document loaded and embedded in {time.time() - main_load_start} seconds")

    process_docs_start = time.time()
    with ThreadPoolExecutor() as executor:
        futures = []
        for file_path in file_paths:
            futures.append(executor.submit(load_and_process_document, file_path, loader))

        for future in as_completed(futures):
            chunks = future.result()
            if chunks:
                all_chunks.extend(chunks)

    print(f"Other documents processed in {time.time() - process_docs_start} seconds")

    if all_chunks:
        lda_start = time.time()
        stop_words = set(stopwords.words('english'))
        texts = [' '.join([word for word in doc.page_content.lower().split() if word not in stop_words]) for doc in all_chunks]

        lda_model, count_vectorizer = perform_lda_topic_modeling(texts)

        all_chunks = [enhance_metadata_with_topic_modeling(chunk, lda_model, count_vectorizer) for chunk in all_chunks]
        print(f"LDA topic modeling completed in {time.time() - lda_start} seconds")

        similarity_start = time.time()
        chunk_embeddings = embed_documents_in_batches(all_chunks, OpenAIEmbeddings(), batch_size=10)
        similarity_scores = [calculate_similarity(main_doc_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings]
        doc_similarities = list(zip(all_chunks, similarity_scores))
        print(f"Similarity calculations completed in {time.time() - similarity_start} seconds")

        save_start = time.time()
        with open(save_path, 'wb') as f:
            pickle.dump((doc_similarities, main_doc_embedding, all_chunks), f)
        print(f"Precomputed data saved in {time.time() - save_start} seconds")

        vectordb_start = time.time()
        embedding = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(
            documents=all_chunks,
            embedding=embedding,
            persist_directory="db_optimized3"
        )

        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(max_tokens=512),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        print(f"Vector store and QA chain creation completed in {time.time() - vectordb_start} seconds")
        print(f"Process documents completed in {time.time() - start_time} seconds")
        return qa_chain, doc_similarities, main_doc_embedding, all_chunks

    print(f"Process documents completed in {time.time() - start_time} seconds")
    return None, None, None, all_chunks

def calculate_similarity(embedding1, embedding2):
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    harsher_similarity = similarity ** 2
    return harsher_similarity