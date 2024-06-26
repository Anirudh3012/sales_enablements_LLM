import os
import time  # Ensure this import is included
import tkinter as tk
from tkinter import filedialog
from dotenv import load_dotenv
from web_scraping import scrape_website_content
from reviews_api import get_reviews
from document_processing import process_documents, CustomTextLoader, Document, embed_documents_in_batches
from qa_chain import get_llm_responses
from langchain_community.embeddings import OpenAIEmbeddings  # Correct import

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

def main():
    root = tk.Tk()
    root.withdraw()

    company_url = input("Enter the company's website URL: ")
    company_product_name = input("Enter the company's product name: ")
    competitor_url = input("Enter the competitor's website URL: ")
    competitor_product_name = input("Enter the competitor's product name: ")

    print("Scraping company website...")
    company_web_text = scrape_website_content(company_url)
    # print("Fetching company reviews...")
    # company_reviews = get_reviews(company_product_name, RAPIDAPI_KEY)

    print("Scraping competitor website...")
    competitor_web_text = scrape_website_content(competitor_url)
    # print("Fetching competitor reviews...")
    # competitor_reviews = get_reviews(competitor_product_name, RAPIDAPI_KEY)

    main_document_path = filedialog.askopenfilename(title="Select Main Document", filetypes=[("PDF files", "*.pdf")])
    if not main_document_path:
        print("No main document selected.")
        return

    file_paths = list(filedialog.askopenfilenames(title="Select Other Documents", filetypes=[("All files", "*.*")]))
    if not file_paths:
        print("No other documents selected.")
        return

    loader = CustomTextLoader()
    company_web_doc = Document(page_content=company_web_text, metadata={"source": "company_web"})
    # company_reviews_doc = Document(page_content='\n\n'.join([f"Title: {r['title']}\nReviewer: {r['reviewer']}\nPositive: {r['positive']}\nNegative: {r['negative']}\nOverall: {r['overall']}" for r in company_reviews]), metadata={"source": "company_reviews"})
    competitor_web_doc = Document(page_content=competitor_web_text, metadata={"source": "competitor_web"})
    # competitor_reviews_doc = Document(page_content='\n\n'.join([f"Title: {r['title']}\nReviewer: {r['reviewer']}\nPositive: {r['positive']}\nNegative: {r['negative']}\nOverall: {r['overall']}" for r in competitor_reviews]), metadata={"source": "competitor_reviews"})

    file_paths.extend([company_web_doc,
                       # company_reviews_doc,
                       competitor_web_doc,
                       # competitor_reviews_doc
                       ])

    save_path = "doc_embeddings.pkl"
    start_time = time.time()
    qa_chain, doc_similarities, main_doc_embedding, all_chunks = process_documents(main_document_path, file_paths, save_path)
    print(f"Documents processed in {time.time() - start_time:.2f} seconds")

    if qa_chain:
        embedding = OpenAIEmbeddings()
        
        start_time = time.time()
        doc_embeddings = embed_documents_in_batches([doc for doc, _ in doc_similarities], embedding, batch_size=10)
        print(f"Embeddings collected in {time.time() - start_time:.2f} seconds")

        query1 = ("What jobs are open at plum?")
        queries = [query1]

        start_time = time.time()
        responses = get_llm_responses(queries, doc_similarities)
        print(f"LLM responses retrieved in {time.time() - start_time:.2f} seconds")

        for i, response in enumerate(responses):
            card_text = response["result"]
            print(f"Response {i+1}:")
            print(card_text)

            sources = response["source_documents"]
            print("\nSources:")
            for source in sources:
                print(f"Source: {source.metadata['source']}")

            print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()