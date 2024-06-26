import os
import time  # Ensure this import is included
import tkinter as tk
from tkinter import filedialog
from dotenv import load_dotenv
from web_scraping import scrape_website_content
from reviews_api import get_reviews
from document_processing import process_documents
from qa_chain import get_llm_responses
from langchain_community.embeddings import OpenAIEmbeddings  # Correct import
from qa_chain import update_conversation_history

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

def main():
    root = tk.Tk()
    root.withdraw()

    # company_url = input("Enter the company's website URL: ")
    # company_product_name = input("Enter the company's product name: ")
    # competitor_url = input("Enter the competitor's website URL: ")
    # competitor_product_name = input("Enter the competitor's product name: ")
    #
    # print("Scraping company website...")
    # # company_web_text = scrape_website_content(company_url)
    # # print("Fetching company reviews...")
    # # company_reviews = get_reviews(company_product_name, RAPIDAPI_KEY)
    #
    # print("Scraping competitor website...")
    # # competitor_web_text = scrape_website_content(competitor_url)
    # # print("Fetching competitor reviews...")
    # # competitor_reviews = get_reviews(competitor_product_name, RAPIDAPI_KEY)
    
    main_document_path = filedialog.askopenfilename(title="Select Main Document", filetypes=[("All files", "*.*")])
    if not main_document_path:
        print("No main document selected.")
        return

    file_paths = list(filedialog.askopenfilenames(title="Select Other Documents", filetypes=[("All files", "*.*")]))
    if not file_paths:
        print("No other documents selected.")
        return

    save_path = "doc_embeddings.pkl"
    start_time = time.time()
    qa_chain, doc_similarities, main_doc_embedding, all_chunks = process_documents(main_document_path, file_paths, save_path)
    print(f"Documents processed in {time.time() - start_time:.2f} seconds")

    conversation_history = []
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        conversation_history = update_conversation_history(conversation_history, user_query)
        queries = [user_query]

        start_time = time.time()
        responses = get_llm_responses(queries, conversation_history)
        print(f"LLM responses retrieved in {time.time() - start_time:.2f} seconds")

        for i, response in enumerate(responses):
            card_text = response["result"]
            relevant_contents = response["content"]
            confidence_score = response["confidence_score"]
            confidence_level = response["confidence_level"]

            print(f"Response {i+1}:")
            print(card_text)
            print(f"Confidence Score: {confidence_score:.2f} ({confidence_level})")
            print("Relevant Content:")
            for content in relevant_contents:
                print(content)
                print("="*50)

            conversation_history = update_conversation_history(conversation_history, card_text)

if __name__ == "__main__":
    main()
