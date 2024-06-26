import os
import time
import tkinter as tk
from tkinter import filedialog
from dotenv import load_dotenv
from web_scraping import scrape_website_content
from reviews_api import get_reviews
from document_processing import process_documents, CustomTextLoader, Document, embed_documents_in_batches
from qa_chain import get_llm_responses
from langchain_community.embeddings import OpenAIEmbeddings


class OnboardingHandler:
    # unitl you save vectort db - croma
    def __init__(self, customer_name, custom_json):
        self.customer_name = customer_name
        self.custom_json = custom_json
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.rapidapi_key = os.getenv("RAPIDAPI_KEY")
        self.custom_text_loader = CustomTextLoader()
        self.root = tk.Tk()
        self.root.withdraw()  # Hides the main window as we only need the file dialog functionality

    def onboard_customer(self):
        print(f"Starting onboarding for {self.customer_name}")

        company_url = self.custom_json.get('company_url')
        company_product_name = self.custom_json.get('company_product_name')
        competitor_url = self.custom_json.get('competitor_url')
        competitor_product_name = self.custom_json.get('competitor_product_name')

        print("Scraping company website...")
        company_web_text = scrape_website_content(company_url)
        print("Fetching company reviews...")
        company_reviews = get_reviews(company_product_name, self.rapidapi_key)

        print("Scraping competitor website...")
        competitor_web_text = scrape_website_content(competitor_url)
        print("Fetching competitor reviews...")
        competitor_reviews = get_reviews(competitor_product_name, self.rapidapi_key)

        main_document_path = filedialog.askopenfilename(title="Select Main Document",
                                                        filetypes=[("PDF files", "*.pdf")])
        if not main_document_path:
            print("No main document selected.")
            return

        file_paths = list(filedialog.askopenfilenames(title="Select Other Documents", filetypes=[("All files", "*.*")]))
        if not file_paths:
            print("No other documents selected.")
            return

        company_web_doc = Document(page_content=company_web_text, metadata={"source": "company_web"})
        company_reviews_doc = Document(page_content='\n\n'.join([
                                                                    f"Title: {r['title']}\nReviewer: {r['reviewer']}\nPositive: {r['positive']}\nNegative: {r['negative']}\nOverall: {r['overall']}"
                                                                    for r in company_reviews]),
                                       metadata={"source": "company_reviews"})
        competitor_web_doc = Document(page_content=competitor_web_text, metadata={"source": "competitor_web"})
        competitor_reviews_doc = Document(page_content='\n\n'.join([
                                                                       f"Title: {r['title']}\nReviewer: {r['reviewer']}\nPositive: {r['positive']}\nNegative: {r['negative']}\nOverall: {r['overall']}"
                                                                       for r in competitor_reviews]),
                                          metadata={"source": "competitor_reviews"})

        file_paths.extend([company_web_doc, company_reviews_doc, competitor_web_doc, competitor_reviews_doc])

        save_path = "doc_embeddings.pkl"
        start_time = time.time()
        qa_chain, doc_similarities, main_doc_embedding, all_chunks = process_documents(main_document_path, file_paths,
                                                                                       save_path)
        print(f"Documents processed in {time.time() - start_time:.2f} seconds")

        ## mongo store vector



