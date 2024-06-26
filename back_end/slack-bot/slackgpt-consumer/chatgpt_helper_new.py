import os
from dotenv import load_dotenv
import openai
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from typing import List

load_dotenv()

OpenAI_API_Key: str = os.environ.get("OPENAI_API_KEY")
if not OpenAI_API_Key:
    raise ValueError("OpenAI API Key not found in environment variables.")
openai.api_key = OpenAI_API_Key


class EmptyRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str) -> List:
        return []


def get_llm_response(message: str) -> str:
    llm = OpenAI(max_tokens=1024)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=EmptyRetriever(),
        return_source_documents=False
    )

    llm_response = qa_chain.run(message)
    return llm_response


message = "Tell me a fun fact."
response = get_llm_response(message)
print(f"ChatGPT Response: {response}")
