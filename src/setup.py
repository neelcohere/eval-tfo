import os
from typing import Any, Dict, List

from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.pdf import PyMuPDFLoader

from dotenv import load_dotenv
load_dotenv()

docs = []
folder = "data"
urls = [
    "https://drive.google.com/file/d/1iequwjaDGzfScm79IWpKDLTr9IdFTuTr/view?usp=drive_link",
    "https://drive.google.com/file/d/1i2P9YL5R6-eHuJaWUpnbwYw6ITzrqfWf/view?usp=drive_link"
]
files = ["investment-1.pdf", "investment-2.pdf"]
for file, url in zip(files, urls):
    print(file)
    print(url)
    filepath = os.path.join(folder, file)
    loader = PyMuPDFLoader(filepath)
    _docs = loader.load()
    for doc in _docs:
        # remove "~" char to avoid strike through formatting in response
        doc.page_content = doc.page_content.replace("~", "")
        doc.metadata["source"] = url
        doc.metadata["title"] = file + f" (pg. {doc.metadata.get('page', '')})"
    docs.extend(_docs)
    print(docs[0])

db = FAISS.from_documents(
    docs,
    CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"), model="embed-english-v3.0")
)

db.save_local("data/memo_index")