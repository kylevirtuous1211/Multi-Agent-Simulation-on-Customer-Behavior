"""
測試模型: intfloat/multilingual-e5-large-instruct、BAAI/bge-m3
排行榜: https://huggingface.co/spaces/mteb/leaderboard

pip install langchain==0.3.25 langchain-community==0.3.23 pypdf==5.4.0 python-docx==1.1.2 sentence-transformers==4.1.0 faiss-cpu==1.11.0
pip install --no-cache-dir --force-reinstall sentence-transformers
pip uninstall sentence-transformers torch
pip install sentence-transformers torch
"""
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class CustomE5Embedding(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")

def GetDocument(upload_dir="./uploaded_docs"):
    folder_path = upload_dir
    documents = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if file.endswith(".txt"):
            loader = TextLoader(path)
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(path)
        else:
            continue
        documents.extend(loader.load())

    return documents

def main(database_dir="./uploaded_docs"):

    documents = GetDocument(database_dir)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    embedding_model = CustomE5Embedding(model_name="intfloat/multilingual-e5-small")
    vectorstore = FAISS.from_documents(split_docs, embedding_model)

    vectorstore.save_local("faiss_db")

if __name__ == "__main__":
    main()