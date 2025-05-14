import os
import openai
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

base_url="https://api.groq.com/openai/v1"
api_key = os.environ.get("API_KEY")
client = openai.OpenAI(api_key = api_key,
                       base_url = base_url)
model = "gemma2-9b-it"

system_prompt = """
你是個商業分析的專家。
"""

prompt_template = """
根據下列資料回答問題：
{retrieved_chunks}

使用者的問題是：{question}

請根據資料內容回覆。
"""

chat_history = []

class CustomE5Embedding(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")

def chat_with_rag(user_input: str, retriever, prompt_template: str, system_prompt: str, model: str, chat_history: List):
    # 取回相關資料
    docs = retriever.get_relevant_documents(user_input)
    retrieved_chunks = "\n\n".join([doc.page_content for doc in docs])

    # 將自定 prompt 套入格式
    final_prompt = prompt_template.format(retrieved_chunks=retrieved_chunks, question=user_input)

    # 呼叫 OpenAI API
    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_prompt},
    ]
    )
    answer = response.choices[0].message.content

    chat_history.append((user_input, answer))
    return answer

if __name__ == "__main__":
    embedding_model = CustomE5Embedding(model_name="intfloat/multilingual-e5-small")
    db = FAISS.load_local("./uploaded_docs/faiss_db", embedding_model, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    while True:
        query = input("👤 使用者：")
        if query.lower() in ["exit", "quit"]:
            break
        answer = chat_with_rag(query, retriever, prompt_template, system_prompt, model, chat_history)
        print("🤖 助理：", answer)