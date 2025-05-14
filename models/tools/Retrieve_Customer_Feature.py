"""
pip install requests beautifulsoup4 requests_html lxml_html_clean
"""

import os
import openai
import json
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import asyncio
from typing import List
from googlesearch import search as _search
from requests_html import AsyncHTMLSession
from bs4 import BeautifulSoup
from charset_normalizer import from_bytes
from Priority_Customer_Feature import analyze_feature

# 請替換為你自己的 OpenAI API 金鑰
base_url = "https://api.groq.com/openai/v1"
api_key = os.environ.get("API_KEY")
client = openai.OpenAI(api_key=api_key, base_url=base_url)
model = "gemma2-9b-it"

def extract_user_features_via_ai(user_input: str, model: str):
    """
    使用 GPT 模型來提取輸入內容的特徵。

    參數:
        user_input (str): 使用者輸入的描述或文字
        model (str): 使用的 GPT 模型，預設為 llama3-70b-8192

    回傳:
        str: GPT 回傳的特徵摘要或結構化資料
    """
    system_prompt = (
        "你是一個資訊萃取專家，負責從使用者的描述中找出他們所有的關鍵特徵，並用 JSON 格式輸出。\n"
        "例如：\n"
        "輸入：「我叫做 Alan，是一位來自英國的數學家，二戰期間破解了德國密碼，我設計了圖靈機。」\n"
        "輸出：{\n"
        "  \"name\": \"Alan\",\n"
        "  \"country\": \"UK\",\n"
        "  \"profession\": \"Mathematician\",\n"
        "  \"known_for\": \"Turing Machine, Enigma Codebreaking\"\n"
        "}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ 呼叫 AI 失敗：{e}"

# 範例：使用者輸入查詢
if __name__ == "__main__":
    # user_text = input("請輸入要查詢的關鍵字：")
    feature_list = user_text = ["幸福", "愛情"]
    # feature = extract_user_features_via_ai(user_text)
    print("\n擷取到的特徵：\n", feature_list)

    # 逐一處理所有 feature
    for feature in feature_list:
        print("==============================================")
        print(analyze_feature(feature))
    
    # search_results = web_search_agent(feature)
    # search_results = asyncio.run(web_search_agent(feature, n_results=1))
    
    # print("\n✅ 搜尋結果：\n")
    # for idx, result in enumerate(search_results, 1):
    #     print(f"{idx}. {result['title']}\n{result['snippet']}\n{result['link']}\n")
    # for i, content in enumerate(search_results, 1):
    #     print(f"\n--- 第 {i} 筆結果 ---\n{content[:500]}...\n")