import os
import openai
import json
import requests
from bs4 import BeautifulSoup
import asyncio
from typing import List
from googlesearch import search as _search
from requests_html import AsyncHTMLSession
from bs4 import BeautifulSoup
from charset_normalizer import from_bytes

base_url = "https://api.groq.com/openai/v1"
api_key = os.environ.get("API_KEY")
client = openai.OpenAI(api_key=api_key, base_url=base_url)
model = "gemma2-9b-it"

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search_agent",
            "description": "搜尋網頁並回傳純文字內容",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "n_results": {"type": "integer", "default": 1}
                },
                "required": ["keyword"]
            }
        }
    }
]

# -------- 網頁抓取 Worker --------
async def worker(s: AsyncHTMLSession, url: str) -> str | None:
    try:
        # 先看 headers 確保是 HTML 頁面
        header_response = await asyncio.wait_for(s.head(url, verify=True), timeout=10)
        if 'text/html' not in header_response.headers.get('Content-Type', ''):
            return None
        
        # 抓取完整 HTML 頁面
        response = await asyncio.wait_for(s.get(url, verify=True), timeout=10)
        return response.content.decode(response.encoding or 'utf-8', errors='ignore')

    except Exception as e:
        print(f"[Error] 無法抓取 {url}：{e}")
        return None

# -------- 批次抓取網頁 HTML --------
async def get_htmls(urls: List[str]) -> List[str]:
    session = AsyncHTMLSession()
    tasks = [worker(session, url) for url in urls]
    return await asyncio.gather(*tasks)

# -------- 主搜尋 Agent --------
async def web_search_agent(keyword: str, n_results: int = 1) -> List[str]:
    '''
    非同步網頁搜尋與清洗文字，適用於特徵萃取或語意分析。
    '''
    keyword = keyword[:100]
    urls = list(_search(keyword, num_results=n_results * 2, lang="zh", unique=True))
    htmls = await get_htmls(urls)
    htmls = [x for x in htmls if x is not None]

    results = []
    for html in htmls:
        try:
            encoding = from_bytes(html.encode()).best().encoding
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            results.append(''.join(text.split()))
        except Exception as e:
            continue

    return results[:n_results]

# prompt template
def build_prompt(feature: str) -> str:
    return f"""
    你是一位產品經理助理，請使用 RICE 模型為下列 feature 做評估，並給出每一項指標（1-5 分）與理由，最後計算 priority 分數。

    請依以下格式回覆：
    Feature: <描述>
    Reach: <1-5 分 + 理由>
    Impact: <1-5 分 + 理由>
    Confidence: <1-5 分 + 理由>
    Effort: <1-5 分 + 理由>
    Priority Score: <計算值：Reach x Impact x Confidence / Effort>
    Reason Summary: <綜合排序依據>

    Feature: {feature}
    """

# 呼叫 GPT 模型分析每個 feature
def analyze_feature(feature: str) -> dict:
    prompt = build_prompt(feature)

    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        tools=tools,
        tool_choice="auto",
        messages=[
            {"role": "system", "content": "你是一位產品經理助理，熟悉 feature 評估與 RICE 模型。"},
            {"role": "user", "content": prompt}
        ]
    )

    message = response.choices[0].message
    tool_calls = message.tool_calls

    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            print("arguments:", arguments)

            if function_name == "web_search_agent":
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(web_search_agent(**arguments))
                
            print(len("\n".join(result)))
            # 再包成工具的回應訊息
            tool_response_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "\n".join(result)[0:min(1024,len("\n".join(result)))]  # ⬅️ 這是你的工具回傳的內容
                # "content": "\n".join(map(str, result)) if isinstance(result, list) else str(result)
            }

            # 丟回 GPT 做下一步推理
            second_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一位產品經理助理，熟悉 feature 評估與 RICE 模型。"},
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "tool_calls": [tool_call]  # 呼叫紀錄
                    },
                    tool_response_message
                ]
            )

            # print("\n🟢 GPT 回應：")
            # print(second_response.choices[0].message.content)

    return second_response.choices[0].message.content