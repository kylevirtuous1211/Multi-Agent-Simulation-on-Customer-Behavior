"""
消費者行為模擬系統 - 生成式代理模組
基於"Generative Agents: Interactive Simulacra of Human Behavior"論文實現的代理架構
"""

import os
import json
import logging
import uuid
import numpy as np
import datetime
import random
from collections import deque
import openai  # Import OpenAI module

# Set up OpenAI API key (ensure this is securely stored in your environment)
openai.api_key = os.getenv("OPENAI_API_KEY")

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Memory:
    """記憶系統，管理代理的記憶"""
    
    def __init__(self, capacity=1000):
        """初始化記憶系統
        
        Args:
            capacity: 記憶容量
        """
        self.memories = []
        self.capacity = capacity
    
    def add(self, content, importance=0.5, source="observation", related_memories=None):
        """添加記憶
        
        Args:
            content: 記憶內容
            importance: 重要性 (0-1)
            source: 記憶來源
            related_memories: 相關記憶ID列表
            
        Returns:
            記憶ID
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        memory = {
            "id": memory_id,
            "content": content,
            "importance": importance,
            "timestamp": timestamp,
            "source": source,
            "related_memories": related_memories or [],
            "last_accessed": timestamp,
            "access_count": 0
        }
        
        self.memories.append(memory)
        
        # 如果超過容量，移除最不重要的記憶
        if len(self.memories) > self.capacity:
            self.memories.sort(key=lambda x: x["importance"])
            self.memories.pop(0)
        
        return memory_id
    
        
    def get_recent(self, limit=10):
        """獲取最近的記憶
        
        Args:
            limit: 返回結果數量限制
            
        Returns:
            最近的記憶列表
        """
        # 按時間戳排序
        recent = sorted(self.memories, key=lambda x: x["timestamp"], reverse=True)
        
        return recent[:limit]
    
    def decay(self, rate=0.01):
        """記憶衰減
        
        Args:
            rate: 衰減率
        """
        now = datetime.datetime.now()
        
        for memory in self.memories:
            # 計算時間差（天）
            created = datetime.datetime.fromisoformat(memory["timestamp"])
            days_passed = (now - created).days
            
            # 根據時間和訪問次數計算衰減
            decay_factor = rate * days_passed / (memory["access_count"] + 1)
            
            # 應用衰減
            memory["importance"] = max(0.1, memory["importance"] - decay_factor)
    
    def save(self, path):
        """保存記憶到文件
        
        Args:
            path: 文件路徑
            
        Returns:
            是否保存成功
        """
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"保存記憶失敗: {str(e)}")
            return False
    
    def load(self, path):
        """從文件加載記憶
        
        Args:
            path: 文件路徑
            
        Returns:
            是否加載成功
        """
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.memories = json.load(f)
                return True
            return False
        except Exception as e:
            logger.error(f"加載記憶失敗: {str(e)}")
            return False


class Reflection:
    """反思機制，對記憶進行反思，形成更高層次的見解"""

    def __init__(self, memory):
        """初始化反思機制
        
        Args:
            memory: 記憶系統
        """
        self.memory = memory
        self.insights = []
    
    def reflect(self, trigger="periodic", related_memories=None):
        """進行反思
        
        Args:
            trigger: 觸發反思的原因
            related_memories: 相關記憶ID列表
            
        Returns:
            反思結果
        """
        # 獲取相關記憶
        memories_to_reflect = []
        
        if related_memories:
            for memory_id in related_memories:
                memory = self.memory.get(memory_id)
                if memory:
                    memories_to_reflect.append(memory)
        else:
            # 使用最近和最重要的記憶
            memories_to_reflect = self.memory.get_recent(5) + self.memory.get_important(5)
        
        if not memories_to_reflect:
            return None
        
        # 提取記憶內容
        memory_contents = [m["content"] for m in memories_to_reflect]
        
        # 生成反思內容
        insight = self._generate_insight(memory_contents, trigger)
        
        # 保存反思結果
        insight_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        insight_obj = {
            "id": insight_id,
            "content": insight,
            "timestamp": timestamp,
            "trigger": trigger,
            "related_memories": [m["id"] for m in memories_to_reflect]
        }
        
        self.insights.append(insight_obj)
        
        # 將反思結果添加到記憶中
        self.memory.add(
            content=f"reflection: {insight}",
            importance=0.8,
            source="reflection",
            related_memories=[m["id"] for m in memories_to_reflect]
        )
        
        return insight_obj
    
    def _generate_insight(self, memory_contents, trigger):
        """Generate insights using OpenAI API.

        Args:
            memory_contents: List of memory contents.
            trigger: Reason for triggering reflection.

        Returns:
            Generated insight content.
        """
        if not memory_contents:
            return "No sufficient memories available for reflection."

        # Prepare the prompt for the OpenAI API
        prompt = (
            f"You are an intelligent agent reflecting on your experiences. "
            f"Here are some memories: {memory_contents}. "
            f"The reflection is triggered by: {trigger}. "
            f"Generate a meaningful insight based on these memories."
        )

        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a reflective AI agent."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.9
            )
            # Extract the generated insight
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"Error generating insight: {str(e)}")
            return "Error generating insight."

    def get_recent_insights(self, limit=5):
        """獲取最近的反思
        
        Args:
            limit: 返回結果數量限制
            
        Returns:
            最近的反思列表
        """
        # 按時間戳排序
        recent = sorted(self.insights, key=lambda x: x["timestamp"], reverse=True)
        
        return recent[:limit]
    
    def save(self, path):
        """保存反思到文件
        
        Args:
            path: 文件路徑
            
        Returns:
            是否保存成功
        """
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.insights, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"保存反思失敗: {str(e)}")
            return False
    
    def load(self, path):
        """從文件加載反思
        
        Args:
            path: 文件路徑
            
        Returns:
            是否加載成功
        """
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.insights = json.load(f)
                return True
            return False
        except Exception as e:
            logger.error(f"加載反思失敗: {str(e)}")
            return False


class Interaction:
    """互動能力，與其他代理和用戶進行對話互動"""
    
    def __init__(self, memory, reflection):
        """初始化互動能力
        
        Args:
            memory: 記憶系統
            reflection: 反思機制
        """
        self.memory = memory
        self.reflection = reflection
        self.conversation_history = []
    
    def receive_message(self, sender, message):
        """接收消息
        
        Args:
            sender: 發送者
            message: 消息內容
            
        Returns:
            回覆消息
        """
        # 記錄消息
        timestamp = datetime.datetime.now().isoformat()
        
        message_obj = {
            "sender": sender,
            "content": message,
            "timestamp": timestamp
        }
        
        self.conversation_history.append(message_obj)
        
        # 將消息添加到記憶中
        self.memory.add(
            content=f"{sender}說: {message}",
            importance=0.6,
            source="conversation"
        )
        
        # 生成回覆
        reply = self._generate_reply(sender, message)
        
        # 記錄回覆
        reply_obj = {
            "sender": "agent",
            "content": reply,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.conversation_history.append(reply_obj)
        
        # 將回覆添加到記憶中
        self.memory.add(
            content=f"我回覆{sender}: {reply}",
            importance=0.6,
            source="conversation"
        )
        
        # 觸發反思
        if len(self.conversation_history) % 5 == 0:
            self.reflection.reflect(trigger="conversation")
        
        return reply
    
    def _generate_reply(self, sender, message):
        """Generate replies using OpenAI API.

        Args:
            sender: Sender of the message.
            message: Message content.

        Returns:
            Generated reply content.
        """
        # Prepare the prompt for the OpenAI API
        prompt = (
            f"You are an intelligent agent having a conversation. "
            f"The user said: '{message}'. "
            f"Generate an appropriate and thoughtful reply."
        )

        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a conversational AI agent."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.9
            )
            # Extract the generated reply
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"Error generating reply: {str(e)}")
            return "Error generating reply."

    def get_conversation_history(self, limit=None):
        """獲取對話歷史
        
        Args:
            limit: 返回結果數量限制
            
        Returns:
            對話歷史列表
        """
        if limit:
            return self.conversation_history[-limit:]
        else:
            return self.conversation_history
    
    def save(self, path):
        """保存對話歷史到文件
        
        Args:
            path: 文件路徑
            
        Returns:
            是否保存成功
        """
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"保存對話歷史失敗: {str(e)}")
            return False
    
    def load(self, path):
        """從文件加載對話歷史
        
        Args:
            path: 文件路徑
            
        Returns:
            是否加載成功
        """
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                return True
            return False
        except Exception as e:
            logger.error(f"加載對話歷史失敗: {str(e)}")
            return False


class GenerativeAgent:
    """生成式代理，模擬人類行為和決策過程"""
    
    def __init__(self, agent_id=None, profile=None):
        """初始化生成式代理
        
        Args:
            agent_id: 代理ID，為None時自動生成
            profile: 代理資料
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.profile = profile or {}
        
        # 初始化各個組件
        self.memory = Memory()
        self.reflection = Reflection(self.memory)
        self.interaction = Interaction(self.memory, self.reflection)
        
        # 初始化代理狀態
        self.state = {
            "mood": "neutral",  # 情緒: positive, neutral, negative
            "energy": 0.9,      # 能量: 0-1
            "focus": 0.5        # 專注度: 0-1
        }
        
        # 初始化代理統計
        self.stats = {
            "created_at": datetime.datetime.now().isoformat(),
            "last_active": datetime.datetime.now().isoformat(),
            "memory_count": 0,
            "reflection_count": 0,
            "conversation_count": 0
        }
    
    def update(self):
        """更新代理狀態"""
        # 更新統計
        self.stats["memory_count"] = len(self.memory.memories)
        self.stats["reflection_count"] = len(self.reflection.insights)
        self.stats["conversation_count"] = len(self.interaction.conversation_history)
        self.stats["last_active"] = datetime.datetime.now().isoformat()
        
        # 記憶衰減
        self.memory.decay()
        
        # 定期反思
        if random.random() < 0.2:  # 20%的概率進行反思
            self.reflection.reflect(trigger="periodic")
    
    def evaluate_product(self, product_info):
        """評估產品
        
        Args:
            product_info: 產品信息
            
        Returns:
            評估結果
        """
        # 提取產品信息
        name = product_info.get("name", "")
        category = product_info.get("category", "")
        price = product_info.get("price", "")
        features = product_info.get("features", [])
        description = product_info.get("description", "")
        
        # 將產品信息添加到記憶中
        self.memory.add(
            content=f"評估產品: {name}，類別: {category}，價格: {price}",
            importance=0.9,
            source="product_evaluation"
        )
        
        # 根據代理資料和記憶生成評估
        sentiment = self._evaluate_sentiment(product_info)
        purchase_intent = self._evaluate_purchase_intent(product_info)
        detailed_feedback = self._generate_detailed_feedback(product_info)
        
        # 將評估結果添加到記憶中
        self.memory.add(
            content=f"對產品 {name} 的評估: 情感={sentiment}, 購買意願={purchase_intent}",
            importance=0.9,
            source="product_evaluation"
        )
        
        # 觸發反思
        self.reflection.reflect(trigger="product_evaluation")
        
        # 更新代理狀態
        self.update()
        
        return {
            "sentiment": sentiment,
            "purchase_intent": purchase_intent,
            "feedback": detailed_feedback
        }
    
    def _evaluate_sentiment(self, product_info):
        """Evaluate sentiment using OpenAI API.

        Args:
            product_info: Product information.

        Returns:
            Sentiment: positive, neutral, negative.
        """
        # Prepare the prompt for the OpenAI API
        prompt = (
            f"You are an intelligent agent evaluating a product. "
            f"Here is the product information: {json.dumps(product_info)}. "
            f"Based on this information, determine the sentiment (positive, neutral, or negative) "
            f"and explain your reasoning."
        )

        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI agent specializing in product evaluation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.9
            )
            # Extract the sentiment from the response
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"Error evaluating sentiment: {str(e)}")
            return "Error evaluating sentiment."
    
    def _evaluate_purchase_intent(self, product_info):
        """Evaluate purchase intent using OpenAI API.

        Args:
            product_info: Product information.

        Returns:
            Purchase intent: high, medium, low.
        """
        # Prepare the prompt for the OpenAI API
        prompt = (
            f"You are an intelligent agent evaluating a product's purchase intent. "
            f"Here is the product information: {json.dumps(product_info)}. "
            f"Based on this information, determine the purchase intent (high, medium, or low) "
            f"and explain your reasoning."
        )

        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI agent specializing in consumer behavior."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.9
            )
            # Extract the purchase intent from the response
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"Error evaluating purchase intent: {str(e)}")
            return "Error evaluating purchase intent."
    
    def _generate_detailed_feedback(self, product_info):
        """Generate detailed feedback using OpenAI API.

        Args:
            product_info: Product information.

        Returns:
            Detailed feedback.
        """
        # Prepare the prompt for the OpenAI API
        prompt = (
            f"You are an intelligent agent providing detailed feedback on a product. "
            f"Here is the product information: {json.dumps(product_info)}. "
            f"Generate a detailed and thoughtful evaluation of the product, including its strengths and weaknesses."
        )

        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI agent specializing in product evaluation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.9
            )
            # Extract the detailed feedback from the response
            return response['choices'][0]['message']['content'].strip()
        
        except Exception as e:
            logger.error(f"Error generating detailed feedback: {str(e)}")
            return "Error generating detailed feedback."
        
    def save(self, base_dir="data/agents"):
        """保存代理到文件
        
        Args:
            base_dir: 基礎目錄
            
        Returns:
            是否保存成功
        """
        try:
            # 確保目錄存在
            agent_dir = os.path.join(base_dir, self.agent_id)
            os.makedirs(agent_dir, exist_ok=True)
            
            # 保存代理資料
            with open(os.path.join(agent_dir, "profile.json"), 'w', encoding='utf-8') as f:
                json.dump(self.profile, f, ensure_ascii=False, indent=2)
            
            # 保存代理狀態
            with open(os.path.join(agent_dir, "state.json"), 'w', encoding='utf-8') as f:
                json.dump({
                    "state": self.state,
                    "stats": self.stats
                }, f, ensure_ascii=False, indent=2)
            
            # 保存各個組件
            self.memory.save(os.path.join(agent_dir, "memory.json"))
            self.reflection.save(os.path.join(agent_dir, "reflection.json"))
            self.interaction.save(os.path.join(agent_dir, "interaction.json"))
            
            logger.info(f"代理 {self.agent_id} 保存成功")
            return True
        except Exception as e:
            logger.error(f"保存代理 {self.agent_id} 失敗: {str(e)}")
            return False
    
    @classmethod
    def load(cls, agent_id, base_dir="data/agents"):
        """從文件加載代理
        
        Args:
            agent_id: 代理ID
            base_dir: 基礎目錄
            
        Returns:
            代理實例
        """
        try:
            agent_dir = os.path.join(base_dir, agent_id)
            
            if not os.path.exists(agent_dir):
                logger.warning(f"找不到代理 {agent_id} 的目錄")
                return None
            
            # 加載代理資料
            with open(os.path.join(agent_dir, "profile.json"), 'r', encoding='utf-8') as f:
                profile = json.load(f)
            
            # 創建代理實例
            agent = cls(agent_id=agent_id, profile=profile)
            
            # 加載代理狀態
            with open(os.path.join(agent_dir, "state.json"), 'r', encoding='utf-8') as f:
                state_data = json.load(f)
                agent.state = state_data.get("state", agent.state)
                agent.stats = state_data.get("stats", agent.stats)
            
            # 加載各個組件
            agent.memory.load(os.path.join(agent_dir, "memory.json"))
            agent.reflection.load(os.path.join(agent_dir, "reflection.json"))
            agent.interaction.load(os.path.join(agent_dir, "interaction.json"))
            
            logger.info(f"代理 {agent_id} 加載成功")
            return agent
        except Exception as e:
            logger.error(f"加載代理 {agent_id} 失敗: {str(e)}")
            return None


class AgentManager:
    """代理管理器，管理多個代理"""
    
    def __init__(self):
        """初始化代理管理器"""
        self.agents = {}
        self.db = None  # 將在外部設置
        
        # 確保代理目錄存在
        os.makedirs("data/agents", exist_ok=True)
    
    def create_agent(self, profile):
        """創建代理
        
        Args:
            profile: 代理資料
            
        Returns:
            代理ID
        """
        agent = GenerativeAgent(profile=profile)
        self.agents[agent.agent_id] = agent
        
        # 保存代理
        agent.save()
        
        return agent.agent_id
    
    def create_agent_from_consumer(self, consumer_id):
        """從消費者資料創建代理
        
        Args:
            consumer_id: 消費者ID
            
        Returns:
            代理ID
        """
        # 獲取消費者資料
        consumer = self.db.get_consumer(consumer_id)
        
        if not consumer:
            logger.warning(f"找不到消費者 {consumer_id}")
            return None
        
        # 創建代理資料
        profile = {
            "consumer_id": consumer_id,
            "demographic": consumer["demographic"],
            "behavioral": consumer["behavioral"],
            "psychographic": consumer["psychographic"]
        }
        
        # 創建代理
        agent_id = self.create_agent(profile)
        
        # 初始化代理記憶
        agent = self.agents[agent_id]
        
        # 添加基本記憶
        agent.memory.add(
            content=f"我是一個{consumer['demographic'].get('age', '成年')}歲的{consumer['demographic'].get('gender', '人')}，職業是{consumer['demographic'].get('occupation', '未知')}。",
            importance=0.8,
            source="profile"
        )
        
        agent.memory.add(
            content=f"我的收入是{consumer['demographic'].get('income', '未知')}，教育程度是{consumer['demographic'].get('education', '未知')}。",
            importance=0.9,
            source="profile"
        )
        
        agent.memory.add(
            content=f"我的購買頻率是{consumer['behavioral'].get('purchase_frequency', '未知')}，品牌忠誠度是{consumer['behavioral'].get('brand_loyalty', '未知')}。",
            importance=0.9,
            source="profile"
        )
        
        agent.memory.add(
            content=f"我的性格是{consumer['psychographic'].get('personality', '未知')}，興趣是{consumer['psychographic'].get('interests', '未知')}。",
            importance=0.8,
            source="profile"
        )
        
        # 保存代理
        agent.save()
        
        return agent_id
    
    def create_agents_from_group(self, group):
        """從分類群組創建代理
        
        Args:
            group: 分類群組
            
        Returns:
            代理ID列表
        """
        # 獲取分類系統
        from models.classification import ConsumerClassifier
        classifier = ConsumerClassifier()
        classifier.db = self.db
        
        # 獲取群組中的消費者
        consumer_ids = classifier.get_cluster_consumers(group)
        
        if not consumer_ids:
            logger.warning(f"找不到群組 {group} 中的消費者")
            return []
        
        # 創建代理
        agent_ids = []
        for consumer_id in consumer_ids:
            agent_id = self.create_agent_from_consumer(consumer_id)
            if agent_id:
                agent_ids.append(agent_id)
        
        return agent_ids
    
    def get_agent(self, agent_id):
        """獲取代理
        
        Args:
            agent_id: 代理ID
            
        Returns:
            代理實例
        """
        # 檢查是否已加載
        if agent_id in self.agents:
            return self.agents[agent_id]
        
        # 嘗試從文件加載
        agent = GenerativeAgent.load(agent_id)
        
        if agent:
            self.agents[agent_id] = agent
            return agent
        
        return None
    
    def get_all_agents(self):
        """獲取所有代理ID
        
        Returns:
            代理ID列表
        """
        # 掃描代理目錄
        agent_ids = []
        
        if os.path.exists("data/agents"):
            for agent_id in os.listdir("data/agents"):
                if os.path.isdir(os.path.join("data/agents", agent_id)):
                    agent_ids.append(agent_id)
        
        return agent_ids
    
    def get_agent_summary(self, agent_id):
        """獲取代理摘要
        
        Args:
            agent_id: 代理ID
            
        Returns:
            代理摘要
        """
        agent = self.get_agent(agent_id)
        
        if not agent:
            return None
        
        # 獲取基本資料
        profile = agent.profile
        
        # 獲取最近記憶
        recent_memories = agent.memory.get_recent(5)
        
        # 獲取最近反思
        recent_insights = agent.reflection.get_recent_insights(3)
        
        # 構建摘要
        summary = {
            "agent_id": agent_id,
            "profile": profile,
            "state": agent.state,
            "stats": agent.stats,
            "recent_memories": recent_memories,
            "recent_insights": recent_insights,
            "active_goals": active_goals
        }
        
        return summary
    
    def delete_agent(self, agent_id):
        """刪除代理
        
        Args:
            agent_id: 代理ID
            
        Returns:
            是否刪除成功
        """
        # 從內存中移除
        if agent_id in self.agents:
            del self.agents[agent_id]
        
        # 刪除文件
        agent_dir = os.path.join("data/agents", agent_id)
        
        if os.path.exists(agent_dir):
            try:
                import shutil
                shutil.rmtree(agent_dir)
                return True
            except Exception as e:
                logger.error(f"刪除代理 {agent_id} 文件失敗: {str(e)}")
                return False
        
        return True
    
    def simulate_product_evaluation(self, product_info, agent_ids=None):
        """模擬產品評估
        
        Args:
            product_info: 產品信息
            agent_ids: 代理ID列表，為None時使用所有代理
            
        Returns:
            評估結果字典
        """
        if agent_ids is None:
            agent_ids = self.get_all_agents()
        
        results = {}
        
        for agent_id in agent_ids:
            agent = self.get_agent(agent_id)
            
            if agent:
                # 評估產品
                evaluation = agent.evaluate_product(product_info)
                
                # 保存結果
                results[agent_id] = evaluation
                
                # 保存代理
                agent.save()
        
        return results
