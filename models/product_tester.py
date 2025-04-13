"""
消費者行為模擬系統 - 產品測試模組
實現產品測試和代理反應分析
"""

import os
import json
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductTester:
    """產品測試器，實現產品測試和代理反應分析"""
    
    def __init__(self):
        """初始化產品測試器"""
        self.agent_manager = None  # 將在外部設置
        
        # 確保輸出目錄存在
        os.makedirs('static/images', exist_ok=True)
        os.makedirs('static/uploads', exist_ok=True)
    
    def test_product(self, product_info, agent_ids=None):
        """測試產品
        
        Args:
            product_info: 產品信息
            agent_ids: 代理ID列表，為None時使用所有代理
            
        Returns:
            測試結果
        """
        if not self.agent_manager:
            logger.warning("代理管理器未設置")
            return None
        
        # 模擬產品評估
        results = self.agent_manager.simulate_product_evaluation(product_info, agent_ids)
        
        # 分析結果
        analysis = self.analyze_results(results)
        
        # 生成報告
        report = self.generate_report(product_info, results, analysis)
        
        return {
            'results': results,
            'analysis': analysis,
            'report': report
        }
    
    def analyze_results(self, results):
        """分析測試結果
        
        Args:
            results: 測試結果
            
        Returns:
            分析結果
        """
        if not results:
            return None
        
        # 統計情感分佈
        sentiment_counts = {
            'positive': 0,
            'neutral': 0,
            'negative': 0
        }
        
        # 統計購買意願分佈
        purchase_intent_counts = {
            'high': 0,
            'medium': 0,
            'low': 0
        }
        
        # 收集反饋
        feedback_list = []
        
        # 處理每個代理的結果
        for agent_id, result in results.items():
            # 統計情感
            sentiment = result.get('sentiment')
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            
            # 統計購買意願
            purchase_intent = result.get('purchase_intent')
            if purchase_intent in purchase_intent_counts:
                purchase_intent_counts[purchase_intent] += 1
            
            # 收集反饋
            feedback = result.get('feedback')
            if feedback:
                feedback_list.append({
                    'agent_id': agent_id,
                    'feedback': feedback
                })
        
        # 計算總數
        total_agents = len(results)
        
        # 計算百分比
        sentiment_percentages = {}
        for sentiment, count in sentiment_counts.items():
            sentiment_percentages[sentiment] = round(count / total_agents * 100, 2) if total_agents > 0 else 0
        
        purchase_intent_percentages = {}
        for intent, count in purchase_intent_counts.items():
            purchase_intent_percentages[intent] = round(count / total_agents * 100, 2) if total_agents > 0 else 0
        
        # 生成情感分佈圖
        sentiment_chart = self._generate_sentiment_chart(sentiment_counts)
        
        # 生成購買意願分佈圖
        purchase_intent_chart = self._generate_purchase_intent_chart(purchase_intent_counts)
        
        # 構建分析結果
        analysis = {
            'total_agents': total_agents,
            'sentiment_counts': sentiment_counts,
            'sentiment_percentages': sentiment_percentages,
            'purchase_intent_counts': purchase_intent_counts,
            'purchase_intent_percentages': purchase_intent_percentages,
            'feedback_list': feedback_list,
            'sentiment_chart': sentiment_chart,
            'purchase_intent_chart': purchase_intent_chart
        }
        
        return analysis
    
    def _generate_sentiment_chart(self, sentiment_counts):
        """生成情感分佈圖
        
        Args:
            sentiment_counts: 情感計數
            
        Returns:
            圖表文件路徑
        """
        try:
            # 提取數據
            labels = list(sentiment_counts.keys())
            values = list(sentiment_counts.values())
            
            # 設置顏色
            colors = ['#4CAF50', '#FFC107', '#F44336']
            
            # 創建圖表
            plt.figure(figsize=(8, 6))
            plt.bar(labels, values, color=colors)
            
            # 添加標題和標籤
            plt.title('情感分佈')
            plt.xlabel('情感')
            plt.ylabel('代理數量')
            
            # 添加數值標籤
            for i, v in enumerate(values):
                plt.text(i, v + 0.1, str(v), ha='center')
            
            # 保存圖表
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'static/images/sentiment_chart_{timestamp}.png'
            plt.savefig(output_path)
            plt.close()
            
            return output_path
        except Exception as e:
            logger.error(f"生成情感分佈圖失敗: {str(e)}")
            return None
    
    def _generate_purchase_intent_chart(self, purchase_intent_counts):
        """生成購買意願分佈圖
        
        Args:
            purchase_intent_counts: 購買意願計數
            
        Returns:
            圖表文件路徑
        """
        try:
            # 提取數據
            labels = list(purchase_intent_counts.keys())
            values = list(purchase_intent_counts.values())
            
            # 設置顏色
            colors = ['#4CAF50', '#FFC107', '#F44336']
            
            # 創建圖表
            plt.figure(figsize=(8, 6))
            plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            
            # 添加標題
            plt.title('購買意願分佈')
            
            # 保存圖表
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'static/images/purchase_intent_chart_{timestamp}.png'
            plt.savefig(output_path)
            plt.close()
            
            return output_path
        except Exception as e:
            logger.error(f"生成購買意願分佈圖失敗: {str(e)}")
            return None
    
    def generate_report(self, product_info, results, analysis):
        """生成測試報告
        
        Args:
            product_info: 產品信息
            results: 測試結果
            analysis: 分析結果
            
        Returns:
            報告內容
        """
        if not analysis:
            return "無法生成報告：分析結果為空"
        
        # 提取產品信息
        product_name = product_info.get('name', '未命名產品')
        product_category = product_info.get('category', '未分類')
        product_price = product_info.get('price', '未定價')
        product_features = product_info.get('features', [])
        product_description = product_info.get('description', '')
        
        # 提取分析結果
        total_agents = analysis.get('total_agents', 0)
        sentiment_percentages = analysis.get('sentiment_percentages', {})
        purchase_intent_percentages = analysis.get('purchase_intent_percentages', {})
        feedback_list = analysis.get('feedback_list', [])
        
        # 生成報告
        report = f"""
# {product_name} 消費者反應測試報告

## 產品信息

- **名稱**: {product_name}
- **類別**: {product_category}
- **價格**: {product_price}
- **特性**: {', '.join(product_features)}
- **描述**: {product_description}

## 測試概況

- **測試時間**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **參與代理數**: {total_agents}

## 情感分析

消費者對產品的情感反應:

- **正面**: {sentiment_percentages.get('positive', 0)}%
- **中性**: {sentiment_percentages.get('neutral', 0)}%
- **負面**: {sentiment_percentages.get('negative', 0)}%

## 購買意願分析

消費者的購買意願:

- **高**: {purchase_intent_percentages.get('high', 0)}%
- **中**: {purchase_intent_percentages.get('medium', 0)}%
- **低**: {purchase_intent_percentages.get('low', 0)}%

## 消費者反饋摘要

"""
        
        # 添加反饋
        for i, feedback in enumerate(feedback_list[:10]):  # 最多顯示10條反饋
            report += f"{i+1}. {feedback.get('feedback')}\n"
        
        # 添加結論
        positive_sentiment = sentiment_percentages.get('positive', 0)
        high_intent = purchase_intent_percentages.get('high', 0)
        
        if positive_sentiment > 60 and high_intent > 50:
            conclusion = "產品獲得了非常積極的反應，大多數消費者表示有較高的購買意願。建議進行市場推廣。"
        elif positive_sentiment > 40 and high_intent > 30:
            conclusion = "產品獲得了較為積極的反應，有相當比例的消費者表示有購買意願。建議進一步優化產品後推向市場。"
        elif positive_sentiment > 20 and high_intent > 10:
            conclusion = "產品獲得了一定程度的積極反應，但購買意願不高。建議重新評估產品定位和特性。"
        else:
            conclusion = "產品獲得的積極反應有限，購買意願較低。建議重新設計產品或調整目標市場。"
        
        report += f"""
## 結論與建議

{conclusion}

---
報告生成時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report
    
    def save_product_image(self, image_file):
        """保存產品圖片
        
        Args:
            image_file: 圖片文件
            
        Returns:
            保存路徑
        """
        try:
            # 生成文件名
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'product_{timestamp}.jpg'
            save_path = os.path.join('static/uploads', filename)
            
            # 保存圖片
            image_file.save(save_path)
            
            return save_path
        except Exception as e:
            logger.error(f"保存產品圖片失敗: {str(e)}")
            return None
    
    def process_product_image(self, image_path):
        """處理產品圖片
        
        Args:
            image_path: 圖片路徑
            
        Returns:
            處理後的圖片路徑
        """
        try:
            # 打開圖片
            img = Image.open(image_path)
            
            # 調整大小
            max_size = (800, 800)
            img.thumbnail(max_size, Image.LANCZOS)
            
            # 保存處理後的圖片
            processed_path = image_path.replace('.', '_processed.')
            img.save(processed_path)
            
            return processed_path
        except Exception as e:
            logger.error(f"處理產品圖片失敗: {str(e)}")
            return image_path
