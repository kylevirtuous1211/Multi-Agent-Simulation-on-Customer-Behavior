"""
消費者行為模擬系統 - 分類系統模組
實現消費者分類和聚類分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
import logging
import json
from datetime import datetime

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsumerClassifier:
    """消費者分類系統，實現消費者分類和聚類分析"""
    
    def __init__(self):
        """初始化分類系統"""
        self.db = None  # 將在外部設置
        self.features = None
        self.labels = None
        self.model = None
        self.model_type = None
        self.scaler = StandardScaler()
        
        # 確保輸出目錄存在
        os.makedirs('static/images', exist_ok=True)
    
    def preprocess_data(self):
        """預處理消費者資料，提取特徵
        
        Returns:
            特徵矩陣，消費者ID列表，特徵名稱列表
        """
        try:
            # 獲取所有消費者資料
            consumers = self.db.get_all_consumers()
            
            if not consumers:
                logger.warning("沒有消費者資料可供分類")
                return np.array([]), [], []
            
            # 提取特徵
            features_list = []
            consumer_ids = []
            
            for consumer in consumers:
                # 提取數值特徵
                features = []
                
                # 人口統計特徵
                features.append(consumer['demographic'].get('age', 0) or 0)
                features.append(1 if consumer['demographic'].get('gender') == 'male' else 0)
                features.append(2 if consumer['demographic'].get('gender') == 'female' else 0)
                features.append(consumer['demographic'].get('income', 0) or 0)
                
                # 行為特徵
                features.append(consumer['behavioral'].get('purchase_frequency', 0) or 0)
                features.append(consumer['behavioral'].get('brand_loyalty', 0) or 0)
                features.append(consumer['behavioral'].get('price_consciousness', 0) or 0)
                features.append(consumer['behavioral'].get('tech_savviness', 0) or 0)
                features.append(consumer['behavioral'].get('social_media_usage', 0) or 0)
                
                # 心理統計特徵
                features.append(1 if consumer['psychographic'].get('personality') == 'introvert' else 0)
                features.append(1 if consumer['psychographic'].get('personality') == 'extrovert' else 0)
                features.append(consumer['psychographic'].get('innovativeness', 0) or 0)
                features.append(consumer['psychographic'].get('social_influence', 0) or 0)
                
                features_list.append(features)
                consumer_ids.append(consumer['id'])
            
            # 特徵名稱
            feature_names = [
                'age', 'gender_male', 'gender_female', 'income',
                'purchase_frequency', 'brand_loyalty', 'price_consciousness', 'tech_savviness', 'social_media_usage',
                'personality_introvert', 'personality_extrovert', 'innovativeness', 'social_influence'
            ]
            
            # 轉換為numpy數組
            X = np.array(features_list)
            
            # 標準化特徵
            if X.shape[0] > 0:
                X = self.scaler.fit_transform(X)
            
            return X, consumer_ids, feature_names
        except Exception as e:
            logger.error(f"預處理資料失敗: {str(e)}")
            raise
    
    def train_kmeans(self, n_clusters=5):
        """訓練K-means聚類模型
        
        Args:
            n_clusters: 聚類數量
            
        Returns:
            聚類標籤
        """
        try:
            # 預處理資料
            X, consumer_ids, feature_names = self.preprocess_data()
            
            if X.shape[0] == 0:
                logger.warning("沒有資料可供訓練")
                return np.array([])
            
            # 訓練模型
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X)
            
            # 保存模型和結果
            self.model = kmeans
            self.model_type = 'kmeans'
            self.features = X
            self.labels = labels
            self.consumer_ids = consumer_ids
            self.feature_names = feature_names
            
            logger.info(f"K-means聚類完成，共{n_clusters}個聚類")
            
            return labels
        except Exception as e:
            logger.error(f"訓練K-means模型失敗: {str(e)}")
            raise
    
    def train_dbscan(self, eps=0.5, min_samples=5):
        """訓練DBSCAN聚類模型
        
        Args:
            eps: 鄰域半徑
            min_samples: 最小樣本數
            
        Returns:
            聚類標籤
        """
        try:
            # 預處理資料
            X, consumer_ids, feature_names = self.preprocess_data()
            
            if X.shape[0] == 0:
                logger.warning("沒有資料可供訓練")
                return np.array([])
            
            # 訓練模型
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # 保存模型和結果
            self.model = dbscan
            self.model_type = 'dbscan'
            self.features = X
            self.labels = labels
            self.consumer_ids = consumer_ids
            self.feature_names = feature_names
            
            # 計算聚類數量（不包括噪聲點）
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info(f"DBSCAN聚類完成，共{n_clusters}個聚類")
            
            return labels
        except Exception as e:
            logger.error(f"訓練DBSCAN模型失敗: {str(e)}")
            raise
    
    def train_hierarchical(self, n_clusters=5):
        """訓練層次聚類模型
        
        Args:
            n_clusters: 聚類數量
            
        Returns:
            聚類標籤
        """
        try:
            # 預處理資料
            X, consumer_ids, feature_names = self.preprocess_data()
            
            if X.shape[0] == 0:
                logger.warning("沒有資料可供訓練")
                return np.array([])
            
            # 訓練模型
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            labels = hierarchical.fit_predict(X)
            
            # 保存模型和結果
            self.model = hierarchical
            self.model_type = 'hierarchical'
            self.features = X
            self.labels = labels
            self.consumer_ids = consumer_ids
            self.feature_names = feature_names
            
            logger.info(f"層次聚類完成，共{n_clusters}個聚類")
            
            return labels
        except Exception as e:
            logger.error(f"訓練層次聚類模型失敗: {str(e)}")
            raise
    
    def analyze_clusters(self):
        """分析聚類結果
        
        Returns:
            聚類統計資訊
        """
        try:
            if self.labels is None or self.features is None:
                logger.warning("沒有聚類結果可供分析")
                return {}
            
            # 計算每個聚類的樣本數
            cluster_counts = {}
            for label in set(self.labels):
                count = np.sum(self.labels == label)
                cluster_counts[int(label)] = int(count)
            
            # 計算每個聚類的中心點
            cluster_centers = {}
            for label in set(self.labels):
                if label == -1:  # DBSCAN的噪聲點
                    continue
                
                mask = self.labels == label
                cluster_data = self.features[mask]
                center = np.mean(cluster_data, axis=0)
                
                # 將中心點轉換回原始特徵空間
                if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
                    center = center * self.scaler.scale_ + self.scaler.mean_
                
                cluster_centers[int(label)] = center.tolist()
            
            # 計算每個聚類的特徵統計
            cluster_stats = {}
            for label in set(self.labels):
                if label == -1:  # DBSCAN的噪聲點
                    continue
                
                mask = self.labels == label
                cluster_data = self.features[mask]
                
                # 計算統計量
                mean = np.mean(cluster_data, axis=0)
                std = np.std(cluster_data, axis=0)
                
                # 將統計量轉換回原始特徵空間
                if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
                    mean = mean * self.scaler.scale_ + self.scaler.mean_
                    std = std * self.scaler.scale_
                
                # 構建特徵統計
                feature_stats = {}
                for i, name in enumerate(self.feature_names):
                    feature_stats[name] = {
                        'mean': float(mean[i]),
                        'std': float(std[i])
                    }
                
                cluster_stats[int(label)] = feature_stats
            
            # 獲取每個聚類的消費者ID
            cluster_consumers = {}
            for label in set(self.labels):
                mask = self.labels == label
                consumer_ids = [self.consumer_ids[i] for i, m in enumerate(mask) if m]
                cluster_consumers[int(label)] = consumer_ids
            
            # 構建結果
            result = {
                'counts': cluster_counts,
                'centers': cluster_centers,
                'stats': cluster_stats,
                'consumers': cluster_consumers
            }
            
            return result
        except Exception as e:
            logger.error(f"分析聚類結果失敗: {str(e)}")
            raise
    
    def visualize_clusters(self, method='pca'):
        """視覺化聚類結果
        
        Args:
            method: 降維方法，支持'pca'
            
        Returns:
            圖像文件路徑
        """
        try:
            if self.labels is None or self.features is None:
                logger.warning("沒有聚類結果可供視覺化")
                return None
            
            # 降維
            if method == 'pca':
                pca = PCA(n_components=2)
                X_reduced = pca.fit_transform(self.features)
                
                # 創建圖像
                plt.figure(figsize=(10, 8))
                
                # 繪製散點圖
                for label in set(self.labels):
                    mask = self.labels == label
                    if label == -1:  # DBSCAN的噪聲點
                        plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], label='噪聲', color='gray', alpha=0.5)
                    else:
                        plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], label=f'聚類 {label}', alpha=0.7)
                
                # 添加標題和圖例
                plt.title(f'{self.model_type.upper()} 聚類結果')
                plt.xlabel('主成分 1')
                plt.ylabel('主成分 2')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # 保存圖像
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f'static/images/cluster_{self.model_type}_{timestamp}.png'
                plt.savefig(output_path)
                plt.close()
                
                logger.info(f"聚類視覺化完成，保存至 {output_path}")
                
                return output_path
            else:
                logger.warning(f"不支持的降維方法: {method}")
                return None
        except Exception as e:
            logger.error(f"視覺化聚類結果失敗: {str(e)}")
            raise
    
    def find_optimal_clusters(self, max_clusters=10, X=None):
        """尋找最佳聚類數量
        
        Args:
            max_clusters: 最大聚類數量
            X: 特徵矩陣，為None時使用預處理後的資料
            
        Returns:
            最佳聚類數量，輪廓係數列表
        """
        try:
            if X is None:
                # 預處理資料
                X, _, _ = self.preprocess_data()
            
            if X.shape[0] == 0:
                logger.warning("沒有資料可供分析")
                return 2, np.array([])
            
            # 至少需要2個樣本
            if X.shape[0] < 2:
                logger.warning("樣本數量不足，無法計算輪廓係數")
                return 2, np.array([])
            
            # 計算不同聚類數量的輪廓係數
            silhouette_scores = []
            
            # 從2個聚類開始
            for n_clusters in range(2, min(max_clusters + 1, X.shape[0])):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(X)
                
                # 計算輪廓係數
                score = silhouette_score(X, labels)
                silhouette_scores.append(score)
            
            # 找到最佳聚類數量
            optimal_clusters = np.argmax(silhouette_scores) + 2  # +2是因為從2個聚類開始
            
            logger.info(f"最佳聚類數量: {optimal_clusters}")
            
            return optimal_clusters, np.array(silhouette_scores)
        except Exception as e:
            logger.error(f"尋找最佳聚類數量失敗: {str(e)}")
            raise
    
    def get_consumer_cluster(self, consumer_id):
        """獲取消費者所屬的聚類
        
        Args:
            consumer_id: 消費者ID
            
        Returns:
            聚類標籤
        """
        try:
            if self.labels is None or self.consumer_ids is None:
                logger.warning("沒有聚類結果")
                return None
            
            # 查找消費者索引
            try:
                index = self.consumer_ids.index(consumer_id)
                return int(self.labels[index])
            except ValueError:
                logger.warning(f"找不到消費者 {consumer_id} 的聚類")
                return None
        except Exception as e:
            logger.error(f"獲取消費者聚類失敗: {str(e)}")
            raise
    
    def get_cluster_consumers(self, cluster_label):
        """獲取聚類中的所有消費者
        
        Args:
            cluster_label: 聚類標籤
            
        Returns:
            消費者ID列表
        """
        try:
            if self.labels is None or self.consumer_ids is None:
                logger.warning("沒有聚類結果")
                return []
            
            # 查找聚類中的消費者
            consumer_ids = []
            for i, label in enumerate(self.labels):
                if label == cluster_label:
                    consumer_ids.append(self.consumer_ids[i])
            
            return consumer_ids
        except Exception as e:
            logger.error(f"獲取聚類消費者失敗: {str(e)}")
            raise
    
    def predict_cluster(self, consumer_data):
        """預測新消費者的聚類
        
        Args:
            consumer_data: 消費者資料
            
        Returns:
            聚類標籤
        """
        try:
            if self.model is None:
                logger.warning("沒有訓練好的模型")
                return None
            
            # 提取特徵
            features = []
            
            # 人口統計特徵
            features.append(consumer_data['demographic'].get('age', 0) or 0)
            features.append(1 if consumer_data['demographic'].get('gender') == 'male' else 0)
            features.append(2 if consumer_data['demographic'].get('gender') == 'female' else 0)
            features.append(consumer_data['demographic'].get('income', 0) or 0)
            
            # 行為特徵
            features.append(consumer_data['behavioral'].get('purchase_frequency', 0) or 0)
            features.append(consumer_data['behavioral'].get('brand_loyalty', 0) or 0)
            features.append(consumer_data['behavioral'].get('price_consciousness', 0) or 0)
            features.append(consumer_data['behavioral'].get('tech_savviness', 0) or 0)
            features.append(consumer_data['behavioral'].get('social_media_usage', 0) or 0)
            
            # 心理統計特徵
            features.append(1 if consumer_data['psychographic'].get('personality') == 'introvert' else 0)
            features.append(1 if consumer_data['psychographic'].get('personality') == 'extrovert' else 0)
            features.append(consumer_data['psychographic'].get('innovativeness', 0) or 0)
            features.append(consumer_data['psychographic'].get('social_influence', 0) or 0)
            
            # 轉換為numpy數組
            X = np.array([features])
            
            # 標準化特徵
            X = self.scaler.transform(X)
            
            # 預測聚類
            if self.model_type == 'kmeans':
                label = self.model.predict(X)[0]
            elif self.model_type == 'dbscan':
                # DBSCAN不支持predict，使用最近鄰居的標籤
                distances = np.sqrt(np.sum((self.features - X) ** 2, axis=1))
                nearest_idx = np.argmin(distances)
                label = self.labels[nearest_idx]
            elif self.model_type == 'hierarchical':
                # 層次聚類不支持predict，使用最近鄰居的標籤
                distances = np.sqrt(np.sum((self.features - X) ** 2, axis=1))
                nearest_idx = np.argmin(distances)
                label = self.labels[nearest_idx]
            else:
                logger.warning(f"不支持的模型類型: {self.model_type}")
                return None
            
            return int(label)
        except Exception as e:
            logger.error(f"預測聚類失敗: {str(e)}")
            raise
