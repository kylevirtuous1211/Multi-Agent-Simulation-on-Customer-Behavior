"""
消費者行為模擬系統 - 消費者資料庫模組
處理消費者資料的存儲和管理
"""

import sqlite3
import json
import os
import logging
from datetime import datetime

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConsumerDatabase:
    """消費者資料庫類，處理消費者資料的存儲和管理"""
    
    def __init__(self, db_path='data/consumer_db.sqlite'):
        """初始化消費者資料庫
        
        Args:
            db_path: 資料庫文件路徑
        """
        # 確保資料目錄存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
        # 初始化資料庫
        self._init_db()
    
    def _connect(self):
        """連接到資料庫"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def _disconnect(self):
        """關閉資料庫連接"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def _init_db(self):
        """初始化資料庫表結構"""
        try:
            self._connect()
            
            # 創建消費者表
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS consumers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                demographic TEXT,
                behavioral TEXT,
                psychographic TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            self.conn.commit()
            logger.info("資料庫初始化成功")
        except Exception as e:
            logger.error(f"資料庫初始化失敗: {str(e)}")
            raise
        finally:
            self._disconnect()
    
    def add_consumer(self, demographic=None, behavioral=None, psychographic=None):
        """添加消費者資料
        
        Args:
            demographic: 人口統計資料
            behavioral: 行為資料
            psychographic: 心理統計資料
            
        Returns:
            新添加的消費者ID
        """
        try:
            self._connect()
            
            # 將資料轉換為JSON字符串
            demographic_json = json.dumps(demographic or {}, ensure_ascii=False)
            behavioral_json = json.dumps(behavioral or {}, ensure_ascii=False)
            psychographic_json = json.dumps(psychographic or {}, ensure_ascii=False)
            
            # 插入資料
            self.cursor.execute('''
            INSERT INTO consumers (demographic, behavioral, psychographic, created_at, updated_at)
            VALUES (?, ?, ?, datetime('now'), datetime('now'))
            ''', (demographic_json, behavioral_json, psychographic_json))
            
            self.conn.commit()
            consumer_id = self.cursor.lastrowid
            
            logger.info(f"添加消費者成功，ID: {consumer_id}")
            return consumer_id
        except Exception as e:
            logger.error(f"添加消費者失敗: {str(e)}")
            raise
        finally:
            self._disconnect()
    
    def get_consumer(self, consumer_id):
        """獲取特定消費者資料
        
        Args:
            consumer_id: 消費者ID
            
        Returns:
            消費者資料字典
        """
        try:
            self._connect()
            
            # 查詢資料
            self.cursor.execute('''
            SELECT id, demographic, behavioral, psychographic, created_at, updated_at
            FROM consumers
            WHERE id = ?
            ''', (consumer_id,))
            
            row = self.cursor.fetchone()
            
            if not row:
                logger.warning(f"找不到消費者 {consumer_id}")
                return None
            
            # 解析JSON資料
            consumer = {
                'id': row['id'],
                'demographic': json.loads(row['demographic']),
                'behavioral': json.loads(row['behavioral']),
                'psychographic': json.loads(row['psychographic']),
                'created_at': row['created_at'],
                'updated_at': row['updated_at']
            }
            
            return consumer
        except Exception as e:
            logger.error(f"獲取消費者 {consumer_id} 失敗: {str(e)}")
            raise
        finally:
            self._disconnect()
    
    def update_consumer(self, consumer_id, demographic=None, behavioral=None, psychographic=None):
        """更新消費者資料
        
        Args:
            consumer_id: 消費者ID
            demographic: 人口統計資料
            behavioral: 行為資料
            psychographic: 心理統計資料
            
        Returns:
            是否更新成功
        """
        try:
            self._connect()
            
            # 檢查消費者是否存在
            self.cursor.execute('SELECT id FROM consumers WHERE id = ?', (consumer_id,))
            if not self.cursor.fetchone():
                logger.warning(f"找不到消費者 {consumer_id}")
                return False
            
            # 獲取現有資料
            current = self.get_consumer(consumer_id)
            
            # 合併資料
            if demographic is not None:
                current['demographic'].update(demographic)
            if behavioral is not None:
                current['behavioral'].update(behavioral)
            if psychographic is not None:
                current['psychographic'].update(psychographic)
            
            # 將資料轉換為JSON字符串
            demographic_json = json.dumps(current['demographic'], ensure_ascii=False)
            behavioral_json = json.dumps(current['behavioral'], ensure_ascii=False)
            psychographic_json = json.dumps(current['psychographic'], ensure_ascii=False)
            
            # 更新資料
            self.cursor.execute('''
            UPDATE consumers
            SET demographic = ?, behavioral = ?, psychographic = ?, updated_at = datetime('now')
            WHERE id = ?
            ''', (demographic_json, behavioral_json, psychographic_json, consumer_id))
            
            self.conn.commit()
            
            logger.info(f"更新消費者 {consumer_id} 成功")
            return True
        except Exception as e:
            logger.error(f"更新消費者 {consumer_id} 失敗: {str(e)}")
            raise
        finally:
            self._disconnect()
    
    def delete_consumer(self, consumer_id):
        """刪除消費者資料
        
        Args:
            consumer_id: 消費者ID
            
        Returns:
            是否刪除成功
        """
        try:
            self._connect()
            
            # 檢查消費者是否存在
            self.cursor.execute('SELECT id FROM consumers WHERE id = ?', (consumer_id,))
            if not self.cursor.fetchone():
                logger.warning(f"找不到消費者 {consumer_id}")
                return False
            
            # 刪除資料
            self.cursor.execute('DELETE FROM consumers WHERE id = ?', (consumer_id,))
            
            self.conn.commit()
            
            logger.info(f"刪除消費者 {consumer_id} 成功")
            return True
        except Exception as e:
            logger.error(f"刪除消費者 {consumer_id} 失敗: {str(e)}")
            raise
        finally:
            self._disconnect()
    
    def get_all_consumers(self, limit=None, offset=0):
        """獲取所有消費者資料
        
        Args:
            limit: 限制返回的記錄數
            offset: 起始偏移量
            
        Returns:
            消費者資料列表
        """
        try:
            self._connect()
            
            # 構建查詢
            query = '''
            SELECT id, demographic, behavioral, psychographic, created_at, updated_at
            FROM consumers
            ORDER BY id
            '''
            
            params = []
            
            if limit is not None:
                query += ' LIMIT ?'
                params.append(limit)
                
                if offset > 0:
                    query += ' OFFSET ?'
                    params.append(offset)
            
            # 查詢資料
            self.cursor.execute(query, params)
            
            rows = self.cursor.fetchall()
            
            # 解析JSON資料
            consumers = []
            for row in rows:
                consumer = {
                    'id': row['id'],
                    'demographic': json.loads(row['demographic']),
                    'behavioral': json.loads(row['behavioral']),
                    'psychographic': json.loads(row['psychographic']),
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                }
                consumers.append(consumer)
            
            return consumers
        except Exception as e:
            logger.error(f"獲取所有消費者失敗: {str(e)}")
            raise
        finally:
            self._disconnect()
    
    def search_consumers(self, query):
        """搜索消費者資料
        
        Args:
            query: 搜索關鍵詞
            
        Returns:
            匹配的消費者資料列表
        """
        try:
            self._connect()
            
            # 構建模糊查詢
            search_term = f"%{query}%"
            
            # 查詢資料
            self.cursor.execute('''
            SELECT id, demographic, behavioral, psychographic, created_at, updated_at
            FROM consumers
            WHERE demographic LIKE ? OR behavioral LIKE ? OR psychographic LIKE ?
            ORDER BY id
            ''', (search_term, search_term, search_term))
            
            rows = self.cursor.fetchall()
            
            # 解析JSON資料
            consumers = []
            for row in rows:
                consumer = {
                    'id': row['id'],
                    'demographic': json.loads(row['demographic']),
                    'behavioral': json.loads(row['behavioral']),
                    'psychographic': json.loads(row['psychographic']),
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                }
                consumers.append(consumer)
            
            return consumers
        except Exception as e:
            logger.error(f"搜索消費者失敗: {str(e)}")
            raise
        finally:
            self._disconnect()
    
    def get_consumer_count(self):
        """獲取消費者總數
        
        Returns:
            消費者總數
        """
        try:
            self._connect()
            
            # 查詢總數
            self.cursor.execute('SELECT COUNT(*) as count FROM consumers')
            
            row = self.cursor.fetchone()
            
            return row['count']
        except Exception as e:
            logger.error(f"獲取消費者總數失敗: {str(e)}")
            raise
        finally:
            self._disconnect()
    
    def import_consumers(self, consumers):
        """批量導入消費者資料
        
        Args:
            consumers: 消費者資料列表
            
        Returns:
            成功導入的消費者數量
        """
        try:
            self._connect()
            
            count = 0
            
            # 開始事務
            self.conn.execute('BEGIN TRANSACTION')
            
            for consumer in consumers:
                demographic = consumer.get('demographic', {})
                behavioral = consumer.get('behavioral', {})
                psychographic = consumer.get('psychographic', {})
                
                # 將資料轉換為JSON字符串
                demographic_json = json.dumps(demographic, ensure_ascii=False)
                behavioral_json = json.dumps(behavioral, ensure_ascii=False)
                psychographic_json = json.dumps(psychographic, ensure_ascii=False)
                
                # 插入資料
                self.cursor.execute('''
                INSERT INTO consumers (demographic, behavioral, psychographic, created_at, updated_at)
                VALUES (?, ?, ?, datetime('now'), datetime('now'))
                ''', (demographic_json, behavioral_json, psychographic_json))
                
                count += 1
            
            # 提交事務
            self.conn.commit()
            
            logger.info(f"批量導入 {count} 個消費者成功")
            return count
        except Exception as e:
            # 回滾事務
            if self.conn:
                self.conn.rollback()
            
            logger.error(f"批量導入消費者失敗: {str(e)}")
            raise
        finally:
            self._disconnect()
    
    def export_consumers(self, consumer_ids=None):
        """導出消費者資料
        
        Args:
            consumer_ids: 要導出的消費者ID列表，為None時導出所有
            
        Returns:
            消費者資料列表
        """
        try:
            self._connect()
            
            if consumer_ids:
                # 構建IN查詢的參數
                placeholders = ','.join(['?'] * len(consumer_ids))
                
                # 查詢特定消費者
                self.cursor.execute(f'''
                SELECT id, demographic, behavioral, psychographic, created_at, updated_at
                FROM consumers
                WHERE id IN ({placeholders})
                ORDER BY id
                ''', consumer_ids)
            else:
                # 查詢所有消費者
                self.cursor.execute('''
                SELECT id, demographic, behavioral, psychographic, created_at, updated_at
                FROM consumers
                ORDER BY id
                ''')
            
            rows = self.cursor.fetchall()
            
            # 解析JSON資料
            consumers = []
            for row in rows:
                consumer = {
                    'id': row['id'],
                    'demographic': json.loads(row['demographic']),
                    'behavioral': json.loads(row['behavioral']),
                    'psychographic': json.loads(row['psychographic']),
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                }
                consumers.append(consumer)
            
            return consumers
        except Exception as e:
            logger.error(f"導出消費者失敗: {str(e)}")
            raise
        finally:
            self._disconnect()
