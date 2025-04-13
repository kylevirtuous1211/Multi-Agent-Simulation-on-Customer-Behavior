"""
消費者行為模擬系統 - CRM API整合模組
實現與企業CRM系統的API連接
"""

import requests
import json
import logging
import os
from datetime import datetime

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CRMConnector:
    """CRM連接器，實現與企業CRM系統的API連接"""
    
    def __init__(self, api_url=None, api_key=None):
        """初始化CRM連接器
        
        Args:
            api_url: API基礎URL
            api_key: API密鑰
        """
        self.api_url = api_url
        self.api_key = api_key
        self.db = None  # 將在外部設置
        
        # 確保日誌目錄存在
        os.makedirs('data/logs', exist_ok=True)
        
        # 設置日誌文件
        self.log_file = f'data/logs/crm_api_{datetime.now().strftime("%Y%m%d")}.log'
        
        # 添加文件處理器
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def test_connection(self):
        """測試API連接
        
        Returns:
            是否連接成功
        """
        if not self.api_url or not self.api_key:
            logger.warning("API URL或API密鑰未設置")
            return False
        
        try:
            # 構建請求頭
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # 發送測試請求
            response = requests.get(f'{self.api_url}/test', headers=headers, timeout=10)
            
            # 檢查響應
            if response.status_code == 200:
                logger.info("API連接測試成功")
                return True
            else:
                logger.warning(f"API連接測試失敗: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"API連接測試異常: {str(e)}")
            return False
    
    def configure(self, api_url, api_key):
        """配置API連接
        
        Args:
            api_url: API基礎URL
            api_key: API密鑰
            
        Returns:
            是否配置成功
        """
        self.api_url = api_url
        self.api_key = api_key
        
        # 測試連接
        return self.test_connection()
    
    def import_customers(self, limit=100):
        """從CRM系統導入客戶資料
        
        Args:
            limit: 導入數量限制
            
        Returns:
            導入的客戶數量
        """
        if not self.api_url or not self.api_key:
            logger.warning("API URL或API密鑰未設置")
            return 0
        
        if not self.db:
            logger.warning("資料庫未設置")
            return 0
        
        try:
            # 構建請求頭
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # 構建請求參數
            params = {
                'limit': limit
            }
            
            # 發送請求
            response = requests.get(f'{self.api_url}/customers', headers=headers, params=params, timeout=30)
            
            # 檢查響應
            if response.status_code == 200:
                customers = response.json()
                
                # 轉換為消費者資料格式
                consumers = []
                
                for customer in customers:
                    # 提取人口統計資料
                    demographic = {
                        'age': customer.get('age'),
                        'gender': customer.get('gender'),
                        'income': customer.get('income'),
                        'education': customer.get('education'),
                        'occupation': customer.get('occupation'),
                        'location': customer.get('location')
                    }
                    
                    # 提取行為資料
                    behavioral = {
                        'purchase_frequency': customer.get('purchase_frequency'),
                        'brand_loyalty': customer.get('brand_loyalty'),
                        'price_consciousness': customer.get('price_consciousness'),
                        'tech_savviness': customer.get('tech_savviness'),
                        'social_media_usage': customer.get('social_media_usage')
                    }
                    
                    # 提取心理統計資料
                    psychographic = {
                        'personality': customer.get('personality'),
                        'interests': customer.get('interests'),
                        'values': customer.get('values'),
                        'lifestyle': customer.get('lifestyle'),
                        'innovativeness': customer.get('innovativeness'),
                        'social_influence': customer.get('social_influence')
                    }
                    
                    # 構建消費者資料
                    consumer = {
                        'demographic': demographic,
                        'behavioral': behavioral,
                        'psychographic': psychographic
                    }
                    
                    consumers.append(consumer)
                
                # 導入到資料庫
                count = self.db.import_consumers(consumers)
                
                logger.info(f"從CRM系統導入了 {count} 個客戶")
                return count
            else:
                logger.warning(f"導入客戶失敗: {response.status_code} - {response.text}")
                return 0
        except Exception as e:
            logger.error(f"導入客戶異常: {str(e)}")
            return 0
    
    def export_simulation_results(self, product_info, results):
        """將模擬結果導出到CRM系統
        
        Args:
            product_info: 產品信息
            results: 模擬結果
            
        Returns:
            是否導出成功
        """
        if not self.api_url or not self.api_key:
            logger.warning("API URL或API密鑰未設置")
            return False
        
        try:
            # 構建請求頭
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # 構建請求數據
            data = {
                'product': product_info,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            # 發送請求
            response = requests.post(f'{self.api_url}/simulations', headers=headers, json=data, timeout=30)
            
            # 檢查響應
            if response.status_code == 200 or response.status_code == 201:
                logger.info("模擬結果導出成功")
                return True
            else:
                logger.warning(f"模擬結果導出失敗: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"模擬結果導出異常: {str(e)}")
            return False
    
    def sync_data(self, direction='both'):
        """同步資料
        
        Args:
            direction: 同步方向，'import'、'export'或'both'
            
        Returns:
            同步結果
        """
        if not self.api_url or not self.api_key:
            logger.warning("API URL或API密鑰未設置")
            return {'success': False, 'message': 'API URL或API密鑰未設置'}
        
        if not self.db:
            logger.warning("資料庫未設置")
            return {'success': False, 'message': '資料庫未設置'}
        
        result = {
            'success': True,
            'import_count': 0,
            'export_count': 0,
            'errors': []
        }
        
        try:
            # 導入資料
            if direction in ['import', 'both']:
                import_count = self.import_customers()
                result['import_count'] = import_count
                
                if import_count == 0:
                    result['errors'].append('導入資料失敗或無資料可導入')
            
            # 導出資料
            if direction in ['export', 'both']:
                # 獲取所有消費者
                consumers = self.db.get_all_consumers()
                
                if consumers:
                    # 構建請求頭
                    headers = {
                        'Authorization': f'Bearer {self.api_key}',
                        'Content-Type': 'application/json'
                    }
                    
                    # 構建請求數據
                    data = {
                        'consumers': consumers,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # 發送請求
                    response = requests.post(f'{self.api_url}/consumers', headers=headers, json=data, timeout=30)
                    
                    # 檢查響應
                    if response.status_code == 200 or response.status_code == 201:
                        result['export_count'] = len(consumers)
                        logger.info(f"導出了 {len(consumers)} 個消費者")
                    else:
                        result['errors'].append(f"導出資料失敗: {response.status_code} - {response.text}")
                        result['success'] = False
                else:
                    result['errors'].append('無資料可導出')
            
            return result
        except Exception as e:
            logger.error(f"同步資料異常: {str(e)}")
            result['success'] = False
            result['errors'].append(str(e))
            return result
    
    def get_api_status(self):
        """獲取API狀態
        
        Returns:
            API狀態
        """
        if not self.api_url or not self.api_key:
            return {
                'configured': False,
                'connected': False,
                'api_url': self.api_url,
                'last_sync': None
            }
        
        # 測試連接
        connected = self.test_connection()
        
        # 獲取最後同步時間
        last_sync = None
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    for line in reversed(f.readlines()):
                        if '模擬結果導出成功' in line or '從CRM系統導入了' in line:
                            last_sync = line.split(' - ')[0].strip()
                            break
        except:
            pass
        
        return {
            'configured': True,
            'connected': connected,
            'api_url': self.api_url,
            'last_sync': last_sync
        }
    
    def mock_api(self, enable=True):
        """啟用或禁用模擬API
        
        Args:
            enable: 是否啟用模擬API
            
        Returns:
            是否成功
        """
        if enable:
            # 設置模擬API
            self.api_url = 'http://localhost:5000/api'
            self.api_key = 'mock_api_key'
            
            logger.info("啟用模擬API")
            return True
        else:
            # 清除API設置
            self.api_url = None
            self.api_key = None
            
            logger.info("禁用模擬API")
            return True
