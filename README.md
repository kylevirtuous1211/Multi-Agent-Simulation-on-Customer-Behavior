<<<<<<< HEAD
# Multi-Agent-Simulation-on-Customer-Behavior
=======
# 消費者行為模擬系統

基於生成式代理架構的消費者行為模擬系統，用於預測消費者對新產品的反應。

## Code structure
      ├── app.py # Main Flask application file 
      ├── models/ # Directory containing data models 
         ├── AI_Agent_CoT.ipynb # **目前的prototype**
         └── ... # 以後有一些clustering演算法可以放這裡
      ├── utils/ # Utility functions 之後如果有的話
      ├── data/ # Data storage directory 網站記錄用戶資料的地方 (訓練資料在`training data`)
       ├── consumer_database.db # SQLite database for consumer data 
       ├── agents/ # Directory for storing agent-specific data 
       └── ... # Other data files 
      ├── training_data/ # 訓練用到的data放在這邊
      ├── Load_data_from_kaggle.py # Script to download dataset from Kaggle 
       └── ... # Other training-related files 
      ├── requirements.txt # List of Python dependencies 
      ├── README.md # Project documentation (this file) 
      └── ... # Other project files

## 目前進度
因為修了GenAI的課，有做了一個小prototype把他放在 `models/AI_Agent_CoT.ipynb`

## 系統概述

消費者行為模擬系統是一個基於AI代理的平台，能夠模擬消費者對新產品的反應。系統使用生成式代理架構，基於「Generative Agents: Interactive Simulacra of Human Behavior」論文實現，能夠模擬真實的人類行為和決策過程。

### 主要功能

1. **消費者資料管理**：輸入、存儲和管理消費者資料
2. **消費者分類**：使用多種聚類算法對消費者進行分類
3. **AI代理訓練**：基於消費者資料訓練生成式AI代理
4. **產品測試**：使用文本和圖像輸入測試新產品，分析代理反應
5. **CRM系統整合**：與企業CRM系統進行API連接和資料同步

## 本地部署指南

### 1. 系統需求

- Python 3.10 或更高版本
- 足夠的磁盤空間（至少200MB）
- 現代網頁瀏覽器（Chrome、Firefox、Edge等）

### 2. 安裝步驟

#### 2.1 準備環境

1. **安裝 Python**：
   - 下載並安裝 Python 3.10 或更高版本：https://www.python.org/downloads/
   - 確保在安裝時勾選"Add Python to PATH"選項

2. **安裝 Git**（可選，用於克隆代碼）：
   - 下載並安裝 Git：https://git-scm.com/downloads

#### 2.2 獲取代碼

1. **下載代碼**：
   - 從提供的 ZIP 文件中解壓代碼
   - 或者使用 Git 克隆代碼庫（如果有）

2. **進入項目目錄**：
   ```bash
   cd 路徑/到/consumer_behavior_sim_web
   ```

#### 2.3 設置虛擬環境（推薦）

1. **創建虛擬環境**：
   ```bash
   python -m venv venv
   ```

2. **激活虛擬環境**：
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

#### 2.4 安裝依賴

1. **安裝所需的 Python 包**：
   ```bash
   pip install -r requirements.txt
   ```

### 3. 運行應用

#### 3.1 開發模式

1. **啟動 Flask 應用**：
   ```bash
   python app.py
   ```

2. **訪問應用**：
   - 打開瀏覽器
   - 訪問 http://localhost:5000

#### 3.2 生產模式（可選）

##### Linux/macOS:

1. **安裝 Gunicorn**：
   ```bash
   pip install gunicorn
   ```

2. **使用 Gunicorn 啟動**：
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

##### Windows:

1. **安裝 Waitress**：
   ```bash
   pip install waitress
   ```
   
2. **使用 Waitress 啟動**：
   ```bash
   waitress-serve --port=5000 app:app
   ```

### 4. 使用系統

啟動應用後，您可以通過瀏覽器訪問系統的各個功能：

1. **消費者資料管理**：
   - 添加、編輯、刪除消費者資料
   - 導入/導出消費者資料

2. **消費者分類**：
   - 使用K-means、DBSCAN或層次聚類算法對消費者進行分類
   - 分析聚類結果
   - 視覺化聚類結果

3. **AI代理訓練**：
   - 創建代理
   - 從消費者資料創建代理
   - 從分類群組創建代理

4. **產品測試**：
   - 輸入產品信息和圖片
   - 選擇代理進行測試
   - 分析代理反應
   - 生成測試報告

5. **CRM系統整合**：
   - 配置CRM API連接
   - 導入/導出資料
   - 同步資料

### 5. 停止應用

- 在命令行按 `Ctrl+C` 停止應用
- 停用虛擬環境（如果使用）：
  ```bash
  deactivate
  ```

## 系統架構

系統採用模塊化設計，主要包含以下組件：

1. **資料庫模組**：處理消費者資料的存儲和管理
2. **分類系統**：實現消費者分類和聚類分析
3. **代理架構**：實現生成式代理，包括記憶、反思、規劃和互動能力
4. **產品測試**：實現產品測試和代理反應分析
5. **API整合**：實現與企業CRM系統的連接
6. **Web界面**：提供用戶友好的操作界面

詳細的代碼說明和可編輯性指南請參考 `code_report.md` 文件。

## 擴展建議

系統可以通過以下方式進行擴展：

1. **改進代理模型**：
   - 將簡單的模擬邏輯替換為更複雜的生成模型，如使用大型語言模型
   - 添加更多的代理特性和行為模式
   - 實現代理之間的互動和影響

2. **增強分類系統**：
   - 添加更多的聚類算法
   - 實現監督學習分類
   - 添加更多的特徵提取方法

3. **擴展產品測試**：
   - 添加更多的產品評估維度
   - 實現A/B測試功能
   - 添加競品比較功能

4. **改進CRM整合**：
   - 支持更多的CRM系統
   - 添加更多的資料同步選項
   - 實現實時資料更新

5. **增強Web界面**：
   - 添加用戶認證和權限管理
   - 實現儀表板和數據可視化
   - 添加多語言支持

## 故障排除

### 常見問題

1. **無法啟動應用**：
   - 確保已安裝所有依賴
   - 檢查Python版本是否兼容
   - 確保端口5000未被占用

2. **資料庫錯誤**：
   - 確保有寫入權限
   - 檢查資料庫文件是否損壞

3. **圖像處理錯誤**：
   - 確保已安裝Pillow庫
   - 檢查圖像格式是否支持

### 聯繫支持

如有任何問題或需要進一步的協助，請聯繫系統開發團隊。

## 許可證

本系統基於MIT許可證開源。

## 致謝

- 感謝「Generative Agents: Interactive Simulacra of Human Behavior」論文的作者提供理論基礎
- 感謝所有開源庫的貢獻者
>>>>>>> fa3ac39 (initial commit after code created by Manus)
