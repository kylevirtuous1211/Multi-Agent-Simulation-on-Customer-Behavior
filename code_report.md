# 消費者行為模擬系統 - 代碼報告

## 系統概述

消費者行為模擬系統是一個基於AI代理的平台，能夠模擬消費者對新產品的反應。系統使用生成式代理架構，基於「Generative Agents: Interactive Simulacra of Human Behavior」論文實現，能夠模擬真實的人類行為和決策過程。

系統主要功能包括：
1. 消費者資料管理：輸入、存儲和管理消費者資料
2. 消費者分類：使用多種聚類算法對消費者進行分類
3. AI代理訓練：基於消費者資料訓練生成式AI代理
4. 產品測試：使用文本和圖像輸入測試新產品，分析代理反應
5. CRM系統整合：與企業CRM系統進行API連接和資料同步

## 系統架構

系統採用模塊化設計，主要包含以下組件：

1. **資料庫模組**：處理消費者資料的存儲和管理
2. **分類系統**：實現消費者分類和聚類分析
3. **代理架構**：實現生成式代理，包括記憶、反思、規劃和互動能力
4. **產品測試**：實現產品測試和代理反應分析
5. **API整合**：實現與企業CRM系統的連接
6. **Web界面**：提供用戶友好的操作界面

## 模組詳解

### 1. 資料庫模組 (`models/consumer.py`)

#### 主要類和函數

**ConsumerDatabase 類**：處理消費者資料的存儲和管理

| 函數 | 說明 | 可編輯性 |
|------|------|----------|
| `__init__(db_path)` | 初始化資料庫連接 | 可編輯：可修改默認資料庫路徑 |
| `_connect()` | 連接到資料庫 | 不建議編輯：核心功能 |
| `_disconnect()` | 關閉資料庫連接 | 不建議編輯：核心功能 |
| `_init_db()` | 初始化資料庫表結構 | 可編輯：可添加新的表或字段 |
| `add_consumer(demographic, behavioral, psychographic)` | 添加消費者資料 | 可編輯：可擴展接受的資料類型 |
| `get_consumer(consumer_id)` | 獲取特定消費者資料 | 不建議編輯：核心功能 |
| `update_consumer(consumer_id, demographic, behavioral, psychographic)` | 更新消費者資料 | 可編輯：可添加更新邏輯 |
| `delete_consumer(consumer_id)` | 刪除消費者資料 | 不建議編輯：核心功能 |
| `get_all_consumers(limit, offset)` | 獲取所有消費者資料 | 可編輯：可添加排序或過濾功能 |
| `search_consumers(query)` | 搜索消費者資料 | 可編輯：可改進搜索算法 |
| `get_consumer_count()` | 獲取消費者總數 | 不建議編輯：簡單功能 |
| `import_consumers(consumers)` | 批量導入消費者資料 | 可編輯：可添加驗證邏輯 |
| `export_consumers(consumer_ids)` | 導出消費者資料 | 可編輯：可添加格式選項 |

### 2. 分類系統 (`models/classification.py`)

#### 主要類和函數

**ConsumerClassifier 類**：實現消費者分類和聚類分析

| 函數 | 說明 | 可編輯性 |
|------|------|----------|
| `__init__()` | 初始化分類系統 | 可編輯：可添加更多初始化參數 |
| `preprocess_data()` | 預處理消費者資料，提取特徵 | 可編輯：可改進特徵提取方法 |
| `train_kmeans(n_clusters)` | 訓練K-means聚類模型 | 可編輯：可調整算法參數 |
| `train_dbscan(eps, min_samples)` | 訓練DBSCAN聚類模型 | 可編輯：可調整算法參數 |
| `train_hierarchical(n_clusters)` | 訓練層次聚類模型 | 可編輯：可調整算法參數 |
| `analyze_clusters()` | 分析聚類結果 | 可編輯：可添加更多分析指標 |
| `visualize_clusters(method)` | 視覺化聚類結果 | 可編輯：可添加更多視覺化方法 |
| `find_optimal_clusters(max_clusters, X)` | 尋找最佳聚類數量 | 可編輯：可使用不同的評估指標 |
| `get_consumer_cluster(consumer_id)` | 獲取消費者所屬的聚類 | 不建議編輯：核心功能 |
| `get_cluster_consumers(cluster_label)` | 獲取聚類中的所有消費者 | 不建議編輯：核心功能 |
| `predict_cluster(consumer_data)` | 預測新消費者的聚類 | 可編輯：可改進預測方法 |

### 3. 代理架構 (`models/agent.py`)

#### 主要類和函數

**Memory 類**：管理代理的記憶

| 函數 | 說明 | 可編輯性 |
|------|------|----------|
| `__init__(capacity)` | 初始化記憶系統 | 可編輯：可調整默認容量 |
| `add(content, importance, source, related_memories)` | 添加記憶 | 可編輯：可添加更多記憶屬性 |
| `get(memory_id)` | 獲取特定記憶 | 不建議編輯：核心功能 |
| `search(query, limit)` | 搜索記憶 | 可編輯：可改進搜索算法 |
| `get_recent(limit)` | 獲取最近的記憶 | 可編輯：可添加時間範圍參數 |
| `get_important(limit)` | 獲取最重要的記憶 | 可編輯：可添加重要性閾值參數 |
| `update_importance(memory_id, importance)` | 更新記憶重要性 | 不建議編輯：簡單功能 |
| `forget(memory_id)` | 刪除記憶 | 不建議編輯：簡單功能 |
| `decay(rate)` | 記憶衰減 | 可編輯：可改進衰減算法 |
| `save(path)` | 保存記憶到文件 | 不建議編輯：簡單功能 |
| `load(path)` | 從文件加載記憶 | 不建議編輯：簡單功能 |

**Reflection 類**：對記憶進行反思，形成更高層次的見解

| 函數 | 說明 | 可編輯性 |
|------|------|----------|
| `__init__(memory)` | 初始化反思機制 | 不建議編輯：簡單功能 |
| `reflect(trigger, related_memories)` | 進行反思 | 可編輯：可改進反思生成邏輯 |
| `_generate_insight(memory_contents, trigger)` | 生成反思內容 | 可編輯：可替換為更複雜的生成模型 |
| `get_recent_insights(limit)` | 獲取最近的反思 | 不建議編輯：簡單功能 |
| `save(path)` | 保存反思到文件 | 不建議編輯：簡單功能 |
| `load(path)` | 從文件加載反思 | 不建議編輯：簡單功能 |

**Planning 類**：設定目標並生成實現目標的計劃

| 函數 | 說明 | 可編輯性 |
|------|------|----------|
| `__init__(memory, reflection)` | 初始化規劃能力 | 不建議編輯：簡單功能 |
| `set_goal(goal, importance, deadline)` | 設定目標 | 可編輯：可添加目標類型或優先級 |
| `create_plan(goal_id)` | 為目標創建計劃 | 可編輯：可改進計劃生成邏輯 |
| `_generate_plan(goal, related_memories, recent_insights)` | 生成計劃內容 | 可編輯：可替換為更複雜的生成模型 |
| `_generate_steps(goal, plan)` | 生成計劃步驟 | 可編輯：可替換為更複雜的生成模型 |
| `get_active_goals()` | 獲取活躍的目標 | 不建議編輯：簡單功能 |
| `get_plan(plan_id)` | 獲取特定計劃 | 不建議編輯：簡單功能 |
| `update_plan_status(plan_id, status)` | 更新計劃狀態 | 不建議編輯：簡單功能 |
| `update_step_status(plan_id, step_id, status)` | 更新計劃步驟狀態 | 不建議編輯：簡單功能 |
| `advance_plan(plan_id)` | 推進計劃到下一步 | 可編輯：可添加條件檢查邏輯 |
| `save(path)` | 保存規劃到文件 | 不建議編輯：簡單功能 |
| `load(path)` | 從文件加載規劃 | 不建議編輯：簡單功能 |

**Interaction 類**：與其他代理和用戶進行對話互動

| 函數 | 說明 | 可編輯性 |
|------|------|----------|
| `__init__(memory, reflection, planning)` | 初始化互動能力 | 不建議編輯：簡單功能 |
| `receive_message(sender, message)` | 接收消息 | 可編輯：可添加消息處理邏輯 |
| `_generate_reply(sender, message)` | 生成回覆 | 可編輯：可替換為更複雜的生成模型 |
| `get_conversation_history(limit)` | 獲取對話歷史 | 不建議編輯：簡單功能 |
| `save(path)` | 保存對話歷史到文件 | 不建議編輯：簡單功能 |
| `load(path)` | 從文件加載對話歷史 | 不建議編輯：簡單功能 |

**GenerativeAgent 類**：模擬人類行為和決策過程

| 函數 | 說明 | 可編輯性 |
|------|------|----------|
| `__init__(agent_id, profile)` | 初始化生成式代理 | 可編輯：可添加更多初始化參數 |
| `update()` | 更新代理狀態 | 可編輯：可添加更多更新邏輯 |
| `evaluate_product(product_info)` | 評估產品 | 可編輯：可改進評估邏輯 |
| `_evaluate_sentiment(product_info)` | 評估對產品的情感 | 可編輯：可替換為更複雜的情感分析模型 |
| `_evaluate_purchase_intent(product_info)` | 評估購買意願 | 可編輯：可替換為更複雜的意願預測模型 |
| `_generate_detailed_feedback(product_info)` | 生成詳細評價 | 可編輯：可替換為更複雜的生成模型 |
| `save(base_dir)` | 保存代理到文件 | 不建議編輯：核心功能 |
| `load(agent_id, base_dir)` | 從文件加載代理 | 不建議編輯：核心功能 |

**AgentManager 類**：管理多個代理

| 函數 | 說明 | 可編輯性 |
|------|------|----------|
| `__init__()` | 初始化代理管理器 | 不建議編輯：簡單功能 |
| `create_agent(profile)` | 創建代理 | 可編輯：可添加更多初始化邏輯 |
| `create_agent_from_consumer(consumer_id)` | 從消費者資料創建代理 | 可編輯：可改進資料轉換邏輯 |
| `create_agents_from_group(group)` | 從分類群組創建代理 | 可編輯：可添加群組選擇邏輯 |
| `get_agent(agent_id)` | 獲取代理 | 不建議編輯：核心功能 |
| `get_all_agents()` | 獲取所有代理ID | 不建議編輯：簡單功能 |
| `get_agent_summary(agent_id)` | 獲取代理摘要 | 可編輯：可添加更多摘要信息 |
| `delete_agent(agent_id)` | 刪除代理 | 不建議編輯：核心功能 |
| `simulate_product_evaluation(product_info, agent_ids)` | 模擬產品評估 | 可編輯：可添加更多模擬參數 |

### 4. 產品測試 (`models/product_tester.py`)

#### 主要類和函數

**ProductTester 類**：實現產品測試和代理反應分析

| 函數 | 說明 | 可編輯性 |
|------|------|----------|
| `__init__()` | 初始化產品測試器 | 不建議編輯：簡單功能 |
| `test_product(product_info, agent_ids)` | 測試產品 | 可編輯：可添加更多測試參數 |
| `analyze_results(results)` | 分析測試結果 | 可編輯：可添加更多分析指標 |
| `_generate_sentiment_chart(sentiment_counts)` | 生成情感分佈圖 | 可編輯：可改進圖表樣式 |
| `_generate_purchase_intent_chart(purchase_intent_counts)` | 生成購買意願分佈圖 | 可編輯：可改進圖表樣式 |
| `generate_report(product_info, results, analysis)` | 生成測試報告 | 可編輯：可改進報告格式和內容 |
| `save_product_image(image_file)` | 保存產品圖片 | 不建議編輯：簡單功能 |
| `process_product_image(image_path)` | 處理產品圖片 | 可編輯：可添加更多圖像處理功能 |

### 5. API整合 (`api_integration/crm_api.py`)

#### 主要類和函數

**CRMConnector 類**：實現與企業CRM系統的API連接

| 函數 | 說明 | 可編輯性 |
|------|------|----------|
| `__init__(api_url, api_key)` | 初始化CRM連接器 | 可編輯：可添加更多初始化參數 |
| `test_connection()` | 測試API連接 | 不建議編輯：核心功能 |
| `configure(api_url, api_key)` | 配置API連接 | 不建議編輯：簡單功能 |
| `import_customers(limit)` | 從CRM系統導入客戶資料 | 可編輯：可添加資料轉換邏輯 |
| `export_simulation_results(product_info, results)` | 將模擬結果導出到CRM系統 | 可編輯：可添加資料格式化邏輯 |
| `sync_data(direction)` | 同步資料 | 可編輯：可改進同步策略 |
| `get_api_status()` | 獲取API狀態 | 不建議編輯：簡單功能 |
| `mock_api(enable)` | 啟用或禁用模擬API | 可編輯：可改進模擬邏輯 |

### 6. Web應用 (`app.py`)

#### 主要路由和函數

**Flask應用**：提供Web界面

| 路由/函數 | 說明 | 可編輯性 |
|------|------|----------|
| `index()` | 主頁 | 可編輯：可添加更多統計信息 |
| `consumers()` | 消費者資料管理頁面 | 可編輯：可改進分頁邏輯 |
| `add_consumer()` | 添加消費者 | 可編輯：可添加表單驗證 |
| `view_consumer(consumer_id)` | 查看消費者 | 可編輯：可添加更多顯示信息 |
| `edit_consumer(consumer_id)` | 編輯消費者 | 可編輯：可添加表單驗證 |
| `delete_consumer(consumer_id)` | 刪除消費者 | 不建議編輯：簡單功能 |
| `import_consumers()` | 導入消費者 | 可編輯：可添加更多文件格式支持 |
| `export_consumers()` | 導出消費者 | 可編輯：可添加更多文件格式選項 |
| `classification()` | 分類系統頁面 | 可編輯：可添加更多分類選項 |
| `run_kmeans()` | 運行K-means聚類 | 可編輯：可添加更多參數選項 |
| `run_dbscan()` | 運行DBSCAN聚類 | 可編輯：可添加更多參數選項 |
| `run_hierarchical()` | 運行層次聚類 | 可編輯：可添加更多參數選項 |
| `find_optimal_clusters()` | 尋找最佳聚類數量 | 不建議編輯：核心功能 |
| `agents()` | 代理管理頁面 | 可編輯：可添加更多顯示信息 |
| `create_agent()` | 創建代理 | 可編輯：可添加表單驗證 |
| `create_agent_from_consumer(consumer_id)` | 從消費者創建代理 | 不建議編輯：核心功能 |
| `create_agents_from_group()` | 從群組創建代理 | 可編輯：可添加群組選擇邏輯 |
| `view_agent(agent_id)` | 查看代理 | 可編輯：可添加更多顯示信息 |
| `delete_agent(agent_id)` | 刪除代理 | 不建議編輯：簡單功能 |
| `product_test()` | 產品測試頁面 | 可編輯：可添加更多測試選項 |
| `crm()` | CRM整合頁面 | 可編輯：可添加更多配置選項 |
| `configure_crm()` | 配置CRM API | 不建議編輯：核心功能 |
| `mock_crm()` | 啟用模擬API | 不建議編輯：簡單功能 |
| `import_from_crm()` | 從CRM導入 | 不建議編輯：核心功能 |
| `sync_with_crm()` | 同步資料 | 不建議編輯：核心功能 |
| API端點 | 提供REST API | 可編輯：可添加更多API功能 |

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

## 結論

消費者行為模擬系統提供了一個完整的解決方案，用於模擬消費者對新產品的反應。系統採用模塊化設計，各組件之間耦合度低，便於擴展和維護。通過編輯上述標記為"可編輯"的函數，可以根據特定需求定制系統功能。

系統的核心價值在於生成式代理架構，它能夠模擬真實的人類行為和決策過程，為企業提供有價值的市場洞察。通過與CRM系統的整合，系統可以直接利用企業現有的客戶資料，提高模擬的準確性和實用性。
