"""
消費者行為模擬系統 - 主程序
Flask網頁應用程式入口點
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
import os
import json
import logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 設置Matplotlib後端為Agg，避免需要GUI

# 導入模組
from models.consumer import ConsumerDatabase
from models.classification import ConsumerClassifier
from models.agent import AgentManager, GenerativeAgent
from models.product_tester import ProductTester
from api_integration.crm_api import CRMConnector

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 創建Flask應用
app = Flask(__name__)
app.secret_key = os.urandom(24)

# 初始化模組
db = ConsumerDatabase()
classifier = ConsumerClassifier()
classifier.db = db
agent_manager = AgentManager()
agent_manager.db = db
product_tester = ProductTester()
product_tester.agent_manager = agent_manager
crm_connector = CRMConnector()
crm_connector.db = db

# 確保目錄存在
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs('data/agents', exist_ok=True)
os.makedirs('data/logs', exist_ok=True)

# 主頁
@app.route('/')
def index():
    # 獲取統計數據
    stats = {
        'consumer_count': db.get_consumer_count(),
        'agent_count': len(agent_manager.get_all_agents())
    }
    return render_template('index.html', stats=stats)

# 消費者資料管理
@app.route('/consumers')
def consumers():
    page = request.args.get('page', 1, type=int)
    limit = 10
    offset = (page - 1) * limit
    
    consumers = db.get_all_consumers(limit=limit, offset=offset)
    total_count = db.get_consumer_count()
    
    return render_template('consumer_input.html', 
                          consumers=consumers, 
                          page=page, 
                          total_pages=(total_count // limit) + (1 if total_count % limit > 0 else 0))

@app.route('/consumers/add', methods=['GET', 'POST'])
def add_consumer():
    if request.method == 'POST':
        try:
            # 獲取表單數據
            demographic = {
                'age': int(request.form.get('age', 0)),
                'gender': request.form.get('gender', ''),
                'income': int(request.form.get('income', 0)),
                'education': request.form.get('education', ''),
                'occupation': request.form.get('occupation', ''),
                'location': request.form.get('location', '')
            }
            
            behavioral = {
                'purchase_frequency': int(request.form.get('purchase_frequency', 0)),
                'brand_loyalty': int(request.form.get('brand_loyalty', 0)),
                'price_consciousness': int(request.form.get('price_consciousness', 0)),
                'tech_savviness': int(request.form.get('tech_savviness', 0)),
                'social_media_usage': int(request.form.get('social_media_usage', 0))
            }
            
            psychographic = {
                'personality': request.form.get('personality', ''),
                'interests': request.form.get('interests', ''),
                'values': request.form.get('values', ''),
                'lifestyle': request.form.get('lifestyle', ''),
                'innovativeness': int(request.form.get('innovativeness', 0)),
                'social_influence': int(request.form.get('social_influence', 0))
            }
            
            # 添加消費者
            consumer_id = db.add_consumer(demographic, behavioral, psychographic)
            
            flash(f'成功添加消費者 (ID: {consumer_id})', 'success')
            return redirect(url_for('consumers'))
        except Exception as e:
            flash(f'添加消費者失敗: {str(e)}', 'danger')
            return redirect(url_for('add_consumer'))
    
    return render_template('consumer_form.html')

@app.route('/consumers/<int:consumer_id>')
def view_consumer(consumer_id):
    consumer = db.get_consumer(consumer_id)
    if not consumer:
        flash('找不到消費者', 'danger')
        return redirect(url_for('consumers'))
    
    return render_template('consumer_view.html', consumer=consumer)

@app.route('/consumers/<int:consumer_id>/edit', methods=['GET', 'POST'])
def edit_consumer(consumer_id):
    consumer = db.get_consumer(consumer_id)
    if not consumer:
        flash('找不到消費者', 'danger')
        return redirect(url_for('consumers'))
    
    if request.method == 'POST':
        try:
            # 獲取表單數據
            demographic = {
                'age': int(request.form.get('age', 0)),
                'gender': request.form.get('gender', ''),
                'income': int(request.form.get('income', 0)),
                'education': request.form.get('education', ''),
                'occupation': request.form.get('occupation', ''),
                'location': request.form.get('location', '')
            }
            
            behavioral = {
                'purchase_frequency': int(request.form.get('purchase_frequency', 0)),
                'brand_loyalty': int(request.form.get('brand_loyalty', 0)),
                'price_consciousness': int(request.form.get('price_consciousness', 0)),
                'tech_savviness': int(request.form.get('tech_savviness', 0)),
                'social_media_usage': int(request.form.get('social_media_usage', 0))
            }
            
            psychographic = {
                'personality': request.form.get('personality', ''),
                'interests': request.form.get('interests', ''),
                'values': request.form.get('values', ''),
                'lifestyle': request.form.get('lifestyle', ''),
                'innovativeness': int(request.form.get('innovativeness', 0)),
                'social_influence': int(request.form.get('social_influence', 0))
            }
            
            # 更新消費者
            db.update_consumer(consumer_id, demographic, behavioral, psychographic)
            
            flash('成功更新消費者', 'success')
            return redirect(url_for('view_consumer', consumer_id=consumer_id))
        except Exception as e:
            flash(f'更新消費者失敗: {str(e)}', 'danger')
            return redirect(url_for('edit_consumer', consumer_id=consumer_id))
    
    return render_template('consumer_form.html', consumer=consumer, edit=True)

@app.route('/consumers/<int:consumer_id>/delete', methods=['POST'])
def delete_consumer(consumer_id):
    try:
        db.delete_consumer(consumer_id)
        flash('成功刪除消費者', 'success')
    except Exception as e:
        flash(f'刪除消費者失敗: {str(e)}', 'danger')
    
    return redirect(url_for('consumers'))

@app.route('/consumers/import', methods=['GET', 'POST'])
def import_consumers():
    if request.method == 'POST':
        try:
            # 檢查是否有文件
            if 'file' not in request.files:
                flash('沒有選擇文件', 'danger')
                return redirect(request.url)
            
            file = request.files['file']
            
            # 檢查文件名
            if file.filename == '':
                flash('沒有選擇文件', 'danger')
                return redirect(request.url)
            
            # 檢查文件類型
            if not file.filename.endswith('.json'):
                flash('只支持JSON文件', 'danger')
                return redirect(request.url)
            
            # 讀取文件
            consumers = json.load(file)
            
            # 導入消費者
            count = db.import_consumers(consumers)
            
            flash(f'成功導入 {count} 個消費者', 'success')
            return redirect(url_for('consumers'))
        except Exception as e:
            flash(f'導入消費者失敗: {str(e)}', 'danger')
            return redirect(url_for('import_consumers'))
    
    return render_template('consumer_import.html')

@app.route('/consumers/export')
def export_consumers():
    try:
        # 導出所有消費者
        consumers = db.export_consumers()
        
        # 創建JSON文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'consumers_{timestamp}.json'
        filepath = os.path.join('static/uploads', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(consumers, f, ensure_ascii=False, indent=2)
        
        # 返回文件
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        flash(f'導出消費者失敗: {str(e)}', 'danger')
        return redirect(url_for('consumers'))

# 消費者分類
@app.route('/classification')
def classification():
    return render_template('classification.html')

@app.route('/classification/kmeans', methods=['POST'])
def run_kmeans():
    try:
        # 獲取參數
        n_clusters = int(request.form.get('n_clusters', 5))
        
        # 運行K-means
        labels = classifier.train_kmeans(n_clusters=n_clusters)
        
        # 分析聚類結果
        analysis = classifier.analyze_clusters()
        
        # 視覺化聚類結果
        image_path = classifier.visualize_clusters()
        
        # 返回結果
        return render_template('classification_result.html', 
                              method='K-means',
                              analysis=analysis,
                              image_path=image_path)
    except Exception as e:
        flash(f'運行K-means失敗: {str(e)}', 'danger')
        return redirect(url_for('classification'))

@app.route('/classification/dbscan', methods=['POST'])
def run_dbscan():
    try:
        # 獲取參數
        eps = float(request.form.get('eps', 0.5))
        min_samples = int(request.form.get('min_samples', 5))
        
        # 運行DBSCAN
        labels = classifier.train_dbscan(eps=eps, min_samples=min_samples)
        
        # 分析聚類結果
        analysis = classifier.analyze_clusters()
        
        # 視覺化聚類結果
        image_path = classifier.visualize_clusters()
        
        # 返回結果
        return render_template('classification_result.html', 
                              method='DBSCAN',
                              analysis=analysis,
                              image_path=image_path)
    except Exception as e:
        flash(f'運行DBSCAN失敗: {str(e)}', 'danger')
        return redirect(url_for('classification'))

@app.route('/classification/hierarchical', methods=['POST'])
def run_hierarchical():
    try:
        # 獲取參數
        n_clusters = int(request.form.get('n_clusters', 5))
        
        # 運行層次聚類
        labels = classifier.train_hierarchical(n_clusters=n_clusters)
        
        # 分析聚類結果
        analysis = classifier.analyze_clusters()
        
        # 視覺化聚類結果
        image_path = classifier.visualize_clusters()
        
        # 返回結果
        return render_template('classification_result.html', 
                              method='層次聚類',
                              analysis=analysis,
                              image_path=image_path)
    except Exception as e:
        flash(f'運行層次聚類失敗: {str(e)}', 'danger')
        return redirect(url_for('classification'))

@app.route('/classification/optimal')
def find_optimal_clusters():
    try:
        # 尋找最佳聚類數量
        optimal_clusters, silhouette_scores = classifier.find_optimal_clusters()
        
        # 返回結果
        return jsonify({
            'optimal_clusters': int(optimal_clusters),
            'silhouette_scores': silhouette_scores.tolist()
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

# 代理管理
@app.route('/agents')
def agents():
    # 獲取所有代理
    agent_ids = agent_manager.get_all_agents()
    
    # 獲取代理摘要
    agents = []
    for agent_id in agent_ids:
        summary = agent_manager.get_agent_summary(agent_id)
        if summary:
            agents.append(summary)
    
    return render_template('agents.html', agents=agents)

@app.route('/agents/create', methods=['GET', 'POST'])
def create_agent():
    if request.method == 'POST':
        try:
            # 獲取表單數據
            profile = {
                'demographic': {
                    'age': int(request.form.get('age', 0)),
                    'gender': request.form.get('gender', ''),
                    'income': int(request.form.get('income', 0)),
                    'education': request.form.get('education', ''),
                    'occupation': request.form.get('occupation', ''),
                    'location': request.form.get('location', '')
                },
                'behavioral': {
                    'purchase_frequency': int(request.form.get('purchase_frequency', 0)),
                    'brand_loyalty': int(request.form.get('brand_loyalty', 0)),
                    'price_consciousness': int(request.form.get('price_consciousness', 0)),
                    'tech_savviness': int(request.form.get('tech_savviness', 0)),
                    'social_media_usage': int(request.form.get('social_media_usage', 0))
                },
                'psychographic': {
                    'personality': request.form.get('personality', ''),
                    'interests': request.form.get('interests', ''),
                    'values': request.form.get('values', ''),
                    'lifestyle': request.form.get('lifestyle', ''),
                    'innovativeness': int(request.form.get('innovativeness', 0)),
                    'social_influence': int(request.form.get('social_influence', 0))
                }
            }
            
            # 創建代理
            agent_id = agent_manager.create_agent(profile)
            
            flash(f'成功創建代理 (ID: {agent_id})', 'success')
            return redirect(url_for('agents'))
        except Exception as e:
            flash(f'創建代理失敗: {str(e)}', 'danger')
            return redirect(url_for('create_agent'))
    
    return render_template('agent_form.html')

@app.route('/agents/create_from_consumer/<int:consumer_id>')
def create_agent_from_consumer(consumer_id):
    try:
        # 從消費者創建代理
        agent_id = agent_manager.create_agent_from_consumer(consumer_id)
        
        if agent_id:
            flash(f'成功從消費者創建代理 (ID: {agent_id})', 'success')
        else:
            flash('從消費者創建代理失敗', 'danger')
        
        return redirect(url_for('agents'))
    except Exception as e:
        flash(f'從消費者創建代理失敗: {str(e)}', 'danger')
        return redirect(url_for('agents'))

@app.route('/agents/create_from_group', methods=['POST'])
def create_agents_from_group():
    try:
        # 獲取群組
        group = int(request.form.get('group', 0))
        
        # 從群組創建代理
        agent_ids = agent_manager.create_agents_from_group(group)
        
        if agent_ids:
            flash(f'成功從群組創建 {len(agent_ids)} 個代理', 'success')
        else:
            flash('從群組創建代理失敗', 'danger')
        
        return redirect(url_for('agents'))
    except Exception as e:
        flash(f'從群組創建代理失敗: {str(e)}', 'danger')
        return redirect(url_for('agents'))

@app.route('/agents/<agent_id>')
def view_agent(agent_id):
    # 獲取代理
    agent = agent_manager.get_agent(agent_id)
    
    if not agent:
        flash('找不到代理', 'danger')
        return redirect(url_for('agents'))
    
    # 獲取代理摘要
    summary = agent_manager.get_agent_summary(agent_id)
    
    return render_template('agent_view.html', agent=agent, summary=summary)

@app.route('/agents/<agent_id>/delete', methods=['POST'])
def delete_agent(agent_id):
    try:
        # 刪除代理
        agent_manager.delete_agent(agent_id)
        
        flash('成功刪除代理', 'success')
        return redirect(url_for('agents'))
    except Exception as e:
        flash(f'刪除代理失敗: {str(e)}', 'danger')
        return redirect(url_for('agents'))

# 產品測試
@app.route('/product_test', methods=['GET', 'POST'])
def product_test():
    if request.method == 'POST':
        try:
            # 獲取表單數據
            product_info = {
                'name': request.form.get('name', ''),
                'category': request.form.get('category', ''),
                'price': request.form.get('price', ''),
                'features': request.form.get('features', '').split(','),
                'description': request.form.get('description', '')
            }
            
            # 處理產品圖片
            if 'image' in request.files and request.files['image'].filename:
                image_file = request.files['image']
                image_path = product_tester.save_product_image(image_file)
                if image_path:
                    product_info['image'] = image_path
            
            # 獲取選定的代理
            agent_ids = request.form.getlist('agent_ids')
            if not agent_ids:
                agent_ids = None  # 使用所有代理
            
            # 測試產品
            result = product_tester.test_product(product_info, agent_ids)
            
            # 返回結果
            return render_template('product_test_result.html', 
                                  product_info=product_info,
                                  result=result)
        except Exception as e:
            flash(f'產品測試失敗: {str(e)}', 'danger')
            return redirect(url_for('product_test'))
    
    # 獲取所有代理
    agent_ids = agent_manager.get_all_agents()
    agents = []
    for agent_id in agent_ids:
        summary = agent_manager.get_agent_summary(agent_id)
        if summary:
            agents.append(summary)
    
    return render_template('product_test.html', agents=agents)

# CRM API整合
@app.route('/crm')
def crm():
    # 獲取API狀態
    api_status = crm_connector.get_api_status()
    
    return render_template('crm.html', api_status=api_status)

@app.route('/crm/configure', methods=['POST'])
def configure_crm():
    try:
        # 獲取表單數據
        api_url = request.form.get('api_url', '')
        api_key = request.form.get('api_key', '')
        
        # 配置API
        success = crm_connector.configure(api_url, api_key)
        
        if success:
            flash('成功配置CRM API', 'success')
        else:
            flash('配置CRM API失敗', 'danger')
        
        return redirect(url_for('crm'))
    except Exception as e:
        flash(f'配置CRM API失敗: {str(e)}', 'danger')
        return redirect(url_for('crm'))

@app.route('/crm/mock', methods=['POST'])
def mock_crm():
    try:
        # 啟用模擬API
        crm_connector.mock_api()
        
        flash('成功啟用模擬API', 'success')
        return redirect(url_for('crm'))
    except Exception as e:
        flash(f'啟用模擬API失敗: {str(e)}', 'danger')
        return redirect(url_for('crm'))

@app.route('/crm/import', methods=['POST'])
def import_from_crm():
    try:
        # 從CRM導入
        count = crm_connector.import_customers()
        
        if count > 0:
            flash(f'成功從CRM導入 {count} 個客戶', 'success')
        else:
            flash('從CRM導入客戶失敗或無客戶可導入', 'warning')
        
        return redirect(url_for('crm'))
    except Exception as e:
        flash(f'從CRM導入客戶失敗: {str(e)}', 'danger')
        return redirect(url_for('crm'))

@app.route('/crm/sync', methods=['POST'])
def sync_with_crm():
    try:
        # 同步資料
        result = crm_connector.sync_data()
        
        if result['success']:
            flash(f'成功同步資料 (導入: {result["import_count"]}, 導出: {result["export_count"]})', 'success')
        else:
            flash(f'同步資料失敗: {", ".join(result["errors"])}', 'danger')
        
        return redirect(url_for('crm'))
    except Exception as e:
        flash(f'同步資料失敗: {str(e)}', 'danger')
        return redirect(url_for('crm'))

# API端點
@app.route('/api/consumers', methods=['GET'])
def api_get_consumers():
    try:
        # 獲取參數
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # 獲取消費者
        consumers = db.get_all_consumers(limit=limit, offset=offset)
        
        return jsonify(consumers)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/consumers/<int:consumer_id>', methods=['GET'])
def api_get_consumer(consumer_id):
    try:
        # 獲取消費者
        consumer = db.get_consumer(consumer_id)
        
        if not consumer:
            return jsonify({
                'error': '找不到消費者'
            }), 404
        
        return jsonify(consumer)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/consumers', methods=['POST'])
def api_add_consumer():
    try:
        # 獲取JSON數據
        data = request.json
        
        # 添加消費者
        consumer_id = db.add_consumer(
            data.get('demographic', {}),
            data.get('behavioral', {}),
            data.get('psychographic', {})
        )
        
        return jsonify({
            'consumer_id': consumer_id
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/agents', methods=['GET'])
def api_get_agents():
    try:
        # 獲取所有代理
        agent_ids = agent_manager.get_all_agents()
        
        # 獲取代理摘要
        agents = []
        for agent_id in agent_ids:
            summary = agent_manager.get_agent_summary(agent_id)
            if summary:
                agents.append(summary)
        
        return jsonify(agents)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/agents/<agent_id>', methods=['GET'])
def api_get_agent(agent_id):
    try:
        # 獲取代理摘要
        summary = agent_manager.get_agent_summary(agent_id)
        
        if not summary:
            return jsonify({
                'error': '找不到代理'
            }), 404
        
        return jsonify(summary)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/product_test', methods=['POST'])
def api_test_product():
    try:
        # 獲取JSON數據
        data = request.json
        
        # 測試產品
        result = product_tester.test_product(
            data.get('product_info', {}),
            data.get('agent_ids')
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

# 啟動應用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
