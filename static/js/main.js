// 主要JavaScript文件
document.addEventListener('DOMContentLoaded', function() {
    // 初始化工具提示
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });

    // 初始化彈出框
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl)
    });

    // 通用錯誤處理函數
    window.handleApiError = function(error) {
        console.error('API錯誤:', error);
        let errorMessage = '發生錯誤，請稍後再試。';
        if (error.message) {
            errorMessage = error.message;
        }
        showAlert('danger', errorMessage);
    };

    // 顯示提示訊息
    window.showAlert = function(type, message, container = '#alertContainer') {
        const alertContainer = document.querySelector(container);
        if (!alertContainer) return;

        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.role = 'alert';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;

        alertContainer.innerHTML = '';
        alertContainer.appendChild(alertDiv);

        // 5秒後自動關閉
        setTimeout(() => {
            const alert = bootstrap.Alert.getOrCreateInstance(alertDiv);
            alert.close();
        }, 5000);
    };

    // 格式化日期時間
    window.formatDateTime = function(dateTimeStr) {
        if (!dateTimeStr) return '';
        const date = new Date(dateTimeStr);
        return date.toLocaleString('zh-TW');
    };

    // 創建加載動畫
    window.createSpinner = function(container) {
        const spinnerContainer = document.createElement('div');
        spinnerContainer.className = 'spinner-container';
        spinnerContainer.innerHTML = `
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">加載中...</span>
            </div>
        `;
        container.innerHTML = '';
        container.appendChild(spinnerContainer);
    };

    // 格式化數字為百分比
    window.formatPercentage = function(value) {
        return (value * 100).toFixed(1) + '%';
    };

    // 創建圓餅圖
    window.createPieChart = function(canvasId, labels, data, colors) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        return new Chart(ctx, {
            type: 'pie',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                    }
                }
            }
        });
    };

    // 創建柱狀圖
    window.createBarChart = function(canvasId, labels, data, label, color) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: label,
                    data: data,
                    backgroundColor: color,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    };

    // 創建折線圖
    window.createLineChart = function(canvasId, labels, datasets) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    };

    // 獲取情感顏色
    window.getSentimentColor = function(sentiment) {
        switch(sentiment) {
            case 'positive': return '#198754'; // 綠色
            case 'neutral': return '#ffc107'; // 黃色
            case 'negative': return '#dc3545'; // 紅色
            default: return '#6c757d'; // 灰色
        }
    };

    // 獲取購買意願顏色
    window.getPurchaseIntentColor = function(intent) {
        switch(intent) {
            case 'high': return '#198754'; // 綠色
            case 'medium': return '#ffc107'; // 黃色
            case 'low': return '#dc3545'; // 紅色
            default: return '#6c757d'; // 灰色
        }
    };

    // 獲取情感標籤
    window.getSentimentBadge = function(sentiment) {
        let color, text;
        switch(sentiment) {
            case 'positive':
                color = 'success';
                text = '正面';
                break;
            case 'neutral':
                color = 'warning';
                text = '中性';
                break;
            case 'negative':
                color = 'danger';
                text = '負面';
                break;
            default:
                color = 'secondary';
                text = '未知';
        }
        return `<span class="badge bg-${color}">${text}</span>`;
    };

    // 獲取購買意願標籤
    window.getPurchaseIntentBadge = function(intent) {
        let color, text;
        switch(intent) {
            case 'high':
                color = 'success';
                text = '高';
                break;
            case 'medium':
                color = 'warning';
                text = '中';
                break;
            case 'low':
                color = 'danger';
                text = '低';
                break;
            default:
                color = 'secondary';
                text = '未知';
        }
        return `<span class="badge bg-${color}">${text}</span>`;
    };

    // 通用API請求函數
    window.apiRequest = async function(url, method = 'GET', data = null) {
        try {
            const options = {
                method: method,
                headers: {
                    'Content-Type': 'application/json'
                }
            };

            if (data && method !== 'GET') {
                options.body = JSON.stringify(data);
            }

            const response = await fetch(url, options);
            const result = await response.json();

            if (!result.success) {
                throw new Error(result.message || '請求失敗');
            }

            return result;
        } catch (error) {
            console.error('API請求錯誤:', error);
            throw error;
        }
    };

    // 檢查當前頁面並初始化相應功能
    const currentPath = window.location.pathname;
    
    if (currentPath === '/consumer_input') {
        initConsumerInputPage();
    } else if (currentPath === '/classification') {
        initClassificationPage();
    } else if (currentPath === '/agent_training') {
        initAgentTrainingPage();
    } else if (currentPath === '/product_testing') {
        initProductTestingPage();
    } else if (currentPath === '/crm_integration') {
        initCRMIntegrationPage();
    } else if (currentPath === '/system_test') {
        initSystemTestPage();
    }
});

// 消費者資料輸入頁面初始化
function initConsumerInputPage() {
    console.log('初始化消費者資料輸入頁面');
    
    // 載入消費者列表
    loadConsumerList();
    
    // 綁定表單提交事件
    const consumerForm = document.getElementById('consumerForm');
    if (consumerForm) {
        consumerForm.addEventListener('submit', function(e) {
            e.preventDefault();
            submitConsumerForm();
        });
    }
    
    // 綁定批量導入按鈕事件
    const importButton = document.getElementById('importButton');
    if (importButton) {
        importButton.addEventListener('click', function() {
            document.getElementById('importFile').click();
        });
    }
    
    // 綁定文件選擇事件
    const importFile = document.getElementById('importFile');
    if (importFile) {
        importFile.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                importConsumers(e.target.files[0]);
            }
        });
    }
}

// 分類系統頁面初始化
function initClassificationPage() {
    console.log('初始化分類系統頁面');
    
    // 綁定K-means表單提交事件
    const kmeansForm = document.getElementById('kmeansForm');
    if (kmeansForm) {
        kmeansForm.addEventListener('submit', function(e) {
            e.preventDefault();
            trainKMeans();
        });
    }
    
    // 綁定DBSCAN表單提交事件
    const dbscanForm = document.getElementById('dbscanForm');
    if (dbscanForm) {
        dbscanForm.addEventListener('submit', function(e) {
            e.preventDefault();
            trainDBSCAN();
        });
    }
    
    // 綁定層次聚類表單提交事件
    const hierarchicalForm = document.getElementById('hierarchicalForm');
    if (hierarchicalForm) {
        hierarchicalForm.addEventListener('submit', function(e) {
            e.preventDefault();
            trainHierarchical();
        });
    }
    
    // 綁定尋找最佳聚類數量按鈕事件
    const findOptimalClustersButton = document.getElementById('findOptimalClustersButton');
    if (findOptimalClustersButton) {
        findOptimalClustersButton.addEventListener('click', function() {
            findOptimalClusters();
        });
    }
}

// AI代理訓練頁面初始化
function initAgentTrainingPage() {
    console.log('初始化AI代理訓練頁面');
    
    // 載入代理列表
    loadAgentList();
    
    // 綁定從消費者創建代理表單提交事件
    const createAgentForm = document.getElementById('createAgentForm');
    if (createAgentForm) {
        createAgentForm.addEventListener('submit', function(e) {
            e.preventDefault();
            createAgentFromConsumer();
        });
    }
    
    // 綁定從分類群組創建代理表單提交事件
    const createAgentsFromGroupForm = document.getElementById('createAgentsFromGroupForm');
    if (createAgentsFromGroupForm) {
        createAgentsFromGroupForm.addEventListener('submit', function(e) {
            e.preventDefault();
            createAgentsFromGroup();
        });
    }
    
    // 綁定代理互動表單提交事件
    const agentInteractionForm = document.getElementById('agentInteractionForm');
    if (agentInteractionForm) {
        agentInteractionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            sendMessageToAgent();
        });
    }
}

// 產品測試頁面初始化
function initProductTestingPage() {
    console.log('初始化產品測試頁面');
    
    // 載入代理列表
    loadAgentListForTesting();
    
    // 綁定產品測試表單提交事件
    const productTestForm = document.getElementById('productTestForm');
    if (productTestForm) {
        productTestForm.addEventListener('submit', function(e) {
            e.preventDefault();
            testProduct();
        });
    }
    
    // 綁定代理選擇類型變更事件
    const agentSelectionType = document.getElementById('agentSelectionType');
    if (agentSelectionType) {
        agentSelectionType.addEventListener('change', function() {
            toggleAgentSelectionFields();
        });
    }
    
    // 初始化代理選擇字段顯示
    toggleAgentSelectionFields();
}

// CRM整合頁面初始化
function initCRMIntegrationPage() {
    console.log('初始化CRM整合頁面');
    
    // 綁定測試連接表單提交事件
    const testConnectionForm = document.getElementById('testConnectionForm');
    if (testConnectionForm) {
        testConnectionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            testCRMConnection();
        });
    }
    
    // 綁定獲取客戶按鈕事件
    const getCustomersButton = document.getElementById('getCustomersButton');
    if (getCustomersButton) {
        getCustomersButton.addEventListener('click', function() {
            getCRMCustomers();
        });
    }
    
    // 綁定導入客戶按鈕事件
    const importCustomersButton = document.getElementById('importCustomersButton');
    if (importCustomersButton) {
        importCustomersButton.addEventListener('click', function() {
            importCRMCustomers();
        });
    }
    
    // 綁定同步資料表單提交事件
    const syncDataForm = document.getElementById('syncDataForm');
    if (syncDataForm) {
        syncDataForm.addEventListener('submit', function(e) {
            e.preventDefault();
            syncCRMData();
        });
    }
}

// 系統測試頁面初始化
function initSystemTestPage() {
    console.log('初始化系統測試頁面');
    
    // 綁定運行測試表單提交事件
    const systemTestForm = document.getElementById('systemTestForm');
    if (systemTestForm) {
        systemTestForm.addEventListener('submit', function(e) {
            e.preventDefault();
            runSystemTest();
        });
    }
}
