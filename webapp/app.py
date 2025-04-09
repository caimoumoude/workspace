from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import json
from datetime import datetime

app = Flask(__name__)

# 配置上传文件存储路径
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pcap'}
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), UPLOAD_FOLDER)

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 验证文件扩展名
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 首页/仪表盘路由
@app.route('/')
def index():
    return render_template('index.html')

# 在线流量可视化页面
@app.route('/online_traffic')
def online_traffic():
    return render_template('online_traffic.html')

# 本地PCAP文件导入页面
@app.route('/pcap_import', methods=['GET', 'POST'])
def pcap_import():
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            return render_template('pcap_import.html', error='未上传文件')
        
        file = request.files['file']
        
        # 检查文件名是否为空
        if file.filename == '':
            return render_template('pcap_import.html', error='未选择文件')
        
        # 检查文件扩展名
        if file and allowed_file(file.filename):
            # 保存文件
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # TODO: 分析PCAP文件的代码会在这里实现
            # 暂时使用模拟数据
            analysis_results = {
                'total_malicious': 5,
                'traffic': [
                    {
                        'timestamp': '2023-04-09 10:30:45',
                        'ip': '192.168.1.100',
                        'port': 8080,
                        'confidence': 0.95,
                        'details': '可疑DNS请求'
                    },
                    {
                        'timestamp': '2023-04-09 10:32:15',
                        'ip': '192.168.1.105',
                        'port': 443,
                        'confidence': 0.88,
                        'details': '异常HTTP请求'
                    },
                    {
                        'timestamp': '2023-04-09 10:35:20',
                        'ip': '192.168.1.110',
                        'port': 22,
                        'confidence': 0.75,
                        'details': '可疑SSH连接'
                    },
                    {
                        'timestamp': '2023-04-09 10:40:10',
                        'ip': '192.168.1.100',
                        'port': 53,
                        'confidence': 0.92,
                        'details': '可疑DNS请求'
                    },
                    {
                        'timestamp': '2023-04-09 10:45:30',
                        'ip': '192.168.1.120',
                        'port': 80,
                        'confidence': 0.85,
                        'details': '异常HTTP请求'
                    }
                ]
            }
            
            # 排序选项
            sort_by = request.form.get('sort_by', 'timestamp')
            sort_order = request.form.get('sort_order', 'desc')
            
            # 排序结果
            if sort_by == 'ip':
                analysis_results['traffic'].sort(key=lambda x: x['ip'], reverse=(sort_order == 'desc'))
            else:  # 默认按时间排序
                analysis_results['traffic'].sort(key=lambda x: x['timestamp'], reverse=(sort_order == 'desc'))
            
            return render_template('pcap_import.html', results=analysis_results, filename=file.filename)
        else:
            return render_template('pcap_import.html', error='不支持的文件类型，仅支持.pcap文件')
    
    return render_template('pcap_import.html')

# 设置页面
@app.route('/settings')
def settings():
    return render_template('settings.html')

# 日志页面
@app.route('/logs')
def logs():
    return render_template('logs.html')

if __name__ == '__main__':
    app.run(debug=True)