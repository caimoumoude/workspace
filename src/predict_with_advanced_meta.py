import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from advanced_meta_model import AdvancedMetaModel
from config.cicconfig import mapping  # 假设存在这个映射配置

def preprocess_data(data):
    """预处理输入数据"""
    # 处理列名映射
    data = data.rename(columns=mapping)
    
    # 处理数值型列
    for col in data.columns:
        if col != 'Label':
            try:
                series = pd.to_numeric(data[col], errors='coerce')
                series = series.replace([np.inf, -np.inf], np.nan)
                series = series.fillna(0).astype(float)
                data[col] = series
            except Exception as e:
                print(f"处理列 {col} 时出错: {e}")
    
    return data

def predict_flows(input_file, output_file='malicious_flows_advanced.csv'):
    """
    使用高级元模型进行流量预测
    
    参数:
    input_file: 输入CSV文件路径
    output_file: 输出恶意流量CSV文件路径
    """
    try:
        print(f"加载数据: {input_file}")
        # 读取CSV文件
        data = pd.read_csv(input_file, encoding='gbk')
        
        # 预处理数据
        print("预处理数据...")
        data = preprocess_data(data)
        
        # 保存原始标签(如果有)
        has_label = 'Label' in data.columns
        if has_label:
            original_labels = data['Label'].copy()
            X = data.drop(['Label'], axis=1)
        else:
            X = data
        
        # 加载高级元模型
        print("加载高级元模型...")
        meta_model = AdvancedMetaModel.load('advanced_meta_model.pkl')
        
        # 进行预测
        print("进行预测...")
        predictions = meta_model.predict(X)
        probabilities = meta_model.predict_proba(X)
        
        # 添加预测结果到原始数据
        data_with_predictions = data.copy()
        data_with_predictions['Prediction'] = predictions
        data_with_predictions['Malicious_Probability'] = probabilities[:, 1]
        
        # 统计预测结果
        malicious_count = np.sum(predictions == 1)
        print(f"检测到的恶意流量数量: {malicious_count}")
        
        # 如果有原始标签，计算准确率
        if has_label:
            from sklearn.metrics import accuracy_score, classification_report
            accuracy = accuracy_score(original_labels, predictions)
            print(f"预测准确率: {accuracy:.4f}")
            print("\n分类报告:")
            print(classification_report(original_labels, predictions))
            
            # 绘制混淆矩阵
            from sklearn.metrics import confusion_matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(original_labels, predictions)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                       xticklabels=["良性", "恶意"], 
                       yticklabels=["良性", "恶意"])
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.title('高级元模型混淆矩阵')
            plt.savefig('advanced_confusion_matrix.png')
            print("混淆矩阵已保存到 'advanced_confusion_matrix.png'")
        
        # 筛选出预测为恶意的流量
        malicious_data = data_with_predictions[data_with_predictions['Prediction'] == 1]
        
        # 按恶意概率排序
        malicious_data = malicious_data.sort_values(by='Malicious_Probability', ascending=False)
        
        # 保存恶意流量数据
        malicious_data.to_csv(output_file, index=False)
        print(f"恶意流量数据已保存到 '{output_file}'")
        
        # 输出恶意流量示例
        if len(malicious_data) > 0:
            print("\n恶意流量样例:")
            pd.set_option('display.max_columns', 10)
            print(malicious_data.head(3))
        else:
            print("未检测到恶意流量")
        
        return malicious_data
        
    except Exception as e:
        print(f"预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'malicious_flows_advanced.csv'
        predict_flows(input_file, output_file)
    else:
        print("使用方法: python predict_with_advanced_meta.py <输入文件路径> [输出文件路径]") 