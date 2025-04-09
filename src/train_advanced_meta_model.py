import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from advanced_meta_model import AdvancedMetaModel

def main():
    print("开始训练高级元模型...")
    
    # 1. 加载数据
    try:
        print("加载数据...")
        # 这里需要替换为实际的数据集路径
        data = pd.read_csv('your_dataset.csv', encoding='gbk')
        
        # 根据原始代码中的映射重命名列
        # 如果需要映射，取消下面注释并提供映射字典
        # from config.cicconfig import mapping
        # data = data.rename(columns=mapping)
        
        # 2. 数据预处理
        print("数据预处理...")
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
                    
        # 准备特征和标签
        X = data.drop(['Label'], axis=1, errors='ignore')
        # 假设Label列是二分类标签(0或1)
        y = data['Label']
        
        # 3. 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 获取特征重要性(可选)
        # 如果已有XGBoost模型，可以从中获取特征重要性
        feature_importances = None
        # 如果前面有训练好的XGBoost模型
        # try:
        #     import xgboost as xgb
        #     xgb_model = xgb.XGBClassifier()
        #     xgb_model.load_model('xgb_model.json')
        #     feature_importances = xgb_model.feature_importances_
        # except:
        #     pass
        
        # 4. 初始化并训练高级元模型
        print("初始化高级元模型...")
        meta = AdvancedMetaModel(n_folds=5, use_original_features=True)
        
        print("训练高级元模型...")
        meta.fit(X_train, y_train, feature_importances=feature_importances)
        
        # 5. 模型评估
        print("模型评估...")
        y_pred = meta.predict(X_test)
        probas = meta.predict_proba(X_test)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        # 如果是二分类
        if probas.shape[1] == 2:
            auc = roc_auc_score(y_test, probas[:, 1])
            print(f"准确率: {accuracy:.4f}, AUC: {auc:.4f}")
        else:
            print(f"准确率: {accuracy:.4f}")
            
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=["良性", "恶意"], 
                   yticklabels=["良性", "恶意"])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.savefig('confusion_matrix.png')
        
        # 6. 保存模型
        meta.save('advanced_meta_model.pkl')
        print("高级元模型已保存到 'advanced_meta_model.pkl'")
        
        # 7. 模型推理示例
        print("\n模型推理示例:")
        # 使用测试集前几行数据进行预测
        sample_data = X_test.head(5)
        predictions = meta.predict(sample_data)
        probabilities = meta.predict_proba(sample_data)
        
        print("预测结果示例:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            print(f"样本 {i+1}: 预测类别 = {pred}, 预测概率 = {prob}")
            
        print("\n高级元模型训练与评估完成!")
        
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 