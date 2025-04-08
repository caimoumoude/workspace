import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def main():
    # 1. 数据加载
    
    data = pd.read_csv("datae.csv")
    
    # 2. 特征方差筛选
    print("\n=== 特征方差筛选 ===")
    variance_threshold = 0.01
    variances = data.var(numeric_only=True)
    low_variance_features = variances[variances < variance_threshold].index.tolist()
    print(f"发现 {len(low_variance_features)} 个低方差特征（方差<{variance_threshold}）:")
    print(low_variance_features)
    
    # 3. 特征相关性分析
    print("\n=== 特征相关性分析 ===")
    plt.figure(figsize=(15, 12))
    corr_matrix = data.corr(method='pearson')
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title('特征Pearson相关系数矩阵热力图')
    plt.show()
    
 
    


# Convert all columns (except 'Label') to numeric types
    for col in data.columns:
        if col != 'Label':
            try:
                series = pd.to_numeric(data[col], errors='coerce')
                series = series.replace([np.inf, -np.inf], np.nan)
                series = series.fillna(0).astype(int)
                data[col] = series
            except Exception as e:
                print(f"Error converting column {col}: {e}")

    print("Data types after conversion:")
    print(data.dtypes)

    # Cell 5: 标签映射
    data['Label'] = data['Label'].map(lambda x: 0 if x == 'Benign' else 1)
    print("Label column af ter mapping:")
    print(data['Label'].head(10))
    #4. XGBoost特征重要性分析
    print("\n=== XGBoost特征重要性分析 ===")
    # 数据预处理
    X = data.drop('Label', axis=1)
    y = data['Label']

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 训练XGBoost模型
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # 特征重要性可视化
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(50))
    plt.title('Top 20重要特征')
    plt.xlabel('重要性分数')
    plt.ylabel('特征名称')
    plt.show()
    print(feature_importance)
if __name__ == "__main__":
    main()
