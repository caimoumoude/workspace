import xgboost as xgb
import catboost as ctb
import pickle
import pandas as pd
import numpy as np
import sys
import os
import glob
print(sys.path)
from config.cicconfig import mapping
# from src.datasetconfig import mapping

# 加载XGBoost模型
clf = xgb.XGBClassifier()
clf.load_model('model/xgb_model.json')

# 加载CatBoost模型
cat_clf = ctb.CatBoostClassifier()
cat_clf.load_model('model/cat_model.cbm')

# 加载元模型
with open('model/meta_model.pkl', 'rb') as f:
    meta_model = pickle.load(f)

# 获取最新的流量文件
def get_latest_flow_file():
    files = glob.glob('dir/original_dir/*_Flow.csv')
    if not files:
        raise FileNotFoundError("未找到任何流量文件！")
    
    # 按文件创建时间排序，返回最新文件
    latest_file = max(files, key=os.path.getctime)
    print(f"处理最新的流量文件: {latest_file}")
    return latest_file

# 获取最新流量文件并读取数据
latest_flow_file = get_latest_flow_file()
data = pd.read_csv(latest_flow_file, encoding='gbk')  #自动获取最新的流量文件
# data.columns = data.columns.str.strip()
# data = data.drop(['Fwd Header Length.1'], axis=1, errors='ignore')
# 重命名列
data = data.rename(columns=mapping)


# 转换数据类型
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

X_new = data.drop(['Label'], axis=1, errors='ignore')


# 获取基模型的预测概率
xgb_pred_proba = clf.predict_proba(X_new)
cat_pred_proba = cat_clf.predict_proba(X_new)
print(xgb_pred_proba)
print(cat_pred_proba)
# 获取各模型的预测结果
xgb_pred = clf.predict(X_new)
cat_pred = cat_clf.predict(X_new)

# 实现更专业的模型集成方法

# 1. 创建更丰富的元特征
meta_features = np.column_stack((
    xgb_pred_proba[:, 1],  # XGBoost的第二类(恶意)预测概率
    cat_pred_proba[:, 1],  # CatBoost的第二类(恶意)预测概率
    np.mean([xgb_pred_proba[:, 1], cat_pred_proba[:, 1]], axis=0),  # 两个模型概率的平均值
    np.abs(xgb_pred_proba[:, 1] - cat_pred_proba[:, 1])  # 两个模型预测差异
))

# 2. 使用元模型进行预测
meta_pred = meta_model.predict(meta_features)

# 3. 计算模型性能动态权重（基于置信度）
xgb_confidence = np.max(xgb_pred_proba, axis=1)
cat_confidence = np.max(cat_pred_proba, axis=1)

# 标准化置信度作为权重
total_confidence = xgb_confidence + cat_confidence
xgb_weight = xgb_confidence / total_confidence
cat_weight = cat_confidence / total_confidence

# 4. 加权投票集成
weighted_proba = np.zeros_like(xgb_pred_proba)
for i in range(len(X_new)):
    weighted_proba[i] = xgb_weight[i] * xgb_pred_proba[i] + cat_weight[i] * cat_pred_proba[i]

# 确定最终预测（使用加权概率中最大值对应的类别）
weighted_pred = np.argmax(weighted_proba, axis=1)

# 5. 多种集成策略融合
# 5.1 加权投票
# 5.2 元模型预测
# 5.3 最大置信度模型选择
max_conf_pred = np.where(xgb_confidence > cat_confidence, xgb_pred, cat_pred)

# 使用投票机制确定最终预测
# 创建投票矩阵（每行表示一个样本，每列表示一个预测方法的预测结果）
votes = np.column_stack((weighted_pred, meta_pred, max_conf_pred))
# 计算每个样本获得的最多票数的类别（众数）
from scipy import stats
final_pred = stats.mode(votes, axis=1)[0].flatten()

# 如果需要，可以为重要的决策添加额外的置信度检查
# 例如，对于高风险预测（预测为恶意流量），增加额外的验证步骤
for i in range(len(final_pred)):
    if final_pred[i] == 1:  # 如果预测为恶意
        # 如果所有模型的恶意预测概率都低于某个阈值，重新考虑
        if xgb_pred_proba[i, 1] < 0.6 and cat_pred_proba[i, 1] < 0.6:
            # 重新评估或保守预测为良性
            if xgb_pred_proba[i, 1] + cat_pred_proba[i, 1] < 1.0:
                final_pred[i] = 0  # 更保守的预测

# 统计各模型恶性流量的数量
# 假设1为恶性流量标签
xgb_malicious_count = np.sum(xgb_pred == 1)
cat_malicious_count = np.sum(cat_pred == 1)
meta_malicious_count = np.sum(final_pred == 1)

# 输出各模型的恶性流量数量
print(f'XGBoost模型检测到的恶性流量数量: {xgb_malicious_count}')
print(f'CatBoost模型检测到的恶性流量数量: {cat_malicious_count}')
print(f'元模型检测到的恶性流量数量: {meta_malicious_count}')



# 从最后一个代码单元格继续，创建一个包含预测结果的数据框
data_with_predictions = data.copy()
data_with_predictions['Prediction'] = final_pred

# 筛选出预测为恶意的流量 (label=1 表示恶意)
malicious_data = data_with_predictions[data_with_predictions['Prediction'] == 1]

# 打印恶意流量数据
print(f"恶意流量数量: {len(malicious_data)}")
print("\n恶意流量数据样例:")
print(malicious_data.head())

# 从原始文件名中提取日期部分
file_name = os.path.basename(latest_flow_file)
date_part = file_name.split('_')[0]  # 例如：2025-04-09

# 保存恶意流量数据到对应日期的结果文件
result_file = f'dir/result_dir/{date_part}_Flow_result.csv'
file_exists = os.path.isfile(result_file)

# 以添加方式保存结果
if len(malicious_data) > 0:
    malicious_data.to_csv(result_file, mode='a', index=False, header=not file_exists)
    print(f"\n恶意流量数据已保存到 '{result_file}'")

