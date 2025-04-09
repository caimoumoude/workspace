import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import catboost as ctb
import pickle
import joblib

class AdvancedMetaModel:
    """
    高级元模型实现，包含:
    1. 交叉验证堆叠
    2. 特征增强堆叠
    3. 多样化基模型
    4. 概率校准
    """
    
    def __init__(self, n_folds=5, use_original_features=True):
        """
        初始化高级元模型
        
        参数:
        n_folds: 交叉验证折数
        use_original_features: 是否在元模型中使用原始特征
        """
        self.n_folds = n_folds
        self.use_original_features = use_original_features
        
        # 基础模型定义
        self.base_models = {
            'xgboost': None,  # 将在fit中加载或训练
            'catboost': None,
            'mlp': None,      # 新增的神经网络模型
            'lr': None        # 新增的逻辑回归模型
        }
        
        # 元模型
        self.meta_model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        
        # 特征缩放器，用于原始特征
        self.scaler = StandardScaler()
        
        # 每个模型的交叉验证预测
        self.oof_predictions = {}
        
        # 校准后的模型
        self.calibrated_models = {}
        
    def fit(self, X, y, feature_importances=None):
        """
        训练元模型
        
        参数:
        X: 原始特征
        y: 目标变量
        feature_importances: 特征重要性，可选，用于特征选择
        """
        print("开始训练高级元模型...")
        
        # 1. 加载或训练基础模型
        self._load_or_train_base_models(X, y)
        
        # 2. 使用交叉验证生成元特征
        meta_features = self._generate_meta_features(X, y)
        
        # 3. 如果启用原始特征，则添加到元特征中
        if self.use_original_features:
            # 选择重要特征(如果提供了特征重要性)
            if feature_importances is not None:
                top_features = pd.Series(feature_importances).sort_values(ascending=False).head(20).index
                X_selected = X.iloc[:, top_features]
            else:
                X_selected = X
                
            # 标准化原始特征
            X_scaled = self.scaler.fit_transform(X_selected)
            # 合并基模型预测和原始特征
            meta_features = np.hstack([meta_features, X_scaled])
            
        # 4. 训练元模型
        self.meta_model.fit(meta_features, y)
        
        # 5. 模型校准
        self._calibrate_models(X, y)
        
        print("元模型训练完成！")
        return self
    
    def _load_or_train_base_models(self, X, y):
        """加载现有模型或训练新模型"""
        try:
            # 尝试加载现有的XGBoost模型
            self.base_models['xgboost'] = xgb.XGBClassifier()
            self.base_models['xgboost'].load_model('xgb_model.json')
            print("已加载XGBoost模型")
        except:
            print("训练新的XGBoost模型...")
            self.base_models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=5
            )
            self.base_models['xgboost'].fit(X, y)
            
        try:
            # 尝试加载现有的CatBoost模型
            self.base_models['catboost'] = ctb.CatBoostClassifier(verbose=0)
            self.base_models['catboost'].load_model('cat_model.cbm')
            print("已加载CatBoost模型")
        except:
            print("训练新的CatBoost模型...")
            self.base_models['catboost'] = ctb.CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=5,
                verbose=0
            )
            self.base_models['catboost'].fit(X, y)
        
        # 训练多层感知机模型
        print("训练MLP模型...")
        self.base_models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=300,
            early_stopping=True
        )
        self.base_models['mlp'].fit(X, y)
        
        # 训练逻辑回归模型
        print("训练逻辑回归模型...")
        self.base_models['lr'] = LogisticRegression(C=1.0, max_iter=1000)
        self.base_models['lr'].fit(X, y)
    
    def _generate_meta_features(self, X, y):
        """使用交叉验证生成元特征"""
        print("生成交叉验证元特征...")
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # 初始化存储交叉验证预测的数组
        self.oof_predictions = {
            'xgboost': np.zeros((X.shape[0], 2)),
            'catboost': np.zeros((X.shape[0], 2)),
            'mlp': np.zeros((X.shape[0], 2)),
            'lr': np.zeros((X.shape[0], 2))
        }
        
        # 对每个折进行训练和预测
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"处理第 {fold+1}/{self.n_folds} 折")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 为每个模型克隆一个新实例
            fold_models = {
                'xgboost': xgb.XGBClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=5),
                'catboost': ctb.CatBoostClassifier(
                    iterations=100, learning_rate=0.1, depth=5, verbose=0),
                'mlp': MLPClassifier(
                    hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                    alpha=0.0001, max_iter=300, early_stopping=True),
                'lr': LogisticRegression(C=1.0, max_iter=1000)
            }
            
            # 训练每个折的模型
            for name, model in fold_models.items():
                model.fit(X_train, y_train)
                # 保存验证集上的预测概率
                self.oof_predictions[name][val_idx] = model.predict_proba(X_val)
                
        # 合并所有模型的OOF预测作为元特征
        meta_features = np.hstack([
            self.oof_predictions['xgboost'],
            self.oof_predictions['catboost'],
            self.oof_predictions['mlp'],
            self.oof_predictions['lr']
        ])
        
        return meta_features
    
    def _calibrate_models(self, X, y):
        """校准模型概率输出"""
        print("进行概率校准...")
        for name, model in self.base_models.items():
            self.calibrated_models[name] = CalibratedClassifierCV(
                base_estimator=model, cv='prefit', method='isotonic'
            )
            self.calibrated_models[name].fit(X, y)
    
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X):
        """预测概率"""
        # 获取所有基础模型的校准预测
        all_preds = []
        for name, model in self.calibrated_models.items():
            all_preds.append(model.predict_proba(X))
            
        # 合并所有预测
        meta_features = np.hstack(all_preds)
        
        # 如果使用原始特征，添加到元特征
        if self.use_original_features:
            if hasattr(X, 'iloc'):
                X_selected = X
            else:
                X_selected = pd.DataFrame(X)
                
            X_scaled = self.scaler.transform(X_selected)
            meta_features = np.hstack([meta_features, X_scaled])
            
        # 使用元模型进行最终预测
        return self.meta_model.predict_proba(meta_features)
    
    def save(self, filepath='advanced_meta_model.pkl'):
        """保存模型"""
        joblib.dump(self, filepath)
        print(f"模型已保存到 {filepath}")
        
    @classmethod
    def load(cls, filepath='advanced_meta_model.pkl'):
        """加载模型"""
        return joblib.load(filepath)

# 使用示例
if __name__ == "__main__":
    # 这里是完整的示例代码
    print("高级元模型使用示例:")
    
    # 1. 加载数据 (或创建示例数据)
    try:
        # 尝试加载真实数据
        data = pd.read_csv('merge_data.csv')
    except:
        # 创建示例数据用于演示
        print("创建示例数据...")
        import numpy as np
        from sklearn.datasets import make_classification
        
        # 生成示例数据
        X_sample, y_sample = make_classification(
            n_samples=1000, n_features=20, n_classes=2, 
            random_state=42, n_informative=10
        )
        data = pd.DataFrame(X_sample, columns=[f'feature_{i}' for i in range(20)])
        data['Label'] = y_sample
    
    # 2. 准备训练和测试数据
    from sklearn.model_selection import train_test_split
    X = data.drop('Label', axis=1)
    y = data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 3. 初始化高级元模型
    print("\n初始化并训练高级元模型...")
    meta = AdvancedMetaModel(n_folds=3, use_original_features=True)  # 使用更少的折数加快示例运行
    
    # 4. 训练模型
    meta.fit(X_train, y_train)
    
    # 5. 保存模型
    meta.save('advanced_meta_model_example.pkl')
    
    # 6. 在测试集上进行预测
    print("\n在测试集上进行预测...")
    predictions = meta.predict(X_test)
    probabilities = meta.predict_proba(X_test)
    
    # 7. 评估模型性能
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, predictions)
    print(f"测试集上的准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, predictions))
    
    # 8. 单个样本预测示例
    print("\n单个样本预测示例:")
    sample_data = X_test.iloc[0:1]  # 使用测试集的第一个样本
    sample_pred = meta.predict(sample_data)
    sample_proba = meta.predict_proba(sample_data)
    print(f"预测类别: {sample_pred[0]}")
    print(f"预测概率: {sample_proba[0]}")
    
    print("\n高级元模型示例完成") 