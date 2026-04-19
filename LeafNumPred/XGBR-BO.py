import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import time
from bayes_opt import BayesianOptimization
import warnings
import matplotlib

matplotlib.rc("font", family='YouYuan')

# 忽略警告
warnings.filterwarnings('ignore')

# 加载训练集和测试集
data_test = pd.read_csv('./dataset/data_testset.csv', encoding='utf-8-sig')
data_train = pd.read_csv('./dataset/data_trainset.csv', encoding='utf-8-sig')

X_train = data_train.iloc[:, :5].values
y_train = data_train.iloc[:, 5].values

X_test = data_test.iloc[:, :5].values
y_test = data_test.iloc[:, 5].values

feature_names = list(data_train.columns[:5])

# 创建全局变量存储交叉验证结果
cv_results = []


# 定义贝叶斯优化的目标函数（使用十折交叉验证）
def xgb_cv(n_estimators, learning_rate, max_depth, min_child_weight,
           subsample, colsample_bytree, gamma, reg_alpha, reg_lambda):
    """XGBoost交叉验证函数，使用十折交叉验证"""
    # 转换参数类型
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': int(n_estimators),
        'learning_rate': max(learning_rate, 0.01),  # 确保学习率不低于0.01
        'max_depth': int(max_depth),
        'min_child_weight': int(min_child_weight),
        'subsample': max(min(subsample, 1), 0.5),  # 保持在0.5-1.0之间
        'colsample_bytree': max(min(colsample_bytree, 1), 0.5),
        'gamma': gamma,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'seed': 42,
        'n_jobs': -1
    }

    # 使用十折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_rmse_scores = []
    cv_mae_scores = []
    cv_r2_scores = []

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # 创建DMatrix
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

        # 训练模型
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=int(n_estimators),
            evals=[(dval, 'eval')],
            early_stopping_rounds=10,
            verbose_eval=False,
        )

        # 在验证集上预测
        y_pred = model.predict(dval)
        # 计算评估指标
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        mae = mean_absolute_error(y_val_fold, y_pred)
        r2 = r2_score(y_val_fold, y_pred)

        cv_rmse_scores.append(rmse)
        cv_mae_scores.append(mae)
        cv_r2_scores.append(r2)

    # 计算平均指标
    avg_rmse = np.mean(cv_rmse_scores)
    avg_mae = np.mean(cv_mae_scores)
    avg_r2 = np.mean(cv_r2_scores)

    # 存储当前参数组合的评估结果
    cv_results.append({
        'params': params.copy(),
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'avg_r2': avg_r2,
        'cv_rmse_scores': cv_rmse_scores,
        'cv_mae_scores': cv_mae_scores,
        'cv_r2_scores': cv_r2_scores
    })

    # 打印当前参数组合的评估结果
    print(f"\n当前参数组合评估结果:")
    print(f"平均RMSE: {avg_rmse:.4f}, 平均MAE: {avg_mae:.4f}, 平均R²: {avg_r2:.4f}")

    # 返回平均RMSE的负值
    return -avg_rmse


# 定义参数边界
pbounds = {
    'n_estimators': (10, 1000),
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 50),
    'min_child_weight': (1, 10),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'gamma': (0, 5),
    'reg_alpha': (0, 10),
    'reg_lambda': (0.1, 10)
}

# 创建优化器
optimizer = BayesianOptimization(
    f=xgb_cv,
    pbounds=pbounds,
    random_state=42,
)

# 执行贝叶斯优化（记录时间）
print("开始贝叶斯优化...")
start_time = time.time()
optimizer.maximize(
    init_points=10,  # 初始随机点
    n_iter=100,  # 贝叶斯优化迭代次数
)
end_time = time.time()
print(f"贝叶斯优化耗时: {end_time - start_time:.2f} 秒")

# 获取最佳参数
best_params = optimizer.max['params']
print("\n最佳参数组合:")
for key, value in best_params.items():
    print(f"{key}: {value}")

# 转换参数类型
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_child_weight'] = int(best_params['min_child_weight'])
best_params['objective'] = 'reg:squarederror'

# 使用最佳参数训练最终模型
print("\n使用最佳参数训练最终模型...")

# ===== 转换数据格式 =====
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# ===== 参数拆分 =====
params = best_params.copy()
num_boost_round = params.pop('n_estimators')  # 🔥 关键：拿出来

params['objective'] = 'reg:squarederror'
params['eval_metric'] = 'rmse'

# ===== 训练模型 =====
final_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round
)

# 在测试集上评估
y_pred = final_model.predict(dtest)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n测试集评估结果:")
print(f"均方误差(MSE): {mse:.4f}")
print(f"均方根误差(RMSE): {rmse:.4f}")
print(f"平均绝对误差(MAE): {mae:.4f}")
print(f"决定系数(R²): {r2:.4f}")

# 特征重要性分析
importance_dict = final_model.get_score(importance_type='gain')

feature_importances = np.array([
    importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))
])
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print("\n特征重要性排序:")
print(importance_df)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('重要性分数')
plt.title('XGBoost特征重要性')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('./img/xgbr_feature_importance_bayesian.png', dpi=300)
plt.show()

# 绘制贝叶斯优化过程 - 显示实际RMSE值
target_values = [-x for x in optimizer.space.target]  # 取负值得到实际RMSE

plt.figure(figsize=(12, 8))
plt.plot(range(1, len(target_values) + 1), target_values, 'o-')
plt.xlabel('迭代次数')
plt.ylabel('RMSE')
plt.title('XGB贝叶斯优化过程 (RMSE)')
plt.grid(True)

# 标记最佳点
best_iteration = np.argmin(target_values)  # 找到最小RMSE对应的迭代
best_rmse = target_values[best_iteration]
plt.scatter(best_iteration + 1, best_rmse, color='red', s=100, zorder=5)
plt.annotate(f'Optimal value: {best_rmse:.4f}',
             (best_iteration + 1, best_rmse),
             textcoords="offset points",
             xytext=(0, 10),
             ha='center',
             fontsize=12)

plt.tight_layout()
plt.savefig('./img/xgbr_bayesian_optimization_progress.png', dpi=300)
plt.show()

# 保存最佳模型、训练集和测试集
joblib.dump({
    'model': final_model,
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test,
    'y_test': y_test,
    'feature_names': feature_names
}, './model/best_xgbr_model_bayesian.pkl')
joblib.dump(final_model, './model/xgbr+bo.pkl')
print("\n最佳模型已保存为 './model/best_xgbr_model_bayesian.pkl' (包含训练集和测试集)")

# 保存优化过程数据
optimization_history = pd.DataFrame(optimizer.res)
optimization_history.to_csv('./result/xgbr_bayesian_optimization_history.csv', index=False)
print("优化过程历史已保存为 './result/xgbr_bayesian_optimization_history.csv'")

# 保存交叉验证结果
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv('./result/xgbr_cross_validation_results.csv', index=False)
print("交叉验证结果已保存为 './result/xgbr_cross_validation_results.csv'")

# 输出最佳交叉验证结果
best_cv_result = cv_results_df.loc[cv_results_df['avg_rmse'].idxmin()]
print("\n最佳交叉验证结果:")
print(f"平均RMSE: {best_cv_result['avg_rmse']:.4f}")
print(f"平均MAE: {best_cv_result['avg_mae']:.4f}")
print(f"平均R²: {best_cv_result['avg_r2']:.4f}")

# 模型解释：SHAP值
try:
    import shap

    print("\n生成SHAP解释...")

    # ⭐ 关键：包装预测函数
    def model_predict(X):
        return final_model.predict(xgb.DMatrix(X))

    explainer = shap.Explainer(model_predict, X_train)
    shap_values = explainer(X_train)

    # ===== summary =====
    plt.figure()
    shap.summary_plot(shap_values.values, X_train, feature_names=feature_names)
    plt.tight_layout()
    plt.savefig('./img/xgbr_shap_summary_bayesian.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ===== waterfall =====
    sample_idx = 0
    plt.figure()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[sample_idx].values,
            base_values=shap_values[sample_idx].base_values,
            data=X_train[sample_idx],
            feature_names=feature_names
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig('./img/xgbr_shap_waterfall_bayesian.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ===== SHAP重要性 =====
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    shap_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'MeanAbsSHAP': mean_abs_shap
    }).sort_values(by='MeanAbsSHAP', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(shap_importance_df['Feature'], shap_importance_df['MeanAbsSHAP'])
    plt.xlabel('平均|SHAP值|')
    plt.title('特征重要性（基于SHAP）')
    plt.tight_layout()
    plt.savefig('./img/xgbr_shap_bar_importance.png', dpi=300)
    plt.show()
    shap_importance_df.to_csv('./result/xgbr_shap_feature_importance.csv', index=False)

    # 画聚类热力图（样本和特征都聚类排序）
    plt.figure(figsize=(12, 10))
    shap.plots.heatmap(shap_values, max_display=len(feature_names))
    plt.tight_layout()
    plt.savefig('./img/xgbr_shap_clustered_heatmap.png', dpi=300)
    shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
    shap_df.to_csv('./result/xgbr_shap_clustered_values.csv', index=False)

    print("SHAP解释已保存为图片")

except ImportError:
    print("\n未安装SHAP库，跳过SHAP分析")

