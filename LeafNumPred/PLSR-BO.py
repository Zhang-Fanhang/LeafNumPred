import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from bayes_opt import BayesianOptimization
import warnings
import matplotlib
import joblib

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

# 数据标准化 - PLSR对特征尺度敏感
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 创建全局变量存储交叉验证结果
cv_results = []

def plsr_cv(n_components):
    """PLSR 交叉验证函数，使用十折交叉验证"""
    # 限定成分数范围
    n_components = int(max(min(n_components, X_train.shape[1]), 1))

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_rmse_scores, cv_mae_scores, cv_r2_scores = [], [], []

    for train_index, val_index in kf.split(X_train):
        # 折内原始数据划分
        X_tr_raw, X_val_raw = X_train[train_index], X_train[val_index]
        y_tr_raw, y_val_raw = y_train[train_index], y_train[val_index]

        # 特征标准化（折内）
        scaler_X_fold = StandardScaler()
        X_tr = scaler_X_fold.fit_transform(X_tr_raw)
        X_val = scaler_X_fold.transform(X_val_raw)

        # 响应变量标准化（折内）
        scaler_y_fold = StandardScaler()
        y_tr = scaler_y_fold.fit_transform(y_tr_raw.reshape(-1,1)).ravel()

        # 训练模型
        model = PLSRegression(n_components=n_components, scale=False)
        model.fit(X_tr, y_tr)

        # 验证集预测 & 反标准化
        y_pred = scaler_y_fold.inverse_transform(
            model.predict(X_val).ravel().reshape(-1,1)
        ).ravel()

        # 计算指标（原始尺度）
        cv_rmse_scores.append(np.sqrt(mean_squared_error(y_val_raw, y_pred)))
        cv_mae_scores.append(mean_absolute_error(y_val_raw, y_pred))
        cv_r2_scores.append(r2_score(y_val_raw, y_pred))

    # 平均指标
    avg_rmse = np.mean(cv_rmse_scores)
    avg_mae = np.mean(cv_mae_scores)
    avg_r2 = np.mean(cv_r2_scores)

    cv_results.append({
        'params': {'n_components': n_components},
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'avg_r2': avg_r2,
        'rmse_scores': cv_rmse_scores,
        'mae_scores': cv_mae_scores,
        'r2_scores': cv_r2_scores
    })

    print(f"\nPLSR (n_components={n_components}) 10-fold CV -> "
          f"RMSE={avg_rmse:.4f}, MAE={avg_mae:.4f}, R²={avg_r2:.4f}")

    return -avg_rmse


# 定义参数边界
pbounds = {
    'n_components': (1, min(20, X_train.shape[1])),  # 成分数范围
}

# 创建优化器
optimizer = BayesianOptimization(
    f=plsr_cv,
    pbounds=pbounds,
    random_state=42,
    verbose=2  # 显示详细输出
)

# 执行贝叶斯优化（记录时间）
print("开始贝叶斯优化...")
start_time = time.time()

# 确保执行完整的100次迭代
optimizer.maximize(
    init_points=10,  # 初始随机点
    n_iter=100,  # 贝叶斯优化迭代次数
)

end_time = time.time()
print(f"贝叶斯优化耗时: {end_time - start_time:.2f} 秒")
print(f"完成迭代次数: {len(optimizer.res)}")

# 获取最佳参数
best_params = optimizer.max['params']
best_n_components = int(best_params['n_components'])
print("\n最佳参数组合:")
print(f"n_components: {best_n_components}")

# 使用最佳参数训练最终模型（使用整个训练集）
print("\n使用最佳参数训练最终模型...")

# 标准化整个训练集
scaler_X_final = StandardScaler()
X_train_scaled = scaler_X_final.fit_transform(X_train)
scaler_y_final = StandardScaler()
y_train_scaled = scaler_y_final.fit_transform(y_train.reshape(-1, 1)).ravel()

final_model = PLSRegression(
    n_components=best_n_components,
    scale=False
)
final_model.fit(X_train_scaled, y_train_scaled)

# 在测试集上评估
X_test_scaled = scaler_X_final.transform(X_test)
y_pred_scaled = final_model.predict(X_test_scaled).ravel()
y_pred = scaler_y_final.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# 计算评估指标（使用原始尺度）
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n测试集评估结果:")
print(f"均方误差(MSE): {mse:.4f}")
print(f"均方根误差(RMSE): {rmse:.4f}")
print(f"平均绝对误差(MAE): {mae:.4f}")
print(f"决定系数(R²): {r2:.4f}")


# 特征重要性分析（使用变量投影重要性VIP）
def calculate_vip(model):
    """计算变量投影重要性(VIP)"""
    t = model.x_scores_  # 得分矩阵
    w = model.x_weights_  # 权重矩阵
    q = model.y_loadings_  # Y载荷

    # 计算每个成分的SSY
    ssy = np.sum((model.y_scores_ @ q.T) ** 2, axis=0)

    # 计算VIP分数
    vip_scores = np.zeros((X_train.shape[1],))
    for i in range(X_train.shape[1]):
        numerator = np.sum(ssy * (w[i, :] ** 2))
        denominator = np.sum(ssy)
        vip_scores[i] = np.sqrt(X_train.shape[1] * numerator / denominator)

    return vip_scores


# 计算VIP分数
vip_scores = calculate_vip(final_model)
importance_df = pd.DataFrame({
    '特征': feature_names,
    'VIP分数': vip_scores
}).sort_values('VIP分数', ascending=False)

print("\n特征重要性排序(VIP分数):")
print(importance_df)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(importance_df['特征'], importance_df['VIP分数'])
plt.xlabel('VIP分数', fontsize=12)
plt.ylabel('特征', fontsize=12)
plt.title('PLSR特征重要性 (VIP)', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('./img/plsr_vip_importance_bayesian.png', dpi=300)
plt.show()

# 绘制贝叶斯优化过程
# 提取目标函数值（负RMSE）并转换为实际RMSE
target_values = [-x for x in optimizer.space.target]  # 取负值得到实际RMSE

plt.figure(figsize=(12, 8))
plt.plot(range(1, len(target_values) + 1), target_values, 'o-', linewidth=2)
plt.xlabel('迭代次数', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('PLSR贝叶斯优化过程 (RMSE)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

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
plt.savefig('./img/plsr_bayesian_optimization_progress.png', dpi=300)
plt.show()

# # 绘制实际值与预测值对比图
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', linewidth=2)
# plt.xlabel('实际值', fontsize=12)
# plt.ylabel('预测值', fontsize=12)
# plt.title('实际值 vs 预测值', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig('plsr_actual_vs_predicted.png', dpi=300)
# plt.show()

# # 绘制预测残差图
# residuals = y_test - y_pred
# plt.figure(figsize=(10, 6))
# plt.scatter(y_pred, residuals, alpha=0.6, color='green')
# plt.axhline(y=0, color='r', linestyle='--')
# plt.xlabel('预测值', fontsize=12)
# plt.ylabel('残差', fontsize=12)
# plt.title('预测值 vs 残差', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig('plsr_residual_plot.png', dpi=300)
# plt.show()

# 保存模型系数
coefs = final_model.coef_.ravel()
coef_df = pd.DataFrame({
    '特征': feature_names,
    '系数': coefs
})
coef_df.to_csv('./result/plsr_coefficients_bayesian.csv', index=False, encoding='utf-8-sig')
print("\n模型系数已保存为 './result/plsr_coefficients_bayesian.csv'")

# 保存最佳参数
best_params_df = pd.DataFrame([{'n_components': best_n_components}])
best_params_df.to_csv('./result/plsr_best_params_bayesian.csv', index=False, encoding='utf-8-sig')
print("最佳参数已保存为 './result/plsr_best_params_bayesian.csv'")

# 保存最佳模型、训练集和测试集
joblib.dump({
    'model': final_model,
    'scaler_X': scaler_X_final,  # 保存X标准化器
    'scaler_y': scaler_y_final,  # 保存y标准化器
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test,
    'y_test': y_test,
    'feature_names': feature_names
}, './model/best_plsr_model_bayesian.pkl')
joblib.dump({'model': final_model,
             'scaler_X': scaler_X_final,
             'scaler_y': scaler_y_final,
             }, './model/plsr+bo.pkl')
print("\n最佳模型已保存为 './model/best_plsr_model_bayesian.pkl' (包含训练集、测试集和标准化器)")

# 保存优化过程数据
optimization_history = pd.DataFrame(optimizer.res)
optimization_history.to_csv('./result/plsr_bayesian_optimization_history.csv', index=False)
print("优化过程历史已保存为 './result/plsr_bayesian_optimization_history.csv'")

# 保存交叉验证结果
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv('./result/plsr_cross_validation_results.csv', index=False)
print("交叉验证结果已保存为 './result/plsr_cross_validation_results.csv'")

# 输出最佳交叉验证结果
if not cv_results_df.empty:
    best_cv_result = cv_results_df.loc[cv_results_df['avg_rmse'].idxmin()]
    print("\n最佳交叉验证结果:")
    print(f"平均RMSE: {best_cv_result['avg_rmse']:.4f}")
    print(f"平均MAE: {best_cv_result['avg_mae']:.4f}")
    print(f"平均R²: {best_cv_result['avg_r2']:.4f}")