import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR
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

# 创建全局变量存储交叉验证结果
cv_results = []


# 定义贝叶斯优化的目标函数（使用10折交叉验证）
def svr_cv(log_C, log_gamma, log_epsilon):
    """SVR交叉验证函数，使用10折交叉验证"""
    # 转换参数类型（指数变换处理对数尺度参数）
    C = 10 ** log_C
    gamma = 10 ** log_gamma
    epsilon = 10 ** log_epsilon  # 使用指数变换后的epsilon

    # 使用10折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_rmse_scores = []
    cv_mae_scores = []
    cv_r2_scores = []

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # 标准化数据（每折独立标准化）
        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)

        # 创建SVR模型
        model = SVR(
            C=C,
            gamma=gamma,
            epsilon=epsilon,
            kernel='rbf'  # 核函数：linear(线性关系), rbf(非线性关系), sigmoid(周期性数据), poly(已知多项式关系)
        )

        # 训练模型
        model.fit(X_train_fold_scaled, y_train_fold)

        # 在验证集上预测
        y_pred = model.predict(X_val_fold_scaled)
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
        'params': {'C': C, 'gamma': gamma, 'epsilon': epsilon},
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'avg_r2': avg_r2,
        'cv_rmse_scores': cv_rmse_scores,
        'cv_mae_scores': cv_mae_scores,
        'cv_r2_scores': cv_r2_scores
    })

    # 打印当前参数组合的评估结果
    print(f"\n当前参数组合评估结果 (C={C:.4f}, gamma={gamma:.4f}, epsilon={epsilon:.4f}):")
    print(f"平均RMSE: {avg_rmse:.4f}, 平均MAE: {avg_mae:.4f}, 平均R²: {avg_r2:.4f}")

    # 返回平均RMSE的负值，返回负值因为贝叶斯优化是最大化目标函数
    return -avg_rmse


# 定义参数边界（使用对数空间）
pbounds = {
    'log_C': (-1, 3),  # 对应C: 0.1 到 1000
    'log_gamma': (-4, 1),  # 对应gamma: 0.0001 到 10
    'log_epsilon': (-3, 0)  # 对应epsilon: 0.001 到 1
}

# 创建优化器
optimizer = BayesianOptimization(
    f=svr_cv,
    pbounds=pbounds,
    random_state=42,
)

# 执行贝叶斯优化（记录时间）
print("开始贝叶斯优化...")
start_time = time.time()
optimizer.maximize(
    init_points=10,  # 初始随机点
    n_iter=100,  # 贝叶斯优化迭代次数（SVR训练较慢，减少迭代次数）
)
end_time = time.time()
print(f"贝叶斯优化耗时: {end_time - start_time:.2f} 秒")

# 获取最佳参数
best_params = optimizer.max['params']
print("\n最佳参数组合 (对数形式):")
for key, value in best_params.items():
    print(f"{key}: {value:.4f}")

# 转换最佳参数为实际值
actual_params = {
    'C': 10 ** best_params['log_C'],
    'gamma': 10 ** best_params['log_gamma'],
    'epsilon': 10 ** best_params['log_epsilon']
}
print("\n实际最佳参数:")
for key, value in actual_params.items():
    print(f"{key}: {value:.6f}")

# 使用最佳参数训练最终模型（在完整训练集上）
print("\n使用最佳参数训练最终模型...")

# 标准化训练集和测试集
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建并训练SVR模型
final_model = SVR(
    C=actual_params['C'],
    gamma=actual_params['gamma'],
    epsilon=actual_params['epsilon'],
    kernel='rbf'
)
final_model.fit(X_train_scaled, y_train)

# 在测试集上评估
y_pred = final_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n测试集评估结果:")
print(f"均方误差(MSE): {mse:.4f}")
print(f"均方根误差(RMSE): {rmse:.4f}")
print(f"平均绝对误差(MAE): {mae:.4f}")
print(f"决定系数(R²): {r2:.4f}")

# 绘制贝叶斯优化过程 - 修改为显示实际RMSE值
# 提取目标函数值（负RMSE）并转换为实际RMSE
target_values = [-x for x in optimizer.space.target]  # 取负值得到实际RMSE

plt.figure(figsize=(12, 8))
plt.plot(range(1, len(target_values) + 1), target_values, 'o-')
plt.xlabel('迭代次数')
plt.ylabel('RMSE')
plt.title('SVR贝叶斯优化过程 (RMSE)')
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
plt.savefig('./img/svr_bayesian_optimization_progress.png', dpi=300)
plt.show()

# # 特征重要性分析（SVR没有内置特征重要性，使用排列重要性）
# try:
#     from sklearn.inspection import permutation_importance
#
#     print("\n计算排列重要性...")
#     start_time = time.time()
#
#     # 计算排列重要性
#     result = permutation_importance(
#         final_model,
#         X_test_scaled,
#         y_test,
#         n_repeats=10,
#         random_state=42
#     )
#
#     # 获取重要性分数
#     importances = result.importances_mean
#     importance_df = pd.DataFrame({
#         'Feature': feature_names,
#         'Importance': importances
#     }).sort_values('Importance', ascending=False)
#
#     print(f"排列重要性计算耗时: {time.time() - start_time:.2f} 秒")
#     print("\n特征重要性排序 (排列重要性):")
#     print(importance_df)
#
#     # 可视化特征重要性
#     plt.figure(figsize=(10, 6))
#     plt.barh(importance_df['Feature'], importance_df['Importance'])
#     plt.xlabel('重要性分数')
#     plt.title('SVR特征重要性 (排列重要性)')
#     plt.gca().invert_yaxis()
#     plt.tight_layout()
#     plt.savefig('./img/svr_feature_importance_bayesian.png', dpi=300)
#     plt.show()
#
# except ImportError:
#     print("\nscikit-learn版本过低，无法计算排列重要性")

# 保存最佳模型、训练集和测试集
joblib.dump({
    'model': final_model,
    'scaler': scaler,
    'params': actual_params,
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test,
    'y_test': y_test,
    'feature_names': feature_names
}, './model/best_svr_model_bayesian.pkl')
joblib.dump({
    'model': final_model,
    'scaler': scaler,
    'params': actual_params,
}, './model/svr+bo.pkl')
print("\n最佳模型已保存为 './model/best_svr_model_bayesian.pkl' (包含训练集和测试集)")

# 保存优化过程数据
optimization_history = pd.DataFrame(optimizer.res)
optimization_history.to_csv('./result/svr_bayesian_optimization_history.csv', index=False)
print("优化过程历史已保存为 './result/svr_bayesian_optimization_history.csv'")

# 保存交叉验证结果
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv('./result/svr_cross_validation_results.csv', index=False)
print("交叉验证结果已保存为 './result/svr_cross_validation_results.csv'")

# 输出最佳交叉验证结果
best_cv_result = cv_results_df.loc[cv_results_df['avg_rmse'].idxmin()]
print("\n最佳交叉验证结果:")
print(f"平均RMSE: {best_cv_result['avg_rmse']:.4f}")
print(f"平均MAE: {best_cv_result['avg_mae']:.4f}")
print(f"平均R²: {best_cv_result['avg_r2']:.4f}")

# 模型解释：SHAP值（SVR兼容性有限）
try:
    import shap

    print("\n生成SHAP解释...")

    # 使用全部训练集作为背景样本（并用于解释）
    print(f"使用全部训练集作为背景样本 ({X_train_scaled.shape[0]}个样本)")
    explainer = shap.KernelExplainer(
        final_model.predict,
        shap.kmeans(X_train_scaled, 100)
    )

    # 计算SHAP值：对训练集进行解释
    shap_values = explainer.shap_values(X_train_scaled)

    # 特征重要性总结图（基于训练集）
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_train_scaled,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig('./img/svr_shap_summary_bayesian.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 单个预测解释图（使用训练集第一个样本）
    plt.figure()
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_train_scaled[0],
        feature_names=feature_names
    ), show=False)
    plt.tight_layout()
    plt.savefig('./img/svr_shap_waterfall_bayesian.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ✅ 新增：特征重要性条形图（按平均绝对SHAP值排序）
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df_shap = pd.DataFrame({
        'Feature': feature_names,
        'MeanAbsSHAP': mean_abs_shap
    }).sort_values(by='MeanAbsSHAP', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df_shap['Feature'], importance_df_shap['MeanAbsSHAP'])
    plt.xlabel('平均|SHAP值|')
    plt.title('特征重要性（基于SHAP）')
    plt.tight_layout()
    plt.savefig('./img/svr_shap_bar_importance.png', dpi=300)
    plt.show()

    # （可选）保存数值结果
    importance_df_shap.to_csv('./result/svr_shap_feature_importance.csv', index=False)

    # 生成 SHAP 值热力图
    # 重新定义 SHAP Explainer
    explainer = shap.Explainer(final_model.predict, X_train_scaled, feature_names=feature_names)

    # 计算 SHAP 值（建议使用所有训练样本）
    shap_values = explainer(X_train_scaled)

    # 显示聚类后的 SHAP 热力图
    plt.figure(figsize=(12, 10))
    shap.plots.heatmap(shap_values, max_display=len(feature_names))
    plt.tight_layout()
    plt.savefig('./img/svr_shap_clustered_heatmap.png', dpi=300)

    shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
    shap_df.to_csv('./result/svr_shap_clustered_values.csv', index=False)

    print("SHAP解释已保存为图片")

except ImportError:
    print("\n未安装SHAP库，跳过SHAP分析")
except Exception as e:
    print(f"\n生成SHAP解释时出错: {str(e)}")

