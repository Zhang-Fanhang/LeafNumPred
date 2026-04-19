import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import ElasticNet
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

# 数据标准化 - 弹性网络对特征尺度敏感
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建全局变量存储交叉验证结果
cv_results = []


# 定义贝叶斯优化的目标函数（使用十折交叉验证）
def elasticnet_cv(alpha, l1_ratio):
    """弹性网络交叉验证函数，使用十折交叉验证"""
    try:
        # 转换参数类型
        alpha = max(alpha, 1e-5)  # 确保alpha大于0
        l1_ratio = max(min(l1_ratio, 1.0), 0.0)  # 确保在[0,1]范围内

        # 使用十折交叉验证
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_rmse_scores = []
        cv_mae_scores = []
        cv_r2_scores = []

        for train_index, val_index in kf.split(X_train_scaled):
            X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

            # 创建模型
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=10000,  # 增加迭代次数确保收敛
                random_state=42
            )

            # 训练模型
            model.fit(X_train_fold, y_train_fold)

            # 在验证集上预测
            y_pred = model.predict(X_val_fold)
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
            'params': {'alpha': alpha, 'l1_ratio': l1_ratio},
            'avg_rmse': avg_rmse,
            'avg_mae': avg_mae,
            'avg_r2': avg_r2,
            'cv_rmse_scores': cv_rmse_scores,
            'cv_mae_scores': cv_mae_scores,
            'cv_r2_scores': cv_r2_scores
        })

        # 打印当前参数组合的评估结果
        print(f"\n当前参数组合评估结果:")
        print(f"alpha: {alpha:.6f}, l1_ratio: {l1_ratio:.6f}")
        print(f"平均RMSE: {avg_rmse:.4f}, 平均MAE: {avg_mae:.4f}, 平均R²: {avg_r2:.4f}")

        # 返回负的平均RMSE（因为贝叶斯优化是最大化目标函数）
        return -avg_rmse

    except Exception as e:
        print(f"参数错误: alpha={alpha}, l1_ratio={l1_ratio}, 错误: {str(e)}")
        return -1e10  # 返回一个很大的负值，表示此参数组合无效


# 定义参数边界
pbounds = {
    'alpha': (0.0001, 1.0),  # 正则化强度
    'l1_ratio': (0.0, 1.0)  # L1正则化比例 (0=Ridge, 1=Lasso)
}

# 创建优化器
optimizer = BayesianOptimization(
    f=elasticnet_cv,
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
print("\n最佳参数组合:")
for key, value in best_params.items():
    print(f"{key}: {value}")

# 使用最佳参数训练最终模型
print("\n使用最佳参数训练最终模型...")
final_model = ElasticNet(
    alpha=best_params['alpha'],
    l1_ratio=best_params['l1_ratio'],
    max_iter=10000,
    random_state=42
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

# 特征重要性分析（使用系数的绝对值）
feature_importances = np.abs(final_model.coef_)
# 添加截距项作为参考
feature_importances = np.append(feature_importances, np.abs(final_model.intercept_))
feature_names_with_intercept = feature_names + ['Intercept']

importance_df = pd.DataFrame({
    '特征': feature_names_with_intercept,
    '重要性': feature_importances
}).sort_values('重要性', ascending=False)

print("\n特征重要性排序:")
print(importance_df)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(importance_df['特征'], importance_df['重要性'])
plt.xlabel('重要性分数 (|系数|)', fontsize=12)
plt.ylabel('特征', fontsize=12)
plt.title('弹性网络特征重要性', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('./img/elnr_feature_importance_bayesian.png', dpi=300)
plt.show()

# 绘制贝叶斯优化过程
# 提取目标函数值（负RMSE）并转换为实际RMSE
target_values = [-x for x in optimizer.space.target]  # 取负值得到实际RMSE

plt.figure(figsize=(12, 8))
plt.plot(range(1, len(target_values) + 1), target_values, 'o-', linewidth=2)
plt.xlabel('迭代次数', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('弹性网络贝叶斯优化过程 (RMSE)', fontsize=14)
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
plt.savefig('./img/elnr_bayesian_optimization_progress.png', dpi=300)
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
# plt.savefig('./img/elnr_actual_vs_predicted.png', dpi=300)
# plt.show()

# 保存模型系数
coef_df = pd.DataFrame({
    '特征': feature_names_with_intercept,
    '系数': np.append(final_model.coef_, final_model.intercept_)
})
coef_df.to_csv('./result/elnr_coefficients_bayesian.csv', index=False, encoding='utf-8-sig')
print("\n模型系数已保存为 './result/elnr_coefficients_bayesian.csv'")

# 保存最佳参数
best_params_df = pd.DataFrame([best_params])
best_params_df.to_csv('./result/elnr_best_params_bayesian.csv', index=False, encoding='utf-8-sig')
print("最佳参数已保存为 './result/elnr_best_params_bayesian.csv'")

# 保存最佳模型、训练集和测试集
joblib.dump({
    'model': final_model,
    'scaler': scaler,  # 保存标准化器
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test,
    'y_test': y_test,
    'feature_names': feature_names
}, './model/best_elnr_model_bayesian.pkl')
joblib.dump(final_model, './model/elnr+bo.pkl')
print("\n最佳模型已保存为 './model/best_elnr_model_bayesian.pkl' (包含训练集、测试集和标准化器)")

# 保存优化过程数据
optimization_history = pd.DataFrame(optimizer.res)
optimization_history.to_csv('./result/elnr_bayesian_optimization_history.csv', index=False)
print("优化过程历史已保存为 './result/elnr_bayesian_optimization_history.csv'")

# 保存交叉验证结果
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv('./result/elnr_cross_validation_results.csv', index=False)
print("交叉验证结果已保存为 './result/elnr_cross_validation_results.csv'")

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

    # 使用LinearExplainer替代KernelExplainer，因为弹性网络是线性模型
    # 这样计算更快更准确
    explainer = shap.LinearExplainer(final_model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)

    # 特征重要性总结图
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_test_scaled,
        feature_names=feature_names,
        show=False
    )
    plt.title('SHAP特征重要性', fontsize=14)
    plt.tight_layout()
    plt.savefig('./img/elnr_shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 单个预测解释（使用第一个测试样本）
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test_scaled[0],
        feature_names=feature_names
    ), show=False, max_display=10)
    plt.title('单个样本的SHAP解释', fontsize=14)
    plt.tight_layout()
    plt.savefig('./img/elnr_shap_waterfall.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("SHAP解释已保存为图片")

except ImportError:
    print("\n未安装SHAP库，跳过SHAP分析")
except Exception as e:
    print(f"\n生成SHAP解释时出错: {str(e)}")

