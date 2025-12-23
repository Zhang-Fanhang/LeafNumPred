import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import time
from bayes_opt import BayesianOptimization
import warnings
import matplotlib
import joblib

matplotlib.rc("font", family='YouYuan')

warnings.filterwarnings('ignore')

data = pd.read_csv('./dataset/data.csv', encoding='utf-8-sig')
X = data.iloc[:, :5].values
y = data.iloc[:, 5].values
feature_names = list(data.columns[:5])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cv_results = []

def rf_cv(n_estimators, max_features, min_samples_leaf, max_depth, min_samples_split):
    params = {
        'n_estimators': int(n_estimators),
        'max_features': min(max(max_features, 0.1), 1.0),
        'min_samples_leaf': int(min_samples_leaf),
        'max_depth': int(max_depth) if max_depth > 1 else None,
        'min_samples_split': int(min_samples_split),
        'n_jobs': -1,
        'random_state': 42
    }

    if params['max_depth'] == 0:
        params['max_depth'] = None

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_rmse_scores = []
    cv_mae_scores = []
    cv_r2_scores = []

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        model = RandomForestRegressor(**params)
        model.fit(X_train_fold, y_train_fold)

        y_pred = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        mae = mean_absolute_error(y_val_fold, y_pred)
        r2 = r2_score(y_val_fold, y_pred)

        cv_rmse_scores.append(rmse)
        cv_mae_scores.append(mae)
        cv_r2_scores.append(r2)

    avg_rmse = np.mean(cv_rmse_scores)
    avg_mae = np.mean(cv_mae_scores)
    avg_r2 = np.mean(cv_r2_scores)

    cv_results.append({
        'params': params.copy(),
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'avg_r2': avg_r2,
        'cv_rmse_scores': cv_rmse_scores,
        'cv_mae_scores': cv_mae_scores,
        'cv_r2_scores': cv_r2_scores
    })

    print(f"\n当前参数组合评估结果:")
    print(f"平均RMSE: {avg_rmse:.4f}, 平均MAE: {avg_mae:.4f}, 平均R²: {avg_r2:.4f}")

    return -avg_rmse

pbounds = {
    'n_estimators': (10, 1000),
    'max_features': (0.1, 1.0),
    'min_samples_leaf': (1, 20),
    'max_depth': (5, 50),
    'min_samples_split': (2, 20)
}

optimizer = BayesianOptimization(
    f=rf_cv,
    pbounds=pbounds,
    random_state=42,
)

print("开始贝叶斯优化...")
start_time = time.time()
optimizer.maximize(
    init_points=10,
    n_iter=100,
)
end_time = time.time()
print(f"贝叶斯优化耗时: {end_time - start_time:.2f} 秒")

best_params = optimizer.max['params']
print("\n最佳参数组合:")
for key, value in best_params.items():
    print(f"{key}: {value}")

best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
best_params['min_samples_split'] = int(best_params['min_samples_split'])
best_params['max_depth'] = int(best_params['max_depth']) if best_params['max_depth'] > 1 else None

print("\n使用最佳参数训练最终模型...")
final_model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_features=best_params['max_features'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    n_jobs=-1,
    random_state=42
)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n测试集评估结果:")
print(f"均方误差(MSE): {mse:.4f}")
print(f"均方根误差(RMSE): {rmse:.4f}")
print(f"平均绝对误差(MAE): {mae:.4f}")
print(f"决定系数(R²): {r2:.4f}")

target_values = [-x for x in optimizer.space.target]

plt.figure(figsize=(12, 8))
plt.plot(range(1, len(target_values) + 1), target_values, 'o-')
plt.xlabel('迭代次数')
plt.ylabel('RMSE')
plt.title('RFR贝叶斯优化过程 (RMSE)')
plt.grid(True)

best_iteration = np.argmin(target_values)
best_rmse = target_values[best_iteration]
plt.scatter(best_iteration + 1, best_rmse, color='red', s=100, zorder=5)
plt.annotate(f'Optimal value: {best_rmse:.4f}',
             (best_iteration + 1, best_rmse),
             textcoords="offset points",
             xytext=(0, 10),
             ha='center',
             fontsize=12)

plt.tight_layout()
plt.savefig('./img/rfr_bayesian_optimization_progress.png', dpi=300)
plt.show()

joblib.dump({
    'model': final_model,
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test,
    'y_test': y_test,
    'feature_names': feature_names
}, './model/best_rfr_model_bayesian.pkl')
joblib.dump(final_model, './model/rfr+bo.pkl')
print("\n最佳模型已保存为 './model/best_rfr_model_bayesian.pkl' (包含训练集和测试集)")

optimization_history = pd.DataFrame(optimizer.res)
optimization_history.to_csv('./result/rfr_bayesian_optimization_history.csv', index=False)
print("优化过程历史已保存为 './result/rfr_bayesian_optimization_history.csv'")

cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv('./result/rfr_cross_validation_results.csv', index=False)
print("交叉验证结果已保存为 './result/rfr_cross_validation_results.csv'")

best_cv_result = cv_results_df.loc[cv_results_df['avg_rmse'].idxmin()]
print("\n最佳交叉验证结果:")
print(f"平均RMSE: {best_cv_result['avg_rmse']:.4f}")
print(f"平均MAE: {best_cv_result['avg_mae']:.4f}")
print(f"平均R²: {best_cv_result['avg_r2']:.4f}")

try:
    import shap

    print("\n生成SHAP解释...")

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_train)

    plt.figure()
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('./img/rfr_shap_summary_bayesian.png', dpi=300, bbox_inches='tight')
    plt.show()

    sample_idx = 0
    plt.figure()
    shap.plots.waterfall(shap.Explanation(values=shap_values[sample_idx],
                                          base_values=explainer.expected_value[0],
                                          data=X_train[sample_idx],
                                          feature_names=feature_names), show=False)
    plt.tight_layout()
    plt.savefig('./img/rfr_shap_waterfall_bayesian.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("SHAP解释已保存为图片")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'MeanAbsSHAP': mean_abs_shap
    }).sort_values(by='MeanAbsSHAP', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(shap_importance_df['Feature'], shap_importance_df['MeanAbsSHAP'])
    plt.xlabel('平均|SHAP值|')
    plt.title('特征重要性（基于SHAP）')
    plt.tight_layout()
    plt.savefig('./img/rfr_shap_bar_importance.png', dpi=300)
    plt.show()
    shap_importance_df.to_csv('./result/rfr_shap_feature_importance.csv', index=False)

    explainer = shap.Explainer(final_model, X_train, feature_names=feature_names)

    shap_values = explainer(X_train)

    plt.figure(figsize=(12, 10))
    shap.plots.heatmap(shap_values, max_display=len(feature_names))
    plt.tight_layout()
    plt.savefig('./img/rfr_shap_clustered_heatmap.png', dpi=300)
    shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
    shap_df.to_csv('./result/rfr_shap_clustered_values.csv', index=False)

    print("SHAP解释已保存为图片")

except ImportError:
    print("\n未安装SHAP库，跳过SHAP分析")