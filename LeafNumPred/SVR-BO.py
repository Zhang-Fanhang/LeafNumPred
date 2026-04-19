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


warnings.filterwarnings('ignore')


data_test = pd.read_csv('./dataset/data_testset.csv', encoding='utf-8-sig')
data_train = pd.read_csv('./dataset/data_trainset.csv', encoding='utf-8-sig')

X_train = data_train.iloc[:, :5].values
y_train = data_train.iloc[:, 5].values

X_test = data_test.iloc[:, :5].values
y_test = data_test.iloc[:, 5].values

feature_names = list(data_train.columns[:5])


cv_results = []



def svr_cv(log_C, log_gamma, log_epsilon):

    C = 10 ** log_C
    gamma = 10 ** log_gamma
    epsilon = 10 ** log_epsilon


    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_rmse_scores = []
    cv_mae_scores = []
    cv_r2_scores = []

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]


        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)


        model = SVR(
            C=C,
            gamma=gamma,
            epsilon=epsilon,
            kernel='rbf'
        )


        model.fit(X_train_fold_scaled, y_train_fold)


        y_pred = model.predict(X_val_fold_scaled)

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
        'params': {'C': C, 'gamma': gamma, 'epsilon': epsilon},
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'avg_r2': avg_r2,
        'cv_rmse_scores': cv_rmse_scores,
        'cv_mae_scores': cv_mae_scores,
        'cv_r2_scores': cv_r2_scores
    })


    print(f"\n当前参数组合评估结果 (C={C:.4f}, gamma={gamma:.4f}, epsilon={epsilon:.4f}):")
    print(f"平均RMSE: {avg_rmse:.4f}, 平均MAE: {avg_mae:.4f}, 平均R²: {avg_r2:.4f}")


    return -avg_rmse



pbounds = {
    'log_C': (-1, 3),
    'log_gamma': (-4, 1),
    'log_epsilon': (-3, 0)
}


optimizer = BayesianOptimization(
    f=svr_cv,
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
print("\n最佳参数组合 (对数形式):")
for key, value in best_params.items():
    print(f"{key}: {value:.4f}")


actual_params = {
    'C': 10 ** best_params['log_C'],
    'gamma': 10 ** best_params['log_gamma'],
    'epsilon': 10 ** best_params['log_epsilon']
}
print("\n实际最佳参数:")
for key, value in actual_params.items():
    print(f"{key}: {value:.6f}")


print("\n使用最佳参数训练最终模型...")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


final_model = SVR(
    C=actual_params['C'],
    gamma=actual_params['gamma'],
    epsilon=actual_params['epsilon'],
    kernel='rbf'
)
final_model.fit(X_train_scaled, y_train)


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


target_values = [-x for x in optimizer.space.target]

plt.figure(figsize=(12, 8))
plt.plot(range(1, len(target_values) + 1), target_values, 'o-')
plt.xlabel('迭代次数')
plt.ylabel('RMSE')
plt.title('SVR贝叶斯优化过程 (RMSE)')
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
plt.savefig('./img/svr_bayesian_optimization_progress.png', dpi=300)
plt.show()


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


optimization_history = pd.DataFrame(optimizer.res)
optimization_history.to_csv('./result/svr_bayesian_optimization_history.csv', index=False)
print("优化过程历史已保存为 './result/svr_bayesian_optimization_history.csv'")


cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv('./result/svr_cross_validation_results.csv', index=False)
print("交叉验证结果已保存为 './result/svr_cross_validation_results.csv'")


best_cv_result = cv_results_df.loc[cv_results_df['avg_rmse'].idxmin()]
print("\n最佳交叉验证结果:")
print(f"平均RMSE: {best_cv_result['avg_rmse']:.4f}")
print(f"平均MAE: {best_cv_result['avg_mae']:.4f}")
print(f"平均R²: {best_cv_result['avg_r2']:.4f}")


try:
    import shap

    print("\n生成SHAP解释...")


    print(f"使用全部训练集作为背景样本 ({X_train_scaled.shape[0]}个样本)")
    explainer = shap.KernelExplainer(
        final_model.predict,
        shap.kmeans(X_train_scaled, 100)
    )


    shap_values = explainer.shap_values(X_train_scaled)


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


    importance_df_shap.to_csv('./result/svr_shap_feature_importance.csv', index=False)


    explainer = shap.Explainer(final_model.predict, X_train_scaled, feature_names=feature_names)


    shap_values = explainer(X_train_scaled)


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

