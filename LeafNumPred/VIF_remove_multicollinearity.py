import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


data = pd.read_csv('C://Users/93622/Desktop/data.csv', encoding='utf-8-sig')
X = data.iloc[:, :7]
feature_names = list(X.columns)


t_scaler = StandardScaler()
X_scaled = pd.DataFrame(t_scaler.fit_transform(X), columns=feature_names)


def compute_vif(df):
    vif = pd.DataFrame({'feature': df.columns})
    vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif


iteration_records = []

def iterative_vif_elimination(df, threshold=10.0):
    df_current = df.copy()
    iteration = 1
    while True:
        vif_df = compute_vif(df_current)

        print(f"迭代 {iteration} 前特征 VIF 值:")
        print(vif_df)

        temp = vif_df.copy()
        temp['iteration'] = iteration
        iteration_records.append(temp)

        max_vif = vif_df['VIF'].max()
        if max_vif <= threshold:
            break
        drop_feat = vif_df.sort_values('VIF', ascending=False)['feature'].iloc[0]
        print(f"剔除特征: {drop_feat}, 最大 VIF={max_vif:.2f}\n")
        df_current = df_current.drop(columns=[drop_feat])
        iteration += 1
    return df_current, compute_vif(df_current)


X_reduced, final_vif = iterative_vif_elimination(X_scaled, threshold=10.0)


all_vif = pd.concat(iteration_records, ignore_index=True)
all_vif.to_csv('./result/vif_feature_remove.csv', index=False, encoding='utf-8-sig')
print("已保存每次迭代前的 VIF 值至 'vif_feature_remove.csv'\n")

print("最终保留特征列表:", list(X_reduced.columns))
print("剔除后 VIF:")
print(final_vif)
