import joblib
import numpy as np
import pandas as pd

data = pd.read_csv('./dataset/dataset_2025_4.csv', encoding='utf-8-sig')


saved_data = joblib.load('./model/best_rfr_model_bayesian.pkl')


model = saved_data['model']
feature_names = saved_data['feature_names']


num_df = data[feature_names]


leaf_num = model.predict(num_df)
print(leaf_num)

leaf_num_df = pd.DataFrame(np.array(leaf_num), columns=['Estimated leaf number'])
leaf_num_df.to_csv('./result/rfr_estimated_leaf_number_2025_4.csv', index=False)

print("预测完成，结果已保存为 './result/rfr_estimated_leaf_number.csv'")