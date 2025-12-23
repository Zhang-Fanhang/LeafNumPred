import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

# 读取数据
file = 'C://Users/93622/Desktop/data.csv'
df = pd.read_csv(file, encoding='utf-8-sig')

# 设置特征列
feature_cols = [
    'East-West crown diameter',
    'South-North crown diameter',
    'Tree height',
    'Tree crown height',
    'Tree crown volume',
    'Tree crown surface area',
    'Crown projection area'
]
num_df = df[feature_cols]

# 计算相关系数矩阵，保留两位小数
corr_matrix = num_df.corr(method='pearson').round(2)

# 左下角：散点图 + 回归线 + 相关性标注
def lower_plot(x, y, **kwargs):
    ax = plt.gca()
    sns.scatterplot(x=x, y=y, ax=ax, s=15, **kwargs)
    sns.regplot(x=x, y=y, ax=ax, scatter=False, color='red', line_kws={"linewidth": 1.2})

    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    if xlabel in corr_matrix.columns and ylabel in corr_matrix.index:
        r = corr_matrix.loc[ylabel, xlabel]
        ax.annotate(f"Cor = {r:.2f}", xy=(0.05, 0.85), xycoords='axes fraction',
                    fontsize=9, color='black')

# 创建 PairGrid
g = sns.PairGrid(num_df)

# 使用第二段代码中的简洁写法替换对角线绘图
g.map_diag(sns.histplot, kde=True)

# 左下角保持自定义绘制
g.map_lower(lower_plot)

# 隐藏右上角子图
for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    g.axes[i, j].set_visible(False)

plt.tight_layout()
plt.show()

# 输出 Pearson 相关系数矩阵
print("Pearson 相关系数矩阵：")
print(corr_matrix)
