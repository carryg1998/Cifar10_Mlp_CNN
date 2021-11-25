r"""
训练完成后平均F1值会存在result文件夹下的表格里，
运行task2_draw可以画出折线图(储存结果数据可以自己改，也可以画出来)
不知道为什么降维以后效果会很差，可能本来就这样吧、、、
"""

import pandas as pd
import matplotlib.pyplot as plt

sheet = "features"

df = pd.read_excel("./result/result.xlsx", names=None, sheet_name=sheet)

df = df.sort_values("ratio")
plt.plot(df['ratio'], df["mean F1"])
plt.show()
