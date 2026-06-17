# 1. 读取数据，查看数据信息

import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

import pandas as pd

data = pd.read_csv("D:/project/1/PJME_hourly.csv")
## 这一段代码的作用是读取CSV文件，并把它保存为一个Pandas DataFrame 类型的数据表, 变量名叫data。

print(data.head())
#print(data.head(10))
## data.head()的作用是查看变量data的前5行数据，head(10)是查看变量data的前10行数据；
## print（）是把结果输入到屏幕上。

#print(data.tail())
#print(data.tail(10))
## data.tail()是查看变量data的后5行数据，tail(10)是查看变量data的后10行数据。


print(data.shape)
## data.shape是查看数据有多少行，多少列：例如（100，2）意味着100行，2列。

# 2. 构造时间特征

data["Datetime"] = pd.to_datetime(data["Datetime"])
## 这段代码的作用：把data数据表中"Datetime"这一列:[str]   转换成真正的日期时间格式:datatime64[us].

data_index = data.set_index("Datetime").sort_index()
## 作用：把 data 数据表中的 ”time” 这一列设置为索引 index
## data = data.set_index("Datetime", drop=False)  drop=False:让 "time" 既作为索引，又保留在表格中。
## data = data.set_index["time"].sort_index()     .sort_index()按时间先后顺序排序。

data_index["hour"] = data_index.index.hour
## 这一段代码的作用是：从 data_index 的时间索引中提取 小时 的信息，并新增一列 hour 并保存这个结果。

data_index["dayofweek"] = data_index.index.dayofweek
## 从 data_index 的时间索引中提取“星期几”，并新增一列 "dayofweek" 保存结果。

data_index["month"] = data_index.index.month
## 从 data_index 的时间索引中提取“月份”，并新增一列 "month" 保存结果。

data_index["quarter"] = data_index.index.quarter
## 从 data_index 的时间索引中提取“季度”，并新增一列 "quarter" 保存结果。


data_index["year"]= data_index.index.year
## 从 data 数据表中的 Datetime 时间列中提取 年份， 然后新增一列 year 保存年份结果

data_index["dayofyear"]= data_index.index.dayofyear
## 从 data 数据表中的 Datetime 时间列中提取 “一年中的第几天”， 然后新增一列 dayofyear 保存年份结果

data_index["is_weekend"] = data_index["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
## 根据 dayofweek 这一列判断是否为周末，并增加一列 is_weekend
## .apply()  把一个函数引用到每一个数据上
## lambda x: 意味着临时定义一个小函数， 其中 x 表示当前正在处理的那个值

# 3. 加入历史负荷特征
data_index["lag_1"] = data_index["PJME_MW"].shift(1)
## 作用是：用上一时刻的 PJME_MW 值，生成一个新特征列 “lag_1”，在电力负荷预测中，这叫做滞后特征。
## .shift(1) 把这一列 整体向下移动1行

data_index["lag_24"] = data_index["PJME_MW"].shift(24)
data_index["lag_168"] = data_index["PJME_MW"].shift(168)
## 把过去的负荷值变成当前时刻的输入特征，让模型利用历史信息预测未来负荷.

# 4. 加入滑动平滑特征

data_index["rolling_mean_24"] = data_index["PJME_MW"].shift(1).rolling(window=24).mean()
## 作用是：计算 “PJME_MW” 这一列最近24个时间点的平均值，并生成一个新列 “rolling_mean_24”.
## .rolling(window=24) 设置滑动窗口大小为 24。
## .mean() 对窗口中的数计算平均值。
## .shift(1) 只使用当前时刻之前的数据来计算近 24 小时平均值。

data_index["rolling_mean_168"] = data_index["PJME_MW"].shift(1).rolling(window=168).mean()

data_index = data_index.dropna()
print(data_index.isna().sum())
print(data_index.shape)

data_index.head()

# 5. 按时间顺序划分训练集和测试集

feature = [
    "hour",
    "dayofweek",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168"
]

x = data_index[feature]
y = data_index["PJME_MW"]

split_index = int(len(data_index)*0.8)

