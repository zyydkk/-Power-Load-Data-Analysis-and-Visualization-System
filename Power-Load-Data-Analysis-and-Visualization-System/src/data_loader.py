import pandas as pd

data = pd.read_csv("PJME_hourly.csv")
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

#data.info()
## data.info 查看变量data这个数据表的整体信息

#data.info(verbose=False)
## verbose表示“是否显示详细信息”

data.info(show_counts=True)
## show_counts 显示每一列的非空数量

#data.isnull().sum()
## data.isnull().sum() 查看每一列具体缺失多少值

print(data.columns)
## data.columns的作用是查看data的列名。

data.describe()
## data.describe()查看数据集中"数值型"列的统计信息:
## mean 平均值；std 标准差；25%下四分位数

data["PJME_MW"].describe()
## 查看【“PJME_MW“】这一数据型列的统计信息

data.describe(include="all") 
## 查看字符串列、类别列的信息. top:出现次数最多的值  freq:出现次数最多的值对应的次数