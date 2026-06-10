import pandas as pd

def preprocess_data(data_path):

    data = pd.read_csv("PJME_hourly.csv")

    data.columns

    print(data)
    data["Datetime"] = pd.to_datetime(data["Datetime"])
    ## 这段代码的作用：把data数据表中"Datetime"这一列:[str]   转换成真正的日期时间格式:datatime64[us].

    # print(data["time"])
    # print(data["Datetime"].dtype)
    # print(data["time"].dtype)
    data["year"] = data["Datetime"].dt.year
    ## 从 data 数据表中的 Datetime 时间列中提取 年份， 然后新增一列 year 保存年份结果

    data.head()

    ## 这一段代码：把普通字符串时间转换为真正的时间类型，为后续按时间分析数据做准备。
    print(data.columns)


    data_index = data.set_index("Datetime").sort_index()
    ## 作用：把 data 数据表中的 ”time” 这一列设置为索引 index
    ## data = data.set_index("Datetime", drop=False)  drop=False:让 "time" 既作为索引，又保留在表格中。
    ## data = data.set_index["time"].sort_index()     .sort_index()按时间先后顺序排序。


    data_index.head()
    #print(data_index.index)
    ## 把普通的时间列变成 DataFrame 的时间索引，为后续按时间处理数据做准备。


    data_index["hour"] = data_index.index.hour
    ## 这一段代码的作用是：从 data_index 的时间索引中提取 小时 的信息，并新增一列 hour 并保存这个结果。

    data_index["dayofweek"] = data_index.index.dayofweek
    ## 从 data_index 的时间索引中提取“星期几”，并新增一列 "dayofweek" 保存结果。
    data_index["month"] = data_index.index.month

    print(data_index.head())

    return data_index