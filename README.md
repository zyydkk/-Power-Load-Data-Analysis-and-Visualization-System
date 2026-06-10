# -Power-Load-Data-Analysis-and-Visualization-System
学习AI的第一个项目-2026.6.7
# 基于机器学习的电力负荷预测项目

## 1. 项目简介

本项目基于公开电力负荷数据，完成数据清洗、特征工程、模型训练、结果评测和可视化分析，实现短期电力负荷预测

## 2. 项目结构

```text
-Power-Load-Data-Analysis-and-Visualization-System
├── data/
├── notebooks/
├── src/
├── models/
├── results/
├── docs/
├── README.md
├── requirements.txt
└── main.py
```
## 3. 项目进展

2026.6.8-
         
         下载公开数据集Hourly Energy Consumption，使用其中的PJME_hourly.csv作为项目使用的第一个数据集。

         使用jupyter notebook 新建第一个data_loader数据加载脚本。

2026.6.9-
         
         使用jupyter notebook 新建preprocessing数据处理脚本。

         从 data 的数据表中的 Datetime 时间列中提取 年、月、小时、星期几，并新增 year、month、hour、dayofweek 等这些列。
         
         将 Datetime 时间列按照时间顺序进行排列。

         使用jupyter notebook 新建visualize可视化脚本。

         画出负荷时间变化曲线(随年份变化）。

2026.6.10-

          鉴于代码能力较弱，改变对每一个部分进行单独编码的计划，将data_loader、preprocessing、visualize三个jupyter notebook脚本
          合成为一个脚本命名为：01_data_analysis.以此脚本作为本项目的第一个可运行程序.

          在负荷时间变化曲线(随年份变化）这个图的基础上添加X轴名、Y轴名、图片名以及网格线.

          导入 train_test_split 函数，以hour,dayofweek,month作为数据集的输入特征，PJME_MW 作为数据集的标签.

          将数据集以 test_size=0.2 的方式划分 训练集和测试集.
          
         

