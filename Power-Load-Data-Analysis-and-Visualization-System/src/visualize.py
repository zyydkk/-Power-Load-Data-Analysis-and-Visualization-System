import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(data_index.index,data_index["PJME_MW"])
plt.savefig("负荷随时间变化曲线(随时间变化).png")