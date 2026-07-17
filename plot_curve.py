import matplotlib.pyplot as plt

# 把你6组训练得到的15个准确率数字填到这里
linear4 = [53.05,59.93,68.51,69.02,70.61,73.21,73.01,72.91,74.74,75.46,75.82,77.73,77.22,77.15,78.23]
linear8 = [61.97,69.45,73.26,75.83,77.05,77.57,78.76,79.50,79.34,79.27,81.82,81.29,81.52,81.29,81.13]
linear16 = [65.82,71.67,70.23,73.76,74.06,74.14,75.92,76.28,77.49,77.48,78.62,78.99,79.57,79.49,79.36]
conv4 = [51.66,64.90,66.12,69.02,72.55,71.35,71.56,74.83,73.73,75.56,76.51,75.03,77.29,74.74,77.48]
conv8 = [62.25,69.20,73.99,74.02,75.70,76.21,78.44,78.95,79.40,79.50,79.60,79.39,81.41,80.34,80.37]
conv16 = [66.36,71.13,70.95,74.37,73.87,75.73,75.87,75.86,77.80,77.66,77.10,78.33,77.65,78.92,78.11]

# 绘制6条收敛曲线
plt.figure(figsize=(12, 6))
plt.plot(linear4, label="Linear-Patch4")
plt.plot(linear8, label="Linear-Patch8")
plt.plot(linear16, label="Linear-Patch16")
plt.plot(conv4, label="Conv-Patch4")
plt.plot(conv8, label="Conv-Patch8")
plt.plot(conv16, label="Conv-Patch16")

# 图表文字
plt.title("Validation Accuracy Convergence Curve of Different Patch Methods and Patch Sizes", fontsize=14)
plt.xlabel("Training Epoch", fontsize=12)
plt.ylabel("Validation Accuracy (%)", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

# 保存图片到文件夹
plt.savefig("acc_compare.png", dpi=150)
plt.show()
