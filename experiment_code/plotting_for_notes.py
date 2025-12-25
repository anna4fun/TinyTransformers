import numpy as np
import matplotlib.pyplot as plt

## Momentum: Plot of 0.9^n
# 定义n的取值范围 (正整数)
n = np.arange(1, 101, 1)
# 计算0.9^n
y = 0.99 ** n

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制曲线和散点
ax.plot(n, y, color='#2E86AB', linewidth=2, label='$y = 0.9^n$')
ax.scatter(n, y, color='#A23B72', s=15, alpha=0.7)

# 添加水平线标记阈值
thresholds = [0.1, 0.05, 0.01]
colors = ['#F18F01', '#C73E1D', '#8B5A3C']
for threshold, color in zip(thresholds, colors):
    ax.axhline(y=threshold, color=color, linestyle='--', linewidth=1.5,
               label=f'Threshold = {threshold}')
    # 找到首次小于阈值的n值
    first_n = np.argmax(y < threshold) + 1  # +1 because n starts from 1
    ax.annotate(f'n={first_n}', xy=(first_n, threshold),
                xytext=(first_n+2, threshold*1.5),
                color=color, fontsize=10, fontweight='bold')

# 设置标题和标签（英文）
ax.set_title('Trend of $y = 0.99^n$ with Positive Integer $n$', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('$n$ (Positive Integer)', fontsize=12)
ax.set_ylabel('$0.99^n$', fontsize=12)

# 设置网格
ax.grid(True, linestyle=':', alpha=0.7)

# 设置坐标轴刻度格式
ax.xaxis.set_major_locator(plt.MultipleLocator(10))
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

# 添加图例
ax.legend(loc='upper right')

# 调整布局并保存
plt.tight_layout()
plt.savefig('./pictures/0_9_9_power_n_plot_english.png', dpi=300, bbox_inches='tight')
plt.close()

print("English version of the plot saved as: /pictures/0_9_power_n_plot_english.png")

# 输出关键节点数据（英文）
print("\nKey Threshold Data:")
for threshold in thresholds:
    first_n = np.argmax(y < threshold) + 1
    print(f"When n = {first_n}, 0.9^{first_n} = {0.9**first_n:.4f} < {threshold}")