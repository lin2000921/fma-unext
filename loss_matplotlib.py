import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 假设你已经有一个包含 'epoch', 'loss', 'val_loss' 的 CSV 文件
csv_path = r'D:\UNeXt-pytorch-main\models\new_unext_emcad\log.csv'  # 替换为你的 CSV 文件路径
df = pd.read_csv(csv_path)

# 创建绘图
plt.figure(figsize=(10, 6))

# 绘制训练损失 (train loss) 和验证损失 (val loss)
plt.plot(df['epoch'], df['loss'], label='Train Loss', color='blue', linestyle='-', marker='o', markersize=5)
plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='orange', linestyle='-', marker='x', markersize=5)

# 添加标题和轴标签
# plt.title('Training and Validation Loss over Epochs', fontsize=20)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)

# 设置 y 轴刻度及字体大小
y_ticks = np.arange(0, 2.25, 0.25)
plt.yticks(y_ticks, fontsize=16)

# 设置 x 轴刻度字体大小
plt.xticks(fontsize=16)


# 添加图例
plt.legend()

# 添加网格线
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()
