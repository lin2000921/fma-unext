import os
import random
import shutil

# 设置图片文件夹路径和目标文件夹路径
source_folder = r'C:\Users\sb\Desktop\k_val\images'  # 图片文件夹路径
train_folder = r'C:\Users\sb\Desktop\k_val\test'  # 训练集文件夹路径
test_folder = r'C:\Users\sb\Desktop\k_val\train'  # 测试集文件夹路径
num_splits = 10  # 划分的份数
num_rounds = 2  # 重复操作的轮数

# 创建训练集和测试集的主文件夹
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# 获取所有图片文件
image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]


# 定义函数：随机划分文件夹
def split_images_randomly(image_files):
    # 将图片列表随机打乱
    random.shuffle(image_files)

    # 将图片划分为 num_splits 等份
    split_size = len(image_files) // num_splits
    splits = [image_files[i:i + split_size] for i in range(0, len(image_files), split_size)]

    # 如果最后一份较小，合并到上一份
    if len(splits) > num_splits:
        splits[-2].extend(splits[-1])
        splits = splits[:-1]

    return splits


# 执行多轮划分
for round_num in range(1, num_rounds + 1):
    print(f"Round {round_num} started...")

    # 随机划分图片
    splits = split_images_randomly(image_files)

    # 随机选择1份作为测试集
    test_split = random.choice(splits)

    # 将其余部分作为训练集
    train_splits = [split for split in splits if split != test_split]
    train_split = [item for sublist in train_splits for item in sublist]

    # 创建子文件夹来存放每一轮的训练集和测试集
    round_train_folder = os.path.join(train_folder, f'round_{round_num}')
    round_test_folder = os.path.join(test_folder, f'round_{round_num}')

    # 创建文件夹
    os.makedirs(round_train_folder, exist_ok=True)
    os.makedirs(round_test_folder, exist_ok=True)

    # 移动文件到训练集文件夹
    for image in train_split:
        shutil.copy(os.path.join(source_folder, image), round_train_folder)

    # 移动文件到测试集文件夹
    for image in test_split:
        shutil.copy(os.path.join(source_folder, image), round_test_folder)

    print(f"Round {round_num} completed. {len(train_split)} images in train, {len(test_split)} images in test.")

print("All rounds completed.")
