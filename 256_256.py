import os
from PIL import Image

# 输入和输出文件夹路径
input_folder = r'F:\TransUNet-main\predictions\TU_Synapse256\TU_pretrain_R50-ViT-B_16_skip3_epo300_bs16_256'  # 请替换为你的输入文件夹路径
output_folder = r'F:\TransUNet-main\predictions\TU_Synapse256\TU_pretrain_R50-ViT-B_16_skip3_epo300_bs16_256'  # 请替换为你的输出文件夹路径

# 创建输出文件夹（如果不存在的话）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    # 确保只处理图片文件
    if file_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # 打开图片
        with Image.open(file_path) as img:
            # 只处理512x512的图片
            if img.size == (512, 512):
                # 调整图片大小为256x256
                img_resized = img.resize((256, 256))

                # 保存调整后的图片到输出文件夹
                output_path = os.path.join(output_folder, filename)
                img_resized.save(output_path)
                print(f"图片 {filename} 已转换为 256x256 并保存到 {output_folder}")
            else:
                print(f"跳过图片 {filename}，因为它的尺寸不是512x512")
