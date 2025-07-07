import os
from PIL import Image


def convert_jpg_to_png(input_folder):
    """
    将文件夹内所有jpg图片转换为png格式，并直接覆盖原文件路径。

    :param input_folder: 输入文件夹路径，包含jpg文件
    """
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # 只处理jpg文件
        if filename.lower().endswith('.jpg'):
            try:
                # 打开jpg图片
                with Image.open(file_path) as img:
                    # 获取没有扩展名的文件名
                    base_name = os.path.splitext(filename)[0]
                    # 设置输出文件路径，直接覆盖原文件（改变扩展名为png）
                    output_path = os.path.join(input_folder, base_name + '.png')
                    # 保存为png格式
                    img.save(output_path, 'PNG')
                    print(f"成功将 {filename} 转换为 {base_name}.png")
                    # 删除原来的jpg文件
                    os.remove(file_path)
                    print(f"已删除原文件 {filename}")
            except Exception as e:
                print(f"无法处理文件 {filename}: {e}")


# 设置输入文件夹路径
input_folder = r'D:\UNeXt-pytorch-main\outputs\new_unext_emcad_ema\0'  # 请替换为包含jpg文件的文件夹路径

# 调用转换函数
convert_jpg_to_png(input_folder)
