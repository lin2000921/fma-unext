import os
import numpy as np
from PIL import Image


def calculate_pixel_accuracy(pred_mask, true_mask):
    """
    计算单张图片的像素准确度 (Accuracy)

    :param pred_mask: 预测的掩码（0/1或0-255值）
    :param true_mask: 真实的掩码（0/1或0-255值）
    :return: 像素准确度
    """
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # 二值化
    true_mask = (true_mask > 0.5).astype(np.uint8)  # 二值化

    accuracy = np.mean(pred_mask == true_mask)  # 计算像素级准确度
    return accuracy


def calculate_mean_accuracy(pred_masks, true_masks):
    """
    计算所有图片的平均像素准确度

    :param pred_masks: 预测掩码的列表
    :param true_masks: 真实掩码的列表
    :return: 平均像素准确度
    """
    assert len(pred_masks) == len(true_masks), "预测掩码和真实掩码的数量必须相同"

    total_accuracy = 0.0
    num_masks = len(pred_masks)

    for i in range(num_masks):
        pred_mask = np.array(pred_masks[i])
        true_mask = np.array(true_masks[i])

        total_accuracy += calculate_pixel_accuracy(pred_mask, true_mask)

    mean_accuracy = total_accuracy / num_masks
    return mean_accuracy


def read_images_from_folder(folder_path, file_extension=".png"):
    """
    从文件夹中读取所有图片，返回图像列表

    :param folder_path: 文件夹路径
    :param file_extension: 读取的文件类型
    :return: 图片列表
    """
    image_list = []
    for filename in sorted(os.listdir(folder_path)):  # 对文件进行排序
        if filename.lower().endswith(file_extension):
            img_path = os.path.join(folder_path, filename)
            image = Image.open(img_path).convert("L")  # 转为灰度图
            image_list.append(np.array(image) / 255.0)  # 归一化到 [0, 1]
    return image_list


def main():
    # 指定真实标签文件夹路径
    true_labels_folder = r"D:\UNeXt-pytorch-main\inputs\new_wusun\mask\test\1"
    # 指定推理结果文件夹路径
    predicted_results_folder = r"D:\UNeXt-pytorch-main\outputs\new_unext_emcad_ema\0"

    # 读取真实标签和预测结果
    true_labels = read_images_from_folder(true_labels_folder)
    predicted_results = read_images_from_folder(predicted_results_folder)

    # 计算所有图片的平均像素准确度
    mean_acc = calculate_mean_accuracy(predicted_results, true_labels)

    print(f"平均像素准确度：{mean_acc:.4f}")


if __name__ == "__main__":
    main()
