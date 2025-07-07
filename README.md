# FMA-UNeXt

本项目为 FMA-UNeXt 网络的官方 PyTorch 代码库。

FMA-UNeXt 是一种基于 MLP 的高效医学图像分割网络，结合了多层感知机与卷积结构，能够在保证准确率的同时实现快速的医学图像分割，适用于多种医学影像数据集。

## 快速开始

### 环境依赖

建议使用 Python 3.6 及以上、CUDA 10.1 及以上。推荐通过 conda 安装依赖：

```bash
conda env create -f environment.yml
conda activate fma-unext
```

或使用 pip 安装主要依赖：

```bash
pip install torch torchvision opencv-python timm mmcv-full
```

### 数据集格式

请将数据集按照如下结构整理：

```
inputs
└── <dataset name>
    ├── images
    │   ├── 001.png
    │   ├── ...
    └── masks
        ├── 0
        │   ├── 001.png
        │   ├── ...
        └── 1
            ├── 001.png
            ├── ...
```

对于二分类分割任务，仅需使用 `0` 文件夹。

### 训练与验证

1. 训练模型：
   ```bash
   python train.py --dataset <dataset name> --arch FMA-UNeXt --name <exp name> --img_ext .png --mask_ext .png --lr 0.0001 --epochs 500 --input_w 512 --input_h 512 --b 8
   ```
2. 验证模型：
   ```bash
   python val.py --name <exp name>
   ```

## 参考

如在研究中使用本项目，请引用相关论文。

---
如需详细使用说明或遇到问题，请参考代码注释或联系作者。
