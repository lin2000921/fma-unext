# FMA-UNeXt

本项目为 FMA-UNeXt 网络的官方 PyTorch 代码库。


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
