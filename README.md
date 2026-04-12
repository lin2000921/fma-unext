# FMA-UNeXt

Official PyTorch implementation of the manuscript:

**Frequency-Adaptive and Multi-Scale Attention Network for Underwater Hull Fouling Segmentation**

This repository corresponds directly to the manuscript currently submitted to *The Visual Computer*.

If you find this repository useful in your research, please consider citing the corresponding manuscript.

---

## Highlights

* Lightweight semantic segmentation framework for underwater hull fouling segmentation
* Based on the UNeXt architecture
* Integrates Frequency-Adaptive Dilated Convolution (FADC), Efficient Multi-scale Convolutional Attention Decoder (EMCAD), and Efficient Multi-Scale Attention (EMA)
* Designed for low-contrast underwater images and large-scale fouling regions
* Achieves 92.41% IoU, 96.03% DSC, and 96.14% Accuracy on the underwater hull fouling dataset

---

## Overview

Underwater hull fouling segmentation is challenging because underwater images often suffer from:

* Low contrast
* Uneven illumination
* Suspended particles and noise
* Large variations in fouling scales
* Strong similarity between fouling regions and background areas

To address these challenges, we propose FMA-UNeXt, an improved UNeXt-based semantic segmentation framework.

The proposed network consists of three key modules:

* **FADC**: Frequency-Adaptive Dilated Convolution for enhancing low-frequency structures and high-frequency details
* **EMCAD**: Efficient Multi-scale Convolutional Attention Decoder for reconstructing fouling regions with large scale variations
* **EMA**: Efficient Multi-Scale Attention for improving global-local feature fusion and large-region segmentation

---

## Network Architecture

The overall FMA-UNeXt framework includes:

1. Encoder with FADC modules replacing standard convolutions
2. Tokenized MLP module inherited from UNeXt
3. EMA modules inserted after the encoder and after the tokenized MLP stage
4. EMCAD decoder for multi-scale feature reconstruction

```text
Encoder (FADC) -> EMA -> Tokenized MLP -> EMA -> EMCAD Decoder -> Prediction
```

You can place your network structure figure in:

```text
figures/framework.png
```

<p align="center">
  <img src="imgs/fma-unext.png" width="800"/>
</p>


---

## Experimental Results

Performance of FMA-UNeXt on the underwater hull fouling segmentation task:

| Method    | IoU (%) | DSC (%) | Acc (%) | Params (M) | GFLOPs | Inference Speed (ms) |
| --------- | ------: | ------: | ------: | ---------: | -----: | -------------------: |
| UNeXt     |   85.84 |   92.19 |   92.95 |       1.47 |   0.57 |                 3.87 |
| FMA-UNeXt |   92.41 |   96.03 |   96.14 |       1.50 |   0.51 |                13.07 |

Compared with the original UNeXt, the proposed method improves:

* IoU by 6.57%
* DSC by 3.84%
* Accuracy by 3.19%

---

## Using the code:

The code is stable while using Python 3.6.13, CUDA >=10.1

- Clone this repository:
```bash
git clone https://github.com/lin2000921/fma-unext.git
cd fma-unext-master
```

To install all the dependencies using conda:

```bash
conda env create -f environment.yml
conda activate unext
```

If you prefer pip, install following versions:

```bash
timm==0.3.2
mmcv-full==1.2.7
torch==1.7.1
torchvision==0.8.2
opencv-python==4.5.1.48
```

---

## Repository Structure

```text
FMA-UNeXt
├── datasets
│   └── <dataset_name>
├── models
├── losses
├── utils
├── configs
├── outputs
├── pretrained_weights
├── figures
├── train.py
├── test.py
├── infer.py
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset used in this study contains underwater hull fouling images captured by underwater cameras.

Due to confidentiality restrictions related to ship hull condition data, the complete dataset cannot be publicly released.

To facilitate reproducibility, this repository provides:

* Dataset folder structure
* Mask organization format
* Data preprocessing pipeline
* Data augmentation strategy
* Training and evaluation scripts
* Recommended train/validation/test split protocol

The dataset used in the manuscript contains 800 underwater hull fouling images, which were expanded to 13,920 images through data augmentation.

The dataset was divided into training, validation, and test sets with a ratio of 8:1:1.

## Data Format

Make sure to put the files as the following structure (e.g. the number of classes is 2):

```
inputs
└── <dataset name>
    ├── images
    |   ├── 001.png
    │   ├── 002.png
    │   ├── 003.png
    │   ├── ...
    |
    └── masks
        ├── 0
        |   ├── 001.png
        |   ├── 002.png
        |   ├── 003.png
        |   ├── ...
        |
        └── 1
            ├── 001.png
            ├── 002.png
            ├── 003.png
            ├── ...
```

For binary segmentation problems, just use folder 0.

### Notes

* Images and masks should have the same file names
* Input images are resized to 256 × 256 before training
* Masks are binary segmentation labels
* Folder `0` and folder `1` represent different segmentation categories

---

## Data Augmentation

The following augmentation methods were used in the manuscript:

* Random cropping
* Translation transformation
* Scaling transformation
* Rotation transformation
* Horizontal flipping
* Vertical flipping

These augmentations were applied to improve the robustness of the model under complex underwater environments.

---

## Training

Before training, please configure the dataset path and training parameters.

Example:

```bash
python train.py --dataset hull_dataset --arch FMA_UNeXt
```

Main training settings:

```text
Epochs: 300
Batch Size: 16
Optimizer: Adam
Initial Learning Rate: 0.001
Input Size: 256 × 256
```

---

## Testing

After training, evaluate the model using:

```bash
python test.py --dataset hull_dataset --weights pretrained_weights/fma_unext_best.pth
```

The evaluation metrics include:

* IoU
* DSC
* Accuracy

---

## Inference

For single-image or folder inference:

```bash
python infer.py --input demo_images --output outputs
```

Prediction results will be saved in the `outputs` folder.

---

---

## Reproducibility Statement

This repository is the official implementation of the manuscript currently submitted to *The Visual Computer*.

To improve transparency and reproducibility, we provide:

* Source code
* Environment configuration
* Training and testing scripts
* Dataset organization instructions
* Experimental settings
* Evaluation metrics
* Pretrained weights

A permanent archived release with DOI will be provided through Zenodo.

```text
Zenodo DOI: Coming Soon
```

---

## Citation

If you use this repository in your work, please cite the corresponding manuscript:

```bibtex
@article{lin2026fmaunext,
  title={Frequency-Adaptive and Multi-Scale Attention Network for Underwater Hull Fouling Segmentation},
  author={Yihao Lin and Yajuan Gu and Lunming Qin and Liang Xue and Houqin Bian and Xi Wang},
  journal={Under review at The Visual Computer},
  year={2026}
}
```

---

## License

Please add an open-source license for this repository, such as MIT License.

```text
MIT License
```

---

## Contact

For questions related to the code or manuscript, please contact:

* Yihao Lin: [linyihao@mail.shiep.edu.cn](mailto:linyihao@mail.shiep.edu.cn)
* Lunming Qin: [lunming.qin@shiep.edu.cn](mailto:lunming.qin@shiep.edu.cn)
