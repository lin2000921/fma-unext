import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
import numpy as np
from albumentations import Compose, Resize, Normalize
from torch.nn import functional as F
from tqdm import tqdm

import archs
from dataset import Dataset
from utils import AverageMeter
from archs import UNext


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self):
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, scale_factor=16, mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()  # ✅ 修复点
        cam -= np.min(cam)
        cam /= np.max(cam) + 1e-8
        return cam


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='new_unext_Grad-CAM', help='model name')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(f'models/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cudnn.benchmark = True

    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    model.load_state_dict(torch.load(f'models/{args.name}/model.pth'))
    model.eval().cuda()

    test_path = r'D:\UNeXt-pytorch-main\inputs\new_wusun\image\test'
    test_mask_path = r'D:\UNeXt-pytorch-main\inputs\new_wusun\mask\test'
    test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in glob(os.path.join(test_path, '*' + config['img_ext']))]

    test_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])

    test_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=test_path,
        mask_dir=test_mask_path,
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    gradcam_dir = os.path.join('outputs', config['name'], 'gradcam')
    os.makedirs(gradcam_dir, exist_ok=True)

    # 👇 修改为你的目标层名称（比如 encoder3）
    target_layer = model.decoder3
    gradcam = GradCAM(model, target_layer)

    for input, target, meta in tqdm(test_loader):
        input = input.cuda()
        target = target.cuda()
        model.zero_grad()

        output = model(input)

        # 🔥 以第 0 通道作为目标类别（适用于二分类）
        score = output[:, 0, :, :].mean()
        score.backward()

        cam = gradcam.generate()

        # 将 cam resize 成与原图一样的尺寸（H, W）
        cam_resized = cv2.resize(cam, (config['input_w'], config['input_h']))
        cam_resized = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)

        # ✅ 处理原图
        img_np = input[0].detach().cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
        img_np = (img_np - img_np.min()) / (img_np.max() + 1e-8)  # Normalize to [0, 1]
        img_np = np.uint8(255 * img_np)
        if img_np.shape[2] == 1:  # 如果是灰度图
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

        # ✅ 保证 heatmap 与原图尺寸一致
        if heatmap.shape[:2] != img_np.shape[:2]:
            heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

        # ✅ 合成
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        cv2.imwrite(os.path.join(gradcam_dir, meta['img_id'][0] + '_cam.jpg'), superimposed_img)

    print(f"✅ Grad-CAM 可视化完成，图像保存在：{gradcam_dir}")


if __name__ == '__main__':
    main()
