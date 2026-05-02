import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt

# ==== 复制训练时的模型结构 ====
class ConvNeXtSegTV(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        backbone = torchvision.models.convnext_tiny(weights=None)
        self.encoder = backbone.features
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decoder(feats)
        return out

# ==== 预处理，与训练时一致 ====
from albumentations.pytorch import ToTensorV2
import albumentations as A

transform = A.Compose([
    A.Resize(256,256),
    A.Normalize(),
    ToTensorV2()
])

def segment_and_visualize(
    model_path,
    input_dir,
    output_dir,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    threshold=0.5,
    alpha=0.4   # 透明度
):
    os.makedirs(output_dir, exist_ok=True)
    # 加载模型
    model = ConvNeXtSegTV().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 遍历图片
    img_suffix = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    img_list = [p for p in Path(input_dir).iterdir() if p.suffix.lower() in img_suffix]
    if not img_list:
        print("No images found in input directory.")
        return

    for img_path in img_list:
        # 加载并预处理
        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil)
        aug = transform(image=img_np)
        img_tensor = aug['image'].unsqueeze(0).to(device)  # (1,3,256,256)

        # 推理分割
        with torch.no_grad():
            pred = model(img_tensor)
            pred_mask = torch.sigmoid(pred).cpu().squeeze().numpy()
            mask_bin = (pred_mask > threshold).astype(np.uint8)

        # 上采样到原图分辨率
        mask_bin = Image.fromarray(mask_bin).resize(img_pil.size, resample=Image.NEAREST)
        mask_bin = np.array(mask_bin)  # (H,W)

        # 可视化叠加：红=滑坡，绿=非滑坡
        overlay = np.zeros((*img_pil.size[::-1], 4), dtype=np.uint8)  # (H,W,4)
        overlay[..., 0] = (mask_bin==1) * 255      # 红通道
        overlay[..., 1] = (mask_bin==0) * 255      # 绿通道
        overlay[..., 3] = int(alpha * 255)         # Alpha通道

        # 叠加原图
        base = img_pil.convert('RGBA')
        blended = Image.alpha_composite(base, Image.fromarray(overlay))

        # 输出高分辨率300dpi图片
        save_path = Path(output_dir) / f"{img_path.stem}_seg_vis.png"
        blended.save(save_path, dpi=(300,300))
        print(f"Saved: {save_path}")

if __name__ == '__main__':
    # ======= 需要自定义的路径 =======
    model_pth = '/Users/fengxiao/Desktop/Work/语义分割/Results/Convnext/Results_convnext/convnext_checkpoints/convnext_best.pth'  # 最佳模型
    input_img_dir = '/Users/fengxiao/Desktop/Work/语义分割/Data/split_dataset/Test_img'     # 测试图片所在路径
    output_vis_dir = '/Users/fengxiao/Desktop/Work/语义分割/Results/Convnext/Results_convnext/Test results'# 可视化输出路径

    segment_and_visualize(
        model_path=model_pth,
        input_dir=input_img_dir,
        output_dir=output_vis_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        threshold=0.5,    # 阈值可调
        alpha=0.4         # 透明度(0-1)
    )
