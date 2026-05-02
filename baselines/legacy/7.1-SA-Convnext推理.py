import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms

# ==== SelfAttention2d====
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention2d(nn.Module):
    def __init__(self, in_channels, heads=4):
        super().__init__()
        assert in_channels % heads == 0, "in_channels 必须能被 heads 整除"
        self.heads = heads
        self.scale = (in_channels // heads) ** -0.5

        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, return_attn=False):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]   # (B, heads, C//heads, HW)

        q = q.permute(0,1,3,2)    # (B, heads, HW, C//heads)
        k = k.permute(0,1,2,3)    # (B, heads, C//heads, HW)
        v = v.permute(0,1,3,2)    # (B, heads, HW, C//heads)

        attn = torch.softmax((q @ k) * self.scale, dim=-1)    # (B, heads, HW, HW)
        out = (attn @ v)    # (B, heads, HW, C//heads)
        out = out.permute(0,1,3,2).reshape(B, C, H, W)
        out = self.proj(out)

        if return_attn:
            # 返回平均注意力 (B, HW, HW)，HW=H*W，可后续聚合
            attn_map = attn.mean(1)   # (B, HW, HW)
            return out, attn_map
        return out


# ==== ConvNeXt+Self-Attn分割模型 ====
class ConvNeXtSegTV_SA(torch.nn.Module):
    def __init__(self, pretrained=False, heads=4):
        super().__init__()
        import torchvision
        backbone = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
        self.encoder = backbone.features  # (B,768,8,8)
        self.self_attn = SelfAttention2d(768, heads=heads)
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(768, 256, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),   # 16x16
            torch.nn.Conv2d(256, 128, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),   # 32x32
            torch.nn.Conv2d(128, 64, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),   # 64x64
            torch.nn.Conv2d(64, 32, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),   # 256x256
            torch.nn.Conv2d(32, 1, 1)
        )

    def forward(self, x, return_attention=False):
        feats = self.encoder(x)
        if return_attention:
            feats_sa, attn_map = self.self_attn(feats, return_attn=True)
            out = self.decoder(feats_sa)
            attn_score = attn_map.mean(1).reshape(x.shape[0], 8, 8)
            attn_score_up = F.interpolate(attn_score.unsqueeze(1), size=x.shape[2:], mode='bilinear', align_corners=False)
            return out, attn_score_up, feats
        else:
            feats_sa = self.self_attn(feats)
            out = self.decoder(feats_sa)
            return out

# =========== 批量推理与可视化 ===========
def infer_and_visualize_folder(
    model_pth,
    model_class,
    image_folder,
    output_folder,
    device='cuda',
    img_size=256
):
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    # 加载模型
    model = model_class(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=device))
    model.eval()

    # 预处理
    mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    img_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff'))]
    for fname in img_files:
        img_path = os.path.join(image_folder, fname)
        img_raw = Image.open(img_path).convert('RGB')
        img_tensor = transform(img_raw).unsqueeze(0).to(device)

        with torch.no_grad():
            pred, attn_map, feats = model(img_tensor, return_attention=True)
            prob_map = torch.sigmoid(pred).cpu().squeeze().numpy()
            attn_map_np = attn_map.cpu().squeeze().numpy()
            feats_mean = feats.mean(1).cpu().squeeze().numpy()

        img_np = np.array(img_raw.resize((img_size,img_size))) / 255.0
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        axes[0].imshow(img_np);           axes[0].set_title('Image');        axes[0].axis('off')
        im1 = axes[1].imshow(attn_map_np, cmap='jet'); axes[1].set_title('Self-Attn'); axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        im2 = axes[2].imshow(feats_mean, cmap='viridis'); axes[2].set_title('Encoder Mean'); axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        im3 = axes[3].imshow(prob_map, cmap='hot'); axes[3].set_title('Decoder Prob'); axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        axes[4].imshow((prob_map > 0.5).astype(np.uint8), cmap='gray'); axes[4].set_title('Mask>0.5'); axes[4].axis('off')
        plt.tight_layout()
        out_path = os.path.join(output_folder, os.path.splitext(fname)[0] + '_viz.png')
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

        # 保存二值mask
        mask_out = os.path.join(output_folder, os.path.splitext(fname)[0] + '_mask.png')
        Image.fromarray((prob_map > 0.5).astype(np.uint8)*255).save(mask_out)

    print(f"全部处理完成，结果保存在: {output_folder}")

# =========== 用法举例 ===========
if __name__ == '__main__':
    infer_and_visualize_folder(
        model_pth='/Users/fengxiao/Desktop/Work/语义分割/Results/SA-Convnext/Results_convnext_sa/convnext_sa_checkpoints/convnext_sa_best.pth',                # 你的权重路径
        model_class=ConvNeXtSegTV_SA,
        image_folder='/Users/fengxiao/Desktop/Work/语义分割/Data/split_dataset/Test_img' ,          # 输入图片文件夹
        output_folder='/Users/fengxiao/Desktop/Work/语义分割/Results/SA-Convnext/Test',  # 输出文件夹
        device='cuda',
        img_size=256
    )
