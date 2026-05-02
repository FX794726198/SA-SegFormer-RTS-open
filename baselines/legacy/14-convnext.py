# ======== Imports ========
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ======== CBAM Attention Module ========
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        att = self.sigmoid(self.conv(x_cat))
        return att

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = x * self.ca(x)
        sa_map = self.sa(x)
        x = x * sa_map
        return x, sa_map  # 返回空间注意力权重，便于可视化

# ======== ConvNeXt Segmentation Model with CBAM ========
class ConvNeXtSegTV_CBAM(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
        self.encoder = backbone.features  # (B,768,8,8) for 256x256 input

        self.cbam = CBAM(768, reduction=16, kernel_size=7)   # <<<<<< CBAM 加在编码输出与解码入口之间

        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),   # 16x16
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),   # 32x32
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),   # 64x64
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),   # 256x256
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x, return_attention=False):
        feats = self.encoder(x)
        feats_cbam, sa_map = self.cbam(feats)
        out = self.decoder(feats_cbam)
        if return_attention:
            # 将注意力map上采样到输入尺寸，便于可视化
            sa_map_upsampled = torch.nn.functional.interpolate(
                sa_map, size=x.shape[2:], mode='bilinear', align_corners=False
            )
            return out, sa_map_upsampled
        return out

# ======== Dataset ========
class LandslideDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir  = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.files    = sorted(p for p in self.img_dir.iterdir()
                               if p.suffix.lower() in ('.jpg','.jpeg','.png','.tif','.tiff'))
        if not self.files:
            raise RuntimeError(f"No images in {img_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path  = self.files[idx]
        mask_path = self.mask_dir / (img_path.stem + '.tif')
        img  = np.array(Image.open(img_path).convert('RGB'))
        mask = (np.array(Image.open(mask_path), dtype=np.uint8) > 0).astype(np.uint8)
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()
        else:
            img  = transforms.ToTensor()(img)
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask

# ======== Metrics ========
def compute_batch_metrics(preds, masks, threshold=0.5, eps=1e-6):
    probs     = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()
    tp = (preds_bin * masks).sum((1,2,3))
    fp = (preds_bin * (1-masks)).sum((1,2,3))
    fn = ((1-preds_bin) * masks).sum((1,2,3))
    tn = ((1-preds_bin) * (1-masks)).sum((1,2,3))
    pixel_acc = ((tp+tn)/(tp+tn+fp+fn+eps)).mean().item()
    precision = (tp/(tp+fp+eps)).mean().item()
    recall    = (tp/(tp+fn+eps)).mean().item()
    f1        = (2*tp/(2*tp+fp+fn+eps)).mean().item()
    iou       = (tp/(tp+fp+fn+eps)).mean().item()
    return {
        'pixel_acc': pixel_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou
    }

# ======== Visualization (with Attention Map) ========
def visualize_samples(model, dataset, device, out_dir, n_samples=4, epoch=0):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    mean,std = np.array([0.485,0.456,0.406]), np.array([0.229,0.224,0.225])
    indices = random.sample(range(len(dataset)), n_samples)
    fig, axes = plt.subplots(n_samples, 4, figsize=(12, 3*n_samples))
    for i, idx in enumerate(indices):
        img, mask = dataset[idx]
        img_np = (img.cpu().numpy().transpose(1,2,0) * std + mean).clip(0,1)
        mask_np = mask[0].cpu().numpy()
        with torch.no_grad():
            pred, sa_map = model(img.unsqueeze(0).to(device), True)
            pred = torch.sigmoid(pred).detach().cpu().squeeze().numpy()
            sa_map = sa_map.detach().cpu().squeeze().numpy()
        pred_bin = (pred > 0.5).astype(np.uint8)
        # 画四列：原图、GT、预测、注意力
        axes[i,0].imshow(img_np);    axes[i,0].set_title('Image')
        axes[i,1].imshow(mask_np, cmap='gray'); axes[i,1].set_title('GT')
        axes[i,2].imshow(pred_bin, cmap='gray'); axes[i,2].set_title('Pred')
        axes[i,3].imshow(img_np, alpha=0.7)
        axes[i,3].imshow(sa_map, cmap='jet', alpha=0.5)
        axes[i,3].set_title('CBAM Attention')
        for ax in axes[i]:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'viz_epoch_{epoch:03d}.png'), dpi=300)
    plt.close(fig)

# ======== Main Script ========
def main():
    base           = '/home/featurize/work/newSEG'
    train_img_dir  = os.path.join(base, 'split_dataset/Train_img')
    train_mask_dir = os.path.join(base, 'split_dataset/Train_label')
    val_img_dir    = os.path.join(base, 'split_dataset/Test_img')
    val_mask_dir   = os.path.join(base, 'split_dataset/Test_label')

    model_name = 'convnext_cbam'
    save_dir   = os.path.join(base, f'Results_{model_name}')
    ckpt_dir   = os.path.join(save_dir, f'{model_name}_checkpoints')
    viz_dir    = os.path.join(save_dir, f'{model_name}_viz')
    plots_dir  = os.path.join(save_dir, f'{model_name}_plots')
    for d in (ckpt_dir, viz_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    train_transform = A.Compose([
        A.RandomResizedCrop(size=(256,256), scale=(0.8,1.0), ratio=(0.9,1.1), p=1.0),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(p=0.5),
        A.GaussNoise(p=0.2),
        A.Normalize(), ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(256,256),
        A.Normalize(), ToTensorV2()
    ])

    train_ds     = LandslideDataset(train_img_dir, train_mask_dir, train_transform)
    val_ds       = LandslideDataset(val_img_dir,   val_mask_dir,   val_transform)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = ConvNeXtSegTV_CBAM(pretrained=True).to(device)  # 用带CBAM的ConvNeXt

    criterion = lambda p, t: (
        nn.BCEWithLogitsLoss()(p, t)
        + smp.losses.DiceLoss(mode='binary')(p, t)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    history, best_loss = [], float('inf')
    for epoch in range(1, 201):
        tr_loss       = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, met = validate(model, val_loader, criterion, device)
        scheduler.step()
        history.append({'epoch': epoch, 'train_loss': tr_loss, 'val_loss': val_loss, **met})
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{model_name}_best.pth'))
        print(f"Epoch [{epoch}/200] Train Loss: {tr_loss:.4f} Val Loss: {val_loss:.4f} IoU: {met['iou']:.4f}")
        visualize_samples(model, val_ds, device, viz_dir, epoch=epoch)

    df = pd.DataFrame(history)
    df.to_excel(os.path.join(save_dir, f'training_history_{model_name}.xlsx'), index=False)

    plt.figure()
    plt.plot(df.epoch, df.train_loss, label='Train Loss')
    plt.plot(df.epoch, df.val_loss,   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir,'loss_curve.png'), dpi=300); plt.close()

    plt.figure()
    for m in ['iou','pixel_acc','precision','recall','f1_score']:
        plt.plot(df.epoch, df[m], label=m)
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(plots_dir,'metrics_curve.png'), dpi=300); plt.close()

# ======== Train & Validate ========
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc='Train', leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss  = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss  = 0
    metrics_sum = {k:0 for k in ['pixel_acc','precision','recall','f1_score','iou']}
    for imgs, masks in tqdm(loader, desc='Val', leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        total_loss += criterion(preds, masks).item() * imgs.size(0)
        bm = compute_batch_metrics(preds, masks)
        for k,v in bm.items():
            metrics_sum[k] += v * imgs.size(0)
    avg_loss    = total_loss / len(loader.dataset)
    avg_metrics = {k: metrics_sum[k] / len(loader.dataset) for k in metrics_sum}
    return avg_loss, avg_metrics

if __name__ == '__main__':
    main()
