import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ========== 超参数设置 ==========
BASE_DIR            = '/home/featurize/work/newSEG'
MODEL_NAME          = 'convnext_sav3'
IMG_SIZE            = (256, 256)
BATCH_SIZE          = 16
NUM_EPOCHS          = 400
LR                  = 3e-4
MAX_LR              = 1e-3
WEIGHT_DECAY        = 1e-4
NUM_WORKERS         = 4
THRESHOLD           = 0.4
HEADS               = 16
BACKBONE_PRETRAINED = True

# 对正类像素的权重（根据数据集正负比例可调整）
POS_WEIGHT          = 5.0

# 损失函数权重（适当降低 BCE 权重，加大 Dice/Focal 权重）
LOSS_WEIGHTS = {
    'bce':   0.5,
    'dice':  1.0,
    'focal': 1.0,
}


# ========== Self-Attention 模块（多头）==========
class SelfAttention2d(nn.Module):
    def __init__(self, in_channels, heads=HEADS):
        super().__init__()
        assert in_channels % heads == 0, "in_channels 必须能被 heads 整除"
        self.heads = heads
        self.scale = (in_channels // heads) ** -0.5
        self.qkv   = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=False)
        self.proj  = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]

        # [B, heads, HW, C//heads]
        q = q.permute(0,1,3,2)
        k = k.permute(0,1,2,3)
        v = v.permute(0,1,3,2)

        attn = torch.softmax((q @ k) * self.scale, dim=-1)  # [B,heads,HW,HW]
        out  = (attn @ v)                                   # [B,heads,HW,C//heads]
        out  = out.permute(0,1,3,2).reshape(B, C, H, W)
        out  = self.proj(out)
        return out


# ========== ConvNeXt + Self-Attention + Cross-Attention 分割模型 ==========
class ConvNeXtSegTV_SA(nn.Module):
    def __init__(self, pretrained=BACKBONE_PRETRAINED, heads=HEADS):
        super().__init__()
        # backbone
        backbone = models.convnext_tiny(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        self.encoder   = backbone.features

        # 在 encoder 输出通道上加一个 Self-Attention
        self.self_attn = SelfAttention2d(768, heads=heads)

        # 可学习的 class token，用于 cross-attention
        self.class_token = nn.Parameter(torch.zeros(1, 1, 768))

        # 多头 cross-attention：query=class_token, key/value=空间 token
        self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=heads, batch_first=True)

        # 简单 decoder
        self.decoder   = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64,  32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(32,   1, 1)
        )

        # 初始化 class token
        nn.init.trunc_normal_(self.class_token, std=0.02)

    def forward(self, x, return_attention=False):
        # x: [B,3,H,W]
        feats = self.encoder(x)                  # [B,768,H/32,W/32]
        feats_sa = self.self_attn(feats)         # [B,768,H/32,W/32]
        out      = self.decoder(feats_sa)        # [B,1,H,W]

        if not return_attention:
            return out

        # === 计算 cross-attention map ===
        B, C, H, W = feats_sa.shape
        tokens = feats_sa.view(B, C, H*W).permute(0,2,1)  # [B, N=H*W, C]
        cls_tok = self.class_token.expand(B, -1, -1)     # [B,1,C]

        # cross-attention
        # attn_weights: [B, query_len=1, key_len=N]
        _, attn_weights = self.cross_attn(
            query=cls_tok,      # [B,1,C]
            key=tokens,         # [B,N,C]
            value=tokens,       # [B,N,C]
            need_weights=True
        )

        # reshape 并上采样到原图
        attn_map    = attn_weights.squeeze(1).reshape(B, H, W)  # [B,H,W]
        attn_map_up = F.interpolate(
            attn_map.unsqueeze(1),
            size=x.shape[2:], mode='bilinear', align_corners=False
        ).squeeze(1)  # [B,H_orig,W_orig]

        return out, attn_map_up


# ========== Dataset ==========
class LandslideDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir  = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.files    = sorted(
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in ('.jpg','.jpeg','.png','.tif','.tiff')
        )
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


# ========== Metrics ==========
def compute_batch_metrics(preds, masks, threshold=THRESHOLD, eps=1e-6):
    probs     = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()

    tp = (preds_bin * masks).sum((1,2,3))
    fp = (preds_bin * (1-masks)).sum((1,2,3))
    fn = ((1-preds_bin) * masks).sum((1,2,3))
    tn = ((1-preds_bin) * (1-masks)).sum((1,2,3))

    precision_pos = (tp / (tp + fp + eps)).mean().item()
    recall_pos    = (tp / (tp + fn + eps)).mean().item()
    f1_pos        = (2*tp / (2*tp + fp + fn + eps)).mean().item()
    iou_pos       = (tp / (tp + fp + fn + eps)).mean().item()
    acc_pos       = recall_pos

    precision_neg = (tn / (tn + fn + eps)).mean().item()
    recall_neg    = (tn / (tn + fp + eps)).mean().item()
    f1_neg        = (2*tn / (2*tn + fn + fp + eps)).mean().item()
    iou_neg       = (tn / (tn + fn + fp + eps)).mean().item()
    acc_neg       = recall_neg

    precision = 0.5 * (precision_pos + precision_neg)
    recall    = 0.5 * (recall_pos + recall_neg)
    f1_score  = 0.5 * (f1_pos + f1_neg)
    iou       = 0.5 * (iou_pos + iou_neg)
    pixel_acc = ((tp + tn) / (tp + tn + fp + fn + eps)).mean().item()

    return {
        'precision_pos': precision_pos,
        'recall_pos':    recall_pos,
        'f1_pos':        f1_pos,
        'iou_pos':       iou_pos,
        'acc_pos':       acc_pos,
        'precision_neg': precision_neg,
        'recall_neg':    recall_neg,
        'f1_neg':        f1_neg,
        'iou_neg':       iou_neg,
        'acc_neg':       acc_neg,
        'precision':     precision,
        'recall':        recall,
        'f1_score':      f1_score,
        'iou':           iou,
        'pixel_acc':     pixel_acc
    }


# ========== Train & Validate ==========
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
    metrics_sum = {k:0 for k in [
        'precision_pos','recall_pos','f1_pos','iou_pos','acc_pos',
        'precision_neg','recall_neg','f1_neg','iou_neg','acc_neg',
        'precision','recall','f1_score','iou','pixel_acc']}
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


# ========== Visualization ==========
def visualize_samples(model, dataset, device, out_dir, n_samples=4, epoch=0):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    mean,std = np.array([0.485,0.456,0.406]), np.array([0.229,0.224,0.225])
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    fig, axes = plt.subplots(n_samples, 6, figsize=(18, 3*n_samples))

    for i, idx in enumerate(indices):
        img, mask = dataset[idx]
        img_np    = (img.cpu().numpy().transpose(1,2,0) * std + mean).clip(0,1)
        mask_np   = mask[0].cpu().numpy()
        img_t     = img.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_mask, cross_map_up = model(img_t, return_attention=True)
        pred_prob   = torch.sigmoid(pred_mask).cpu().squeeze().numpy()
        cross_np    = cross_map_up.cpu().squeeze(0).numpy()

        # encoder 特征平均
        feats = model.encoder(img_t).detach().cpu().squeeze(0)
        mean_feat_up = F.interpolate(
            feats.mean(0,keepdim=True).unsqueeze(0),
            size=IMG_SIZE, mode='bilinear', align_corners=False
        ).squeeze().numpy()

        axes[i,0].imshow(img_np); axes[i,0].set_title('Image'); axes[i,0].axis('off')
        axes[i,1].imshow(mask_np, cmap='gray'); axes[i,1].set_title('Ground Truth'); axes[i,1].axis('off')
        axes[i,2].imshow((pred_prob>THRESHOLD).astype(np.uint8), cmap='gray'); axes[i,2].set_title('Prediction'); axes[i,2].axis('off')
        im3 = axes[i,3].imshow(cross_np, cmap='jet');    axes[i,3].set_title('Cross-Attention'); axes[i,3].axis('off')
        im4 = axes[i,4].imshow(mean_feat_up, cmap='viridis'); axes[i,4].set_title('Encoder Mean'); axes[i,4].axis('off')
        im5 = axes[i,5].imshow(pred_prob, cmap='plasma'); axes[i,5].set_title('Decoder Probability'); axes[i,5].axis('off')

        fig.colorbar(im3, ax=axes[i,3], fraction=0.046, pad=0.04)
        fig.colorbar(im4, ax=axes[i,4], fraction=0.046, pad=0.04)
        fig.colorbar(im5, ax=axes[i,5], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'viz_epoch_{epoch:03d}.png'), dpi=300)
    plt.close(fig)


# ========== Main ==========
def main():
    train_img_dir  = os.path.join(BASE_DIR, 'split_dataset/Train_img')
    train_mask_dir = os.path.join(BASE_DIR, 'split_dataset/Train_label')
    val_img_dir    = os.path.join(BASE_DIR, 'split_dataset/Test_img')
    val_mask_dir   = os.path.join(BASE_DIR, 'split_dataset/Test_label')

    save_dir   = os.path.join(BASE_DIR, f'Results_{MODEL_NAME}')
    ckpt_dir   = os.path.join(save_dir, f'{MODEL_NAME}_checkpoints')
    viz_dir    = os.path.join(save_dir, f'{MODEL_NAME}_viz')
    plots_dir  = os.path.join(save_dir, f'{MODEL_NAME}_plots')
    for d in (ckpt_dir, viz_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    # 数据增强
    train_transform = A.Compose([
        A.RandomResizedCrop(IMG_SIZE, scale=(0.5,1.0), ratio=(0.9,1.1), p=1.0),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(p=0.5),
        A.GaussNoise(p=0.2),
        A.Normalize(), ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(*IMG_SIZE), A.Normalize(), ToTensorV2()
    ])

    train_ds     = LandslideDataset(train_img_dir, train_mask_dir, train_transform)
    val_ds       = LandslideDataset(val_img_dir,   val_mask_dir,   val_transform)

    # 正样本过采样
    mask_paths = [train_ds.mask_dir / (p.stem + '.tif') for p in train_ds.files]
    has_pos = []
    for mpath in mask_paths:
        m = np.array(Image.open(mpath), dtype=np.uint8)
        has_pos.append(m.max() > 0)
    weights = [2.0 if flag else 1.0 for flag in has_pos]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds,
                              batch_size=BATCH_SIZE,
                              sampler=sampler,
                              num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=NUM_WORKERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = ConvNeXtSegTV_SA(pretrained=BACKBONE_PRETRAINED).to(device)

    # 损失 & 优化器
    bce   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device))
    dice  = smp.losses.DiceLoss(mode='binary')
    focal = smp.losses.FocalLoss(mode='binary', alpha=0.8)
    def criterion(preds, targets):
        return (
            LOSS_WEIGHTS['bce']   * bce(preds, targets)
          + LOSS_WEIGHTS['dice']  * dice(preds, targets)
          + LOSS_WEIGHTS['focal'] * focal(preds, targets)
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR,
        epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='linear'
    )

    history, best_loss = [], float('inf')
    for epoch in range(1, NUM_EPOCHS+1):
        tr_loss       = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, met = validate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({'epoch': epoch, 'train_loss': tr_loss, 'val_loss': val_loss, **met})

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{MODEL_NAME}_best.pth'))

        # 每个 epoch 更新 Excel
        pd.DataFrame(history).to_excel(
            os.path.join(save_dir, f'training_history_{MODEL_NAME}.xlsx'),
            index=False
        )

        # 控制台输出完整指标
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
        print(f"Loss: Training {tr_loss:.4f}    Validation {val_loss:.4f}")
        print(f"{'类别':<12} {'Precision':>10} {'Recall':>10} {'F1 Score':>10} {'IoU':>10} {'Accuracy':>10}")
        print(f"{'Positive':<12} {met['precision_pos']:^10.4f} {met['recall_pos']:^10.4f} {met['f1_pos']:^10.4f} {met['iou_pos']:^10.4f} {met['acc_pos']:^10.4f}")
        print(f"{'Negative':<12} {met['precision_neg']:^10.4f} {met['recall_neg']:^10.4f} {met['f1_neg']:^10.4f} {met['iou_neg']:^10.4f} {met['acc_neg']:^10.4f}")
        print(f"{'Overall':<12} {met['precision']:^10.4f} {met['recall']:^10.4f} {met['f1_score']:^10.4f} {met['iou']:^10.4f} {met['pixel_acc']:^10.4f}\n")

        visualize_samples(model, val_ds, device, viz_dir, n_samples=4, epoch=epoch)

    # 训练结束后保存曲线
    df = pd.DataFrame(history)
    plt.figure(); plt.plot(df.epoch, df.train_loss, label='Train Loss'); plt.plot(df.epoch, df.val_loss, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir,'loss_curve.png'), dpi=300); plt.close()

    plt.figure()
    for m in ['precision_pos','precision_neg','precision',
              'recall_pos','recall_neg','recall',
              'f1_pos','f1_neg','f1_score',
              'iou_pos','iou_neg','iou',
              'acc_pos','acc_neg','pixel_acc']:
        plt.plot(df.epoch, df[m], label=m)
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir,'metrics_curve.png'), dpi=300); plt.close()


if __name__ == '__main__':
    main()
