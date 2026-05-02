
import os
# 在导入 torch 之前设置可扩展分段，减少碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch.encoders import get_encoder
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ========== 超参数设置 ==========
BASE_DIR     = '/home/featurize/work/newSEG'
MODEL_NAME   = 'true_segformer_CA_lowres'
IMG_SIZE     = (256, 256)
BATCH_SIZE   = 8
NUM_EPOCHS   = 200
LR           = 3e-4
MAX_LR       = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4
THRESHOLD    = 0.4

POS_WEIGHT   = 5.0
LOSS_WEIGHTS = {'bce':0.5, 'dice':1.0, 'focal':1.0}

# ========== Cross-Attention Block ==========
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=False)

    def forward(self, query, context):
        B, C, Hq, Wq = query.shape
        _, _, Hc, Wc = context.shape
        seq_q  = query.view(B, C, Hq*Wq).permute(2,0,1)
        seq_kv = context.view(B, C, Hc*Wc).permute(2,0,1)
        attn_out, attn_w = self.cross_attn(seq_q, seq_kv, seq_kv)
        out = attn_out.permute(1,2,0).view(B, C, Hq, Wq)
        return out, attn_w

# ========== SegFormer with Heatmap (Memory-Optimized) ==========
class TrueSegFormerWithHeatmap(nn.Module):
    def __init__(self, num_classes=1, dims=128, backbone="mit_b0", pretrained=True,
                 attn_heads=8, attn_downscale=8):
        super().__init__()
        # 将 in_channels 从 3 改为 4，以接收光谱指数通道
        enc = get_encoder(backbone, in_channels=4, depth=4,
                          weights="imagenet" if pretrained else None)
        if hasattr(enc, 'enable_gradient_checkpointing'):
            enc.enable_gradient_checkpointing()
        self.encoder        = enc
        self.channels       = [c for c in enc.out_channels if c>0]
        self.proj_heads     = nn.ModuleList([nn.Conv2d(ch, dims, 1) for ch in self.channels])
        self.cross_attn_blk = CrossAttentionBlock(dim=dims, num_heads=attn_heads)
        self.attn_downscale = attn_downscale
        self.classifier     = nn.Sequential(
            nn.Conv2d(dims, dims, 3, padding=1),
            nn.BatchNorm2d(dims), nn.ReLU(inplace=True),
            nn.Conv2d(dims, num_classes, 1)
        )

    def forward(self, x, return_heatmap=False):
        B, _, H, W = x.shape
        feats_all = self.encoder(x)
        feats     = [f for f in feats_all if f.shape[1]>0]

        fusion = None
        for feat, proj in zip(feats, self.proj_heads):
            p = proj(feat)
            p = F.interpolate(p, size=(H, W), mode='bilinear', align_corners=False)
            fusion = p if fusion is None else fusion + p

        sh, sw     = H//self.attn_downscale, W//self.attn_downscale
        fused_small= F.adaptive_avg_pool2d(fusion, (sh, sw))
        ctx_small  = F.adaptive_avg_pool2d(self.proj_heads[0](feats[0]), (sh, sw))
        attn_small, attn_w = self.cross_attn_blk(fused_small, ctx_small)

        attn_out = F.interpolate(attn_small, size=(H, W), mode='bilinear', align_corners=False)
        out      = self.classifier(attn_out)
        if not return_heatmap:
            return out

        aw      = attn_w.permute(1,0,2)      # (B, Sq, Sk)
        heat_s  = aw.mean(dim=2).view(B,1,sh,sw)
        mn      = heat_s.view(B,-1).amin(1).view(B,1,1,1)
        mx      = heat_s.view(B,-1).amax(1).view(B,1,1,1)
        heat_s  = (heat_s - mn) / (mx - mn + 1e-6)
        heatmap = F.interpolate(heat_s, size=(H, W), mode='bilinear', align_corners=False)

        return out, heatmap.squeeze(1)

# ========== Dataset ==========
class LandslideDataset(Dataset):
    def __init__(self, img_dir, mask_dir, spec_dir, transform=None):
        self.img_dir   = Path(img_dir)
        self.mask_dir  = Path(mask_dir)
        self.spec_dir  = Path(spec_dir)
        self.files     = sorted([p for p in self.img_dir.iterdir()
                                  if p.suffix.lower() in ('.jpg','.png','.tif')])
        if not self.files:
            raise RuntimeError(f"No images in {img_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        stem = self.files[idx].stem
        img  = np.array(Image.open(self.files[idx]).convert('RGB'))
        msk  = (np.array(Image.open(self.mask_dir/f'{stem}.tif'),
                         dtype=np.uint8) > 0).astype(np.uint8)
        spec = np.array(Image.open(self.spec_dir/f'{stem}.tif'), dtype=np.float32)

        if self.transform:
            aug  = self.transform(image=img, mask=msk, spec=spec)
            img, msk, spec = aug['image'], aug['mask'], aug['spec']

            # 处理 mask
            if isinstance(msk, np.ndarray):
                msk = torch.from_numpy(msk).unsqueeze(0).float()
            else:
                if msk.ndim == 2:
                    msk = msk.unsqueeze(0)
                msk = msk.float()

            # 处理 spec
            if isinstance(spec, np.ndarray):
                spec = torch.from_numpy(spec).float()
                if spec.ndim == 2:
                    spec = spec.unsqueeze(0)
            else:
                if spec.ndim == 2:
                    spec = spec.unsqueeze(0)
                spec = spec.float()
        else:
            img  = ToTensorV2()(image=img)['image']
            msk  = torch.from_numpy(msk).unsqueeze(0).float()
            spec = torch.from_numpy(spec).unsqueeze(0).float()

        # 拼接为 4 通道：RGB + spec
        img = torch.cat([img, spec], dim=0)
        return img, msk

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
        'precision_pos': precision_pos, 'recall_pos': recall_pos,
        'f1_pos': f1_pos,               'iou_pos': iou_pos,
        'acc_pos': acc_pos,
        'precision_neg': precision_neg, 'recall_neg': recall_neg,
        'f1_neg': f1_neg,               'iou_neg': iou_neg,
        'acc_neg': acc_neg,
        'precision': precision,         'recall': recall,
        'f1_score': f1_score,           'iou': iou,
        'pixel_acc': pixel_acc
    }

# ========== Train & Validate ==========
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc='Train', leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        with autocast():
            preds = model(imgs)
            loss  = criterion(preds, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, criterion, device, scaler):
    model.eval()
    total_loss = 0
    metrics_sum = {k:0 for k in [
        'precision_pos','recall_pos','f1_pos','iou_pos','acc_pos',
        'precision_neg','recall_neg','f1_neg','iou_neg','acc_neg',
        'precision','recall','f1_score','iou','pixel_acc']}
    for imgs, masks in tqdm(loader, desc='Val', leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        with autocast():
            preds = model(imgs)
            loss  = criterion(preds, masks)
        total_loss += loss.item() * imgs.size(0)
        bm = compute_batch_metrics(preds, masks)
        for k, v in bm.items():
            metrics_sum[k] += v * imgs.size(0)
    avg_loss = total_loss / len(loader.dataset)
    avg_metrics = {k: metrics_sum[k] / len(loader.dataset) for k in metrics_sum}
    return avg_loss, avg_metrics

def visualize_samples(model, dataset, device, out_dir, n_samples=4, epoch=0):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    mean, std = np.array([0.485,0.456,0.406]), np.array([0.229,0.224,0.225])
    idxs = np.random.choice(len(dataset), n_samples, replace=False)
    fig, axes = plt.subplots(n_samples, 5, figsize=(15, 3*n_samples))
    for i, idx in enumerate(idxs):
        img, mask = dataset[idx]
        img_np = (img[:3].cpu().numpy().transpose(1,2,0) * std + mean).clip(0,1)
        img_t   = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred, heat = model(img_t, return_heatmap=True)
        prob    = torch.sigmoid(pred).cpu().squeeze().numpy()
        heat_np = heat.cpu().squeeze().numpy()
        axes[i,0].imshow(img_np); axes[i,0].set_title('Image'); axes[i,0].axis('off')
        axes[i,1].imshow(mask[0].cpu(), cmap='gray'); axes[i,1].set_title('GT'); axes[i,1].axis('off')
        axes[i,2].imshow((prob>THRESHOLD).astype(np.uint8), cmap='gray'); axes[i,2].set_title('Pred'); axes[i,2].axis('off')
        im3 = axes[i,3].imshow(heat_np, cmap='jet'); axes[i,3].set_title('Cross-Attn Heatmap'); axes[i,3].axis('off')
        im4 = axes[i,4].imshow(prob, cmap='plasma'); axes[i,4].set_title('Prob'); axes[i,4].axis('off')
        fig.colorbar(im3, ax=axes[i,3], fraction=0.046, pad=0.04)
        fig.colorbar(im4, ax=axes[i,4], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'viz_epoch_{epoch:03d}.png'), dpi=300)
    plt.close(fig)

# ========== Main ==========
def main():
    train_img_dir  = os.path.join(BASE_DIR, 'split_dataset/Train_img')
    train_mask_dir = os.path.join(BASE_DIR, 'split_dataset/Train_label')
    train_spec_dir = os.path.join(BASE_DIR, 'split_dataset/Train_spec')

    val_img_dir    = os.path.join(BASE_DIR, 'split_dataset/Test_img')
    val_mask_dir   = os.path.join(BASE_DIR, 'split_dataset/Test_label')
    val_spec_dir   = os.path.join(BASE_DIR, 'split_dataset/Test_spec')

    save_dir  = os.path.join(BASE_DIR, f'Results_{MODEL_NAME}')
    ckpt_dir  = os.path.join(save_dir, f'{MODEL_NAME}_checkpoints')
    viz_dir   = os.path.join(save_dir, f'{MODEL_NAME}_viz')
    plots_dir = os.path.join(save_dir, f'{MODEL_NAME}_plots')
    for d in (ckpt_dir, viz_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    train_transform = A.Compose([
            A.RandomResizedCrop(IMG_SIZE, scale=(0.5,1.0), ratio=(0.9,1.1), p=1.0),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(p=0.5),
            A.GaussNoise(p=0.2), A.Normalize(), ToTensorV2()
        ],
        additional_targets={'spec':'image'}
    )
    val_transform = A.Compose([
            A.Resize(*IMG_SIZE), A.Normalize(), ToTensorV2()
        ],
        additional_targets={'spec':'image'}
    )

    train_ds = LandslideDataset(train_img_dir, train_mask_dir, train_spec_dir, train_transform)
    val_ds   = LandslideDataset(val_img_dir,   val_mask_dir,   val_spec_dir,   val_transform)

    # 构造正/负样本采样权重
    mask_paths = [train_ds.mask_dir / (p.stem + '.tif') for p in train_ds.files]
    has_pos    = [(np.array(Image.open(m),dtype=np.uint8)>0).any() for m in mask_paths]
    sampler    = WeightedRandomSampler([2.0 if f else 1.0 for f in has_pos],
                                       num_samples=len(has_pos), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=NUM_WORKERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = TrueSegFormerWithHeatmap(pretrained=True).to(device)
    scaler = GradScaler()

    bce   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device))
    dice  = smp.losses.DiceLoss(mode='binary')
    focal = smp.losses.FocalLoss(mode='binary', alpha=0.8)
    def criterion(preds, targets):
        return (LOSS_WEIGHTS['bce']*bce(preds, targets)
              +LOSS_WEIGHTS['dice']*dice(preds, targets)
              +LOSS_WEIGHTS['focal']*focal(preds, targets))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )

    history, best_loss = [], float('inf')
    for epoch in range(1, NUM_EPOCHS+1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, met = validate(model, val_loader, criterion, device, scaler)
        scheduler.step()

        history.append({'epoch':epoch, 'train_loss':tr_loss, 'val_loss':val_loss, **met})
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{MODEL_NAME}_best.pth'))

        pd.DataFrame(history).to_excel(
            os.path.join(save_dir, f'training_history_{MODEL_NAME}.xlsx'), index=False
        )
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
        print(f"Loss: Train {tr_loss:.4f}  Val {val_loss:.4f}")
        visualize_samples(model, val_ds, device, viz_dir, n_samples=4, epoch=epoch)

    # 绘制并保存训练曲线
    df = pd.DataFrame(history)
    plt.figure(); plt.plot(df.epoch, df.train_loss, label='Train Loss')
    plt.plot(df.epoch, df.val_loss, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir,'loss_curve.png'), dpi=300); plt.close()

    plt.figure()
    for m in ['precision','recall','f1_score','iou','pixel_acc']:
        plt.plot(df.epoch, df[m], label=m)
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir,'metrics_curve.png'), dpi=300); plt.close()


if __name__ == '__main__':
    main()
