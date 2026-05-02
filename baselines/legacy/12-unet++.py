import os
# 在导入 torch 之前设置可扩展分段
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

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ========== 超参数设置 ==========
BASE_DIR            = '/home/featurize/work/newSEG'
MODEL_NAME          = 'unet++'
IMG_SIZE            = (256, 256)
BATCH_SIZE          = 16
NUM_EPOCHS          = 200
LR                  = 3e-4
MAX_LR              = 1e-3
WEIGHT_DECAY        = 1e-4
NUM_WORKERS         = 4
THRESHOLD           = 0.4

POS_WEIGHT          = 5.0
LOSS_WEIGHTS        = {'bce':0.5, 'dice':1.0, 'focal':1.0}

# ========== 使用 Unet++ 模型 ==========
class UNetPlusPlusModel(nn.Module):
    def __init__(self, num_classes=1, encoder_name='resnet34', pretrained=True):
        super().__init__()
        # 使用 segmentation_models_pytorch 中的 Unet++
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=3,
            classes=num_classes
        )

    def forward(self, x):
        # 直接返回 logits
        return self.model(x)

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
            img  = ToTensorV2()(image=img)['image']
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
    mean, std = np.array([0.485,0.456,0.406]), np.array([0.229,0.224,0.225])
    idxs = np.random.choice(len(dataset), n_samples, replace=False)
    # 列数改为 4: 原图, GT, Pred, 概率图
    fig, axes = plt.subplots(n_samples, 4, figsize=(12, 3*n_samples))

    for i, idx in enumerate(idxs):
        img, mask = dataset[idx]
        img_np = (img.cpu().numpy().transpose(1,2,0)*std + mean).clip(0,1)
        img_t   = img.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_t)
        prob    = torch.sigmoid(pred).cpu().squeeze().numpy()

        # 0: 原图
        axes[i,0].imshow(img_np)
        axes[i,0].set_title('Image')
        axes[i,0].axis('off')
        # 1: GT 二值图
        axes[i,1].imshow(mask[0].cpu(), cmap='gray')
        axes[i,1].set_title('GT')
        axes[i,1].axis('off')
        # 2: Pred 二值图
        axes[i,2].imshow((prob>THRESHOLD).astype(np.uint8), cmap='gray')
        axes[i,2].set_title('Pred')
        axes[i,2].axis('off')
        # 3: Prob 概率图
        im3 = axes[i,3].imshow(prob, cmap='plasma')
        axes[i,3].set_title('Prob')
        axes[i,3].axis('off')

        fig.colorbar(im3, ax=axes[i,3], fraction=0.046, pad=0.04)

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

    train_ds = LandslideDataset(train_img_dir, train_mask_dir, train_transform)
    val_ds   = LandslideDataset(val_img_dir,   val_mask_dir,   val_transform)

    mask_paths = [train_ds.mask_dir / (p.stem + '.tif') for p in train_ds.files]
    has_pos    = [ (np.array(Image.open(m),dtype=np.uint8)>0).any() for m in mask_paths ]
    sampler    = WeightedRandomSampler([2.0 if f else 1.0 for f in has_pos],
                                      num_samples=len(has_pos),
                                      replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = UNetPlusPlusModel(pretrained=True).to(device)

    bce   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device))
    dice  = smp.losses.DiceLoss(mode='binary')
    focal = smp.losses.FocalLoss(mode='binary', alpha=0.8)
    def criterion(preds, targets):
        return (
            LOSS_WEIGHTS['bce']   * bce(preds, targets) +
            LOSS_WEIGHTS['dice']  * dice(preds, targets) +
            LOSS_WEIGHTS['focal'] * focal(preds, targets)
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR,
        epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='linear'
    )

    history, best_loss = [], float('inf')
    for epoch in range(1, NUM_EPOCHS+1):
        tr_loss    = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, met = validate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({'epoch': epoch, 'train_loss': tr_loss, 'val_loss': val_loss, **met})

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{MODEL_NAME}_best.pth'))

        pd.DataFrame(history).to_excel(
            os.path.join(save_dir, f'training_history_{MODEL_NAME}.xlsx'),
            index=False
        )

        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
        print(f"Loss: Train {tr_loss:.4f}  Val {val_loss:.4f}")
        print(f"{'类别':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10} {'Acc':>10}")
        print(f"{'Positive':<12} {met['precision_pos']:^10.4f} {met['recall_pos']:^10.4f} {met['f1_pos']:^10.4f} {met['iou_pos']:^10.4f} {met['acc_pos']:^10.4f}")
        print(f"{'Negative':<12} {met['precision_neg']:^10.4f} {met['recall_neg']:^10.4f} {met['f1_neg']:^10.4f} {met['iou_neg']:^10.4f} {met['acc_neg']:^10.4f}")
        print(f"{'Overall':<12} {met['precision']:^10.4f} {met['recall']:^10.4f} {met['f1_score']:^10.4f} {met['iou']:^10.4f} {met['pixel_acc']:^10.4f}")

        visualize_samples(model, val_ds, device, viz_dir, n_samples=4, epoch=epoch)

    # 一次性画出整体训练曲线
    df = pd.DataFrame(history)
    plt.figure(); plt.plot(df.epoch, df.train_loss, label='Train Loss'); plt.plot(df.epoch, df.val_loss, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir,'loss_curve.png'), dpi=300); plt.close()

    plt.figure()
    for m in ['precision','recall','f1_score','iou','pixel_acc']:
        plt.plot(df.epoch, df[m], label=m)
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir,'metrics_curve.png'), dpi=300); plt.close()

if __name__ == '__main__':
    main()
