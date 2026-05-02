import os
import sys
import subprocess


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
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import timm
from tqdm import tqdm

# ========== 超参数 ==========
BASE_DIR     = '/home/featurize/work/newSEG'
MODEL_NAME   = 'upernet_swin'
IMG_SIZE     = (256, 256)
BATCH_SIZE   = 16
NUM_EPOCHS   = 200
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 4
THRESHOLD    = 0.4
NUM_CLASSES  = 1   # 二分类

FPN_CH = 256       # FPN 和 PPM 输出通道
GN_GROUPS = 32     # GroupNorm 分组数

# ========== UPerNet Head with GroupNorm ==========
class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1,2,3,6), reduction=FPN_CH):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, reduction, 1, bias=False),
                nn.GroupNorm(GN_GROUPS, reduction),
                nn.ReLU(inplace=True)
            ) for ps in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes)*reduction, reduction, 3, padding=1, bias=False),
            nn.GroupNorm(GN_GROUPS, reduction),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        priors = [x]
        for stage in self.stages:
            y = stage(x)
            y = nn.functional.interpolate(y, size=(h,w), mode='bilinear', align_corners=False)
            priors.append(y)
        return self.bottleneck(torch.cat(priors, dim=1))

class UPerNetHead(nn.Module):
    def __init__(self, in_chs, fpn_ch=FPN_CH, num_classes=1):
        super().__init__()
        self.ppm = PyramidPoolingModule(in_chs[-1], reduction=fpn_ch)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, fpn_ch, 1) for c in in_chs[:-1]
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1, bias=False),
                nn.GroupNorm(GN_GROUPS, fpn_ch),
                nn.ReLU(inplace=True)
            ) for _ in in_chs[:-1]
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(len(in_chs[:-1])*fpn_ch + fpn_ch, fpn_ch, 3, padding=1, bias=False),
            nn.GroupNorm(GN_GROUPS, fpn_ch),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(fpn_ch, num_classes, 1)

    def forward(self, feats):
        p2, p3, p4, p5 = feats
        ppm_out  = self.ppm(p5)
        laterals = [l(p) for l,p in zip(self.lateral_convs, feats[:-1])]

        td = nn.functional.interpolate(ppm_out, size=laterals[-1].shape[2:], mode='bilinear', align_corners=False)
        fusion = []
        for lat, fpn in zip(reversed(laterals), reversed(self.fpn_convs)):
            td = nn.functional.interpolate(td, size=lat.shape[2:], mode='bilinear', align_corners=False) + lat
            fusion.append(fpn(td))
        fusion = list(reversed(fusion))

        size_p2 = fusion[0].shape[2:]
        ups = [fusion[0]] + [
            nn.functional.interpolate(f, size=size_p2, mode='bilinear', align_corners=False)
            for f in fusion[1:]
        ] + [
            nn.functional.interpolate(ppm_out, size=size_p2, mode='bilinear', align_corners=False)
        ]
        cat = torch.cat(ups, dim=1)
        return self.classifier(self.fuse(cat))

class UPerNetModel(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=True,
            features_only=True,
            out_indices=(0,1,2,3),
            img_size=IMG_SIZE
        )
        with torch.no_grad():
            dummy = torch.zeros(1,3,*IMG_SIZE)
            feats = self.backbone(dummy)
        chs = [f.shape[1] for f in feats]
        self.decoder = UPerNetHead(chs, fpn_ch=FPN_CH, num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        out   = self.decoder(feats)
        return nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

# ========== Dataset ==========
class LandslideDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir  = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.files = sorted(
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in ('.jpg','.jpeg','.png','.tif','.tiff')
        )
        if not self.files:
            raise RuntimeError(f"No images in {img_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_p = self.files[i]
        img = np.array(Image.open(img_p).convert("RGB"))
        mask = (np.array(Image.open(self.mask_dir / (img_p.stem + '.tif'))) > 0).astype(np.uint8)
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
        else:
            img = TF.to_tensor(img)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask

# ========== 训练/验证/指标/可视化 ==========
def compute_batch_metrics(preds, masks, threshold=THRESHOLD, eps=1e-6):
    probs     = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()
    masks     = masks.squeeze(1)
    tp = (preds_bin * masks).sum((1,2))
    fp = (preds_bin * (1-masks)).sum((1,2))
    fn = ((1-preds_bin)*masks).sum((1,2))
    tn = ((1-preds_bin)*(1-masks)).sum((1,2))
    precision_pos = (tp/(tp+fp+eps)).mean().item()
    recall_pos    = (tp/(tp+fn+eps)).mean().item()
    f1_pos        = (2*tp/(2*tp+fp+fn+eps)).mean().item()
    iou_pos       = (tp/(tp+fp+fn+eps)).mean().item()
    acc_pos       = recall_pos
    precision_neg = (tn/(tn+fn+eps)).mean().item()
    recall_neg    = (tn/(tn+fp+eps)).mean().item()
    f1_neg        = (2*tn/(2*tn+fn+fp+eps)).mean().item()
    iou_neg       = (tn/(tn+fn+fp+eps)).mean().item()
    acc_neg       = recall_neg
    precision     = 0.5*(precision_pos+precision_neg)
    recall        = 0.5*(recall_pos+recall_neg)
    f1_score      = 0.5*(f1_pos+f1_neg)
    iou           = 0.5*(iou_pos+iou_neg)
    pixel_acc     = ((tp+tn)/(tp+tn+fp+fn+eps)).mean().item()
    return {
        'precision_pos':precision_pos,'recall_pos':recall_pos,'f1_pos':f1_pos,'iou_pos':iou_pos,'acc_pos':acc_pos,
        'precision_neg':precision_neg,'recall_neg':recall_neg,'f1_neg':f1_neg,'iou_neg':iou_neg,'acc_neg':acc_neg,
        'precision':precision,'recall':recall,'f1_score':f1_score,'iou':iou,'pixel_acc':pixel_acc
    }

def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc='Train'):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss  = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    metrics_sum = {k:0 for k in compute_batch_metrics(torch.zeros(1,1,*IMG_SIZE), torch.zeros(1,1,*IMG_SIZE))}
    for imgs, masks in tqdm(loader, desc='Val'):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss  = criterion(preds, masks)
        total_loss += loss.item() * imgs.size(0)
        bm = compute_batch_metrics(preds, masks)
        for k,v in bm.items():
            metrics_sum[k] += v * imgs.size(0)
    avg_loss = total_loss / len(loader.dataset)
    avg_met  = {k:metrics_sum[k]/len(loader.dataset) for k in metrics_sum}
    return avg_loss, avg_met

def visualize_samples(model, dataset, device, out_dir, n_samples=4, epoch=0):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    mean,std = np.array([0.485,0.456,0.406]), np.array([0.229,0.224,0.225])
    idxs = np.random.choice(len(dataset), n_samples, replace=False)
    fig, axes = plt.subplots(n_samples,4,figsize=(12,3*n_samples))
    for i, idx in enumerate(idxs):
        img, mask = dataset[idx]
        img_np = (img.cpu().numpy().transpose(1,2,0)*std + mean).clip(0,1)
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))
        prob   = torch.sigmoid(pred).cpu().squeeze().numpy()
        bin_map= (prob>THRESHOLD).astype(np.uint8)
        gt     = mask.squeeze(0).numpy()

        axes[i,0].imshow(img_np);        axes[i,0].set_title('Image'); axes[i,0].axis('off')
        axes[i,1].imshow(gt, cmap='gray'); axes[i,1].set_title('GT');   axes[i,1].axis('off')
        axes[i,2].imshow(bin_map, cmap='gray'); axes[i,2].set_title('Pred'); axes[i,2].axis('off')
        im3 = axes[i,3].imshow(prob, cmap='plasma'); axes[i,3].set_title('Prob'); axes[i,3].axis('off')
        fig.colorbar(im3, ax=axes[i,3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'viz_epoch_{epoch:03d}.png'), dpi=300)
    plt.close(fig)

def main():
    train_img_dir  = os.path.join(BASE_DIR, 'split_dataset/Train_img')
    train_mask_dir = os.path.join(BASE_DIR, 'split_dataset/Train_label')
    val_img_dir    = os.path.join(BASE_DIR, 'split_dataset/Test_img')
    val_mask_dir   = os.path.join(BASE_DIR, 'split_dataset/Test_label')

    save_dir  = os.path.join(BASE_DIR, f'Results_{MODEL_NAME}')
    ckpt_dir  = os.path.join(save_dir, f'{MODEL_NAME}_checkpoints')
    viz_dir   = os.path.join(save_dir, f'{MODEL_NAME}_viz')
    plots_dir = os.path.join(save_dir, f'{MODEL_NAME}_plots')
    for d in (ckpt_dir, viz_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    train_ds = LandslideDataset(train_img_dir, train_mask_dir)
    val_ds   = LandslideDataset(val_img_dir,   val_mask_dir)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = UPerNetModel(NUM_CLASSES).to(device)

    bce = nn.BCEWithLogitsLoss()
    criterion = lambda p,t: bce(p,t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history, best_loss = [], float('inf')
    for epoch in range(1, NUM_EPOCHS+1):
        # 分开调用，避免 unpack 错误
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, met = validate(model, val_loader, device, criterion)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{MODEL_NAME}_best.pth'))

        history.append({'epoch':epoch,'train_loss':tr_loss,'val_loss':val_loss,**met})
        pd.DataFrame(history).to_excel(os.path.join(save_dir,f'training_history_{MODEL_NAME}.xlsx'), index=False)

        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
        print(f"Loss: Train {tr_loss:.4f}  Val {val_loss:.4f}")
        print(f"{'类别':<12}{'Precision':>10}{'Recall':>10}{'F1':>10}{'IoU':>10}{'Acc':>10}")
        print(f"{'Positive':<12}{met['precision_pos']:^10.4f}{met['recall_pos']:^10.4f}{met['f1_pos']:^10.4f}{met['iou_pos']:^10.4f}{met['acc_pos']:^10.4f}")
        print(f"{'Negative':<12}{met['precision_neg']:^10.4f}{met['recall_neg']:^10.4f}{met['f1_neg']:^10.4f}{met['iou_neg']:^10.4f}{met['acc_neg']:^10.4f}")
        print(f"{'Overall':<12}{met['precision']:^10.4f}{met['recall']:^10.4f}{met['f1_score']:^10.4f}{met['iou']:^10.4f}{met['pixel_acc']:^10.4f}")

        visualize_samples(model, val_ds, device, viz_dir, n_samples=4, epoch=epoch)

    df = pd.DataFrame(history)
    plt.figure(); plt.plot(df.epoch, df.train_loss, label='Train'); plt.plot(df.epoch, df.val_loss, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir,'loss_curve.png'), dpi=300)
    plt.figure()
    for m in ['precision','recall','f1_score','iou','pixel_acc']:
        plt.plot(df.epoch, df[m], label=m)
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir,'metrics_curve.png'), dpi=300)

if __name__=='__main__':
    main()
