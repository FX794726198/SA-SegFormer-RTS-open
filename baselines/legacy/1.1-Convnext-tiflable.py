import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
from tqdm import tqdm


# ======== 数据集定义 ========
class LandslideDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        """
        img_dir:   存放 .jpg/.jpeg/.png/.tif 影像 的目录
        mask_dir:  存放同名 .tif 二值 mask（值 0/1）的目录
        transform: albumentations 增强（同时作用于 image 和 mask）
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        valid_ext = ('.jpg', '.jpeg', '.png', '.tif')
        self.img_files = sorted(
            [f for f in os.listdir(img_dir)
             if f.lower().endswith(valid_ext)]
        )
        if len(self.img_files) == 0:
            raise RuntimeError(
                f"No images found in {img_dir!r}. "
                f"Supported extensions: {valid_ext}. "
                f"Files in directory: {os.listdir(img_dir)!r}"
            )
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # --- 读取影像 ---
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        # --- 读取同名 .tif mask ---
        mask_name = os.path.splitext(img_name)[0] + '.tif'
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        mask = (mask > 0).astype(np.uint8)  # 二值化成 0/1

        # --- 增强 & 转 Tensor，并确保 mask 变成 [1,H,W] ---
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            # 如果 mask 是 H×W，就加一个 channel 维度
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # 有时 ToTensorV2 会直接输出 float，也可能输出 long，这里统一成 float
            mask = mask.float()
        else:
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


# ======== 全局配置 ========
train_img_dir = '/Users/fengxiao/Desktop/Work/语义分割/Data/split_dataset/Train_img'
train_mask_dir = '/Users/fengxiao/Desktop/Work/语义分割/Data/split_dataset/Train_label'
test_img_dir = '/Users/fengxiao/Desktop/Work/语义分割/Data/split_dataset/Test_img'
test_mask_dir = '/Users/fengxiao/Desktop/Work/语义分割/Data/split_dataset/Test_label'

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Normalize(),
    ToTensorV2()
])

BATCH_SIZE = 16
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
T_MAX = NUM_EPOCHS
SAVE_DIR = '/Users/fengxiao/Desktop/Work/语义分割/Results/checkpoints'


# ======== 辅助函数 ========
def iou_score(preds, masks, threshold=0.5, eps=1e-6):
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    inter = (preds_bin * masks).sum(dim=(1, 2, 3))
    union = (preds_bin + masks - preds_bin * masks).sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean().item()


def compute_metrics(preds, masks, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()
    tp = (preds_bin * masks).sum(dim=(1, 2, 3))
    fp = (preds_bin * (1 - masks)).sum(dim=(1, 2, 3))
    fn = ((1 - preds_bin) * masks).sum(dim=(1, 2, 3))
    tn = ((1 - preds_bin) * (1 - masks)).sum(dim=(1, 2, 3))
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        'tp': tp.sum().item(), 'fp': fp.sum().item(),
        'fn': fn.sum().item(), 'tn': tn.sum().item(),
        'pixel_acc': pixel_acc.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.mean().item(), 'iou': iou.mean().item(),
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader, desc='Train', leave=False):
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou = 0, 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Val  ', leave=False):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)
            total_loss += loss.item() * images.size(0)
            total_iou += iou_score(preds, masks) * images.size(0)
    return total_loss / len(loader.dataset), total_iou / len(loader.dataset)


# ======== 主流程 ========
def main():
    # DataLoader
    train_ds = LandslideDataset(train_img_dir, train_mask_dir, transform=train_transform)
    val_ds = LandslideDataset(test_img_dir, test_mask_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model / Optimizer / Scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.Unet(
        encoder_name="tu-convnext_tiny",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)

    os.makedirs(SAVE_DIR, exist_ok=True)
    history, best_val_loss = [], float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch}/{NUM_EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  "
              f"Val Loss: {val_loss:.4f}  "
              f"Val IoU:  {val_iou:.4f}")

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_iou': val_iou
        })
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
            print("  → Saved best_model.pth")

    # 保存最后权重 & 日志
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'last_model.pth'))
    pd.DataFrame(history).to_csv(os.path.join(SAVE_DIR, 'training_log.csv'), index=False)

    # 测试评估
    test_loader = DataLoader(
        LandslideDataset(test_img_dir, test_mask_dir, transform=val_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth')))
    model.eval()

    global_stats = {k: 0 for k in ['tp', 'fp', 'fn', 'tn', 'pixel_acc', 'precision', 'recall', 'f1', 'iou']}
    num_batches = 0
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc='Test Eval'):
            images, masks = images.to(device), masks.to(device)
            stats = compute_metrics(model(images), masks)
            for k, v in stats.items(): global_stats[k] += v
            num_batches += 1

    metrics = {
        k: (v / num_batches if k not in ['tp', 'fp', 'fn', 'tn'] else v)
        for k, v in global_stats.items()
    }
    print("=== Test Set Metrics ===")
    print(f"PixelAcc : {metrics['pixel_acc']:.4f}  "
          f"Precision: {metrics['precision']:.4f}  "
          f"Recall   : {metrics['recall']:.4f}  "
          f"F1(Dice) : {metrics['f1']:.4f}  "
          f"IoU      : {metrics['iou']:.4f}")
    pd.DataFrame([metrics]).to_csv(os.path.join(SAVE_DIR, 'test_metrics.csv'), index=False)
    print("All done. Metrics saved.")


if __name__ == '__main__':
    # macOS/Windows 多进程安全启动
    from multiprocessing import freeze_support

    freeze_support()
    main()