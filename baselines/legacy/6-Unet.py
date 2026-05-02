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
import segmentation_models_pytorch as smp
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ======== Dataset ========
class LandslideDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.files = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        ])
        if not self.files:
            raise RuntimeError(f"No images in {img_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        mask_path = self.mask_dir / (img_path.stem + '.tif')

        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        mask = (mask > 0).astype(np.uint8)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()
        else:
            img = transforms.ToTensor()(img)
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return img, mask

# ======== Metrics ========
def compute_batch_metrics(preds, masks, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()

    tp = (preds_bin * masks).sum(dim=(1, 2, 3))
    fp = (preds_bin * (1 - masks)).sum(dim=(1, 2, 3))
    fn = ((1 - preds_bin) * masks).sum(dim=(1, 2, 3))
    tn = ((1 - preds_bin) * (1 - masks)).sum(dim=(1, 2, 3))

    pixel_acc = ((tp + tn) / (tp + tn + fp + fn + eps)).mean().item()
    precision = (tp / (tp + fp + eps)).mean().item()
    recall = (tp / (tp + fn + eps)).mean().item()
    f1 = (2 * tp / (2 * tp + fp + fn + eps)).mean().item()
    iou = (tp / (tp + fp + fn + eps)).mean().item()

    return {
        'pixel_acc': pixel_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'iou': iou
    }

# ======== Training & Validation ========
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc='Train', leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    metrics_sum = {k: 0 for k in ['pixel_acc', 'precision', 'recall', 'f1_score', 'iou']}
    for imgs, masks in tqdm(loader, desc='Val', leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        total_loss += criterion(preds, masks).item() * imgs.size(0)
        batch_m = compute_batch_metrics(preds, masks)
        for k, v in batch_m.items():
            metrics_sum[k] += v * imgs.size(0)
    avg_loss = total_loss / len(loader.dataset)
    avg_metrics = {k: metrics_sum[k] / len(loader.dataset) for k in metrics_sum}
    return avg_loss, avg_metrics

# ======== Visualization ========
def visualize_samples(model, dataset, device, out_dir, n_samples=4, epoch=0):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    indices = random.sample(range(len(dataset)), n_samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))
    for i, idx in enumerate(indices):
        img, mask = dataset[idx]
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * std + mean)
        img_np = np.clip(img_np, 0, 1)

        mask_np = mask[0].cpu().numpy()
        pred_tensor = torch.sigmoid(model(img.unsqueeze(0).to(device))).detach().cpu().squeeze().numpy()
        pred_bin = (pred_tensor > 0.5).astype(np.uint8)

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title('Image')
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 2].imshow(pred_bin, cmap='gray')
        axes[i, 2].set_title('Prediction')
        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'viz_epoch_{epoch:03d}.png'), dpi=300)
    plt.close(fig)

# ======== Main Script ========
def main():
    # Paths
    base = '/home/featurize/work/newSEG'
    train_img_dir = os.path.join(base, 'split_dataset/Train_img')
    train_mask_dir = os.path.join(base, 'split_dataset/Train_label')
    val_img_dir = os.path.join(base, 'split_dataset/Test_img')
    val_mask_dir = os.path.join(base, 'split_dataset/Test_label')
    model_name = 'deeplabv3plus'
    save_dir = os.path.join(base, f'Results_{model_name}')
    ckpt_dir = os.path.join(save_dir, f'{model_name}_checkpoints')
    viz_dir = os.path.join(save_dir, f'{model_name}_viz')
    plots_dir = os.path.join(save_dir, f'{model_name}_plots')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Augmentations
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(p=0.5), A.GaussNoise(p=0.2),
        A.Normalize(), ToTensorV2()
    ])
    val_transform = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])

    # Datasets & Loaders
    train_ds = LandslideDataset(train_img_dir, train_mask_dir, train_transform)
    val_ds = LandslideDataset(val_img_dir, val_mask_dir, val_transform)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.DeepLabV3Plus(
        encoder_name='tu-convnext_tiny', encoder_weights='imagenet', in_channels=3, classes=1
    ).to(device)
    criterion = lambda p, t: nn.BCEWithLogitsLoss()(p, t) + smp.losses.DiceLoss(mode='binary')(p, t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Training Loop
    history = []
    best_loss = float('inf')
    for epoch in range(1, 201):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({'epoch': epoch, 'train_loss': tr_loss, 'val_loss': val_loss, **metrics})

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(ckpt_dir, f'{model_name}_best_model.pth')
            )

        print(f"Epoch [{epoch}/200]  Train Loss: {tr_loss:.4f}  Val Loss: {val_loss:.4f}  IoU: {metrics['iou']:.4f}")

        visualize_samples(model, val_ds, device, viz_dir, epoch=epoch)

    # Save metrics log
    df = pd.DataFrame(history)
    df.to_excel(os.path.join(save_dir, f'training_history_{model_name}.xlsx'), index=False)

    # Plot curves
    plt.figure()
    plt.plot(df.epoch, df.train_loss, label='Train Loss')
    plt.plot(df.epoch, df.val_loss,   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'loss_curve.png'), dpi=300); plt.close()

    plt.figure()
    for metric in ['iou', 'pixel_acc', 'precision', 'recall', 'f1_score']:
        plt.plot(df.epoch, df[metric], label=metric)
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'metrics_curve.png'), dpi=300); plt.close()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
