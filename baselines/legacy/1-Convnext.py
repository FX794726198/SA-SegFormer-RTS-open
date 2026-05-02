import os
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

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
    def __init__(self, root_dir, transform=None):
        """
        root_dir: 同时包含 .jpg 图像和同名 .json 标签的目录
        transform: albumentations 组合增强（同时作用于图像和 mask）
        """
        self.root_dir = root_dir
        self.img_files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith('.jpg')])
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 加载图像
        img_name = self.img_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))

        # 加载 JSON 并生成二值 mask
        json_path = os.path.join(self.root_dir, img_name.replace('.jpg', '.json'))
        with open(json_path, 'r') as f:
            label_json = json.load(f)
        mask = self.json_to_mask(label_json, image.shape[:2])

        # 同时增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        else:
            # 转为 tensor
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(mask).long()

        return image, mask.unsqueeze(0).float()

    @staticmethod
    def json_to_mask(label_json, img_size):
        """
        根据 JSON 中的多边形生成二值 mask
        假定 JSON 结构为 { "shapes": [ { "points": [[x1,y1],…] }, … ] }
        """
        h, w = img_size
        mask_img = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask_img)
        for shape in label_json.get('shapes', []):
            points = shape['points']
            # PIL 接受 [(x,y),…]
            polygon = [tuple(p) for p in points]
            draw.polygon(polygon, outline=1, fill=1)
        return np.array(mask_img, dtype=np.uint8)


# ======== 准备数据 ========
train_dir = '/Users/fengxiao/Desktop/Work/语义分割/Train'
test_dir  = '/Users/fengxiao/Desktop/Work/语义分割/Test'

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Normalize(),
    ToTensorV2()
])

train_dataset = LandslideDataset(train_dir, transform=train_transform)
val_dataset   = LandslideDataset(test_dir, transform=val_transform)  # 可用测试集作验证

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)


# ======== 构建模型 ========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = smp.Unet(
    encoder_name='convnext_tiny',        # ConvNeXt Tiny 骨干
    encoder_weights='imagenet',          # 预训练权重
    in_channels=3,                       # RGB
    classes=1,                           # 二分类输出
    activation=None                      # 后续用 BCEWithLogitsLoss
)
model.to(device)


# ======== 损失函数、优化器、调度器 ========
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# ======== 训练与验证函数 ========
def iou_score(preds, masks, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * masks).sum(dim=(1,2,3))
    union = (preds + masks - preds * masks).sum(dim=(1,2,3))
    return ((intersection + eps) / (union + eps)).mean().item()

def train_one_epoch(model, loader, optimizer):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(loader, desc='Train', leave=False):
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Val  ', leave=False):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)
            running_loss += loss.item() * images.size(0)
            running_iou  += iou_score(preds, masks) * images.size(0)
    return running_loss / len(loader.dataset), running_iou / len(loader.dataset)


# ======== 主训练循环 ========
num_epochs = 200
history = []

best_val_loss = float('inf')
save_dir = './checkpoints'
os.makedirs(save_dir, exist_ok=True)

for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_iou = validate(model, val_loader)
    scheduler.step()

    print(f"Epoch [{epoch}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} "
          f"Val Loss: {val_loss:.4f} "
          f"Val IoU: {val_iou:.4f}")

    history.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_iou': val_iou
    })

    # 保存最优模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        print("  → Saved best_model.pth")

# 训练结束，保存完整模型
torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pth'))
print("Training complete. Models saved.")


# ======== 保存训练指标 ========
df = pd.DataFrame(history)
df.to_csv(os.path.join(save_dir, 'training_log.csv'), index=False)
print("Training log saved to training_log.csv")


# ======== 在测试集上评估 ========
# 假设 test_loader 已定义
test_loader = DataLoader(LandslideDataset(test_dir, transform=val_transform),
                         batch_size=16, shuffle=False, num_workers=4)
from tqdm import tqdm

# ======== 定义评估指标 ========
def compute_metrics(preds, masks, threshold=0.5, eps=1e-6):
    """
    输入：
      preds: 模型原始输出 logits，形状 [B,1,H,W]
      masks: 二值真值 mask，形状 [B,1,H,W]
    返回（torch scalar）：
      dict 包含 TP, FP, FN, TN, PixelAcc, Precision, Recall, F1 (Dice), IoU
    """
    # 二值化预测
    probs = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()

    # 统计
    tp = (preds_bin * masks).sum(dim=(1,2,3))
    fp = (preds_bin * (1 - masks)).sum(dim=(1,2,3))
    fn = ((1 - preds_bin) * masks).sum(dim=(1,2,3))
    tn = ((1 - preds_bin) * (1 - masks)).sum(dim=(1,2,3))

    # 累加各样本指标
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        'tp': tp.sum().item(),
        'fp': fp.sum().item(),
        'fn': fn.sum().item(),
        'tn': tn.sum().item(),
        'pixel_acc': pixel_acc.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.mean().item(),
        'iou': iou.mean().item(),
    }

# ======== 在测试集上评估更多指标 ========
# 加载最佳模型
model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
model.to(device)
model.eval()

# 累积全局计数
global_stats = {
    'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
    'pixel_acc': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0
}
num_batches = 0

with torch.no_grad():
    for images, masks in tqdm(test_loader, desc='Test Eval'):
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        batch_stats = compute_metrics(preds, masks)

        # 累加
        for k, v in batch_stats.items():
            global_stats[k] += v
        num_batches += 1

# 取平均
metrics = {k: (v / num_batches) if k not in ('tp','fp','fn','tn') else v
           for k, v in global_stats.items()}

# 打印结果
print("=== Test Set Metrics ===")
print(f"Pixel Accuracy : {metrics['pixel_acc']:.4f}")
print(f"Precision      : {metrics['precision']:.4f}")
print(f"Recall         : {metrics['recall']:.4f}")
print(f"F1 (Dice)      : {metrics['f1']:.4f}")
print(f"IoU (Jaccard)  : {metrics['iou']:.4f}")
print(f"TP={metrics['tp']:.0f}, FP={metrics['fp']:.0f}, FN={metrics['fn']:.0f}, TN={metrics['tn']:.0f}")

# 可选：保存到 CSV 便于后续分析
import pandas as pd
df_test = pd.DataFrame([metrics])
df_test.to_csv(os.path.join(save_dir, 'test_metrics.csv'), index=False)
print("Test metrics saved to test_metrics.csv")


# 加载最佳模型
model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
model.to(device)

test_loss, test_iou = validate(model, test_loader)
print(f"Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}")
