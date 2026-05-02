import os, random
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ======== Dataset for Segmentation ========
class LandslideDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.files = sorted([p for p in self.img_dir.iterdir()
                             if p.suffix.lower() in ('.jpg','.jpeg','.png','.tif','.tiff')])
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
            if mask.ndim == 2: mask = mask.unsqueeze(0)
            mask = mask.float()
        else:
            img = transforms.ToTensor()(img)
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask

# ======== Dataset for Detection ========
class LandslideDetectionDataset(LandslideDataset):
    def __getitem__(self, idx):
        img, mask = super().__getitem__(idx)
        # compute bounding box from mask
        pos = torch.nonzero(mask[0])
        if pos.numel() == 0:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            y_min = pos[:,0].min().item()
            y_max = pos[:,0].max().item()
            x_min = pos[:,1].min().item()
            x_max = pos[:,1].max().item()
            boxes = torch.tensor([[x_min,y_min,x_max,y_max]], dtype=torch.float32)
            labels = torch.ones((1,), dtype=torch.int64)
        target = { 'boxes': boxes, 'labels': labels }
        return img, target

# ======== Metrics for Segmentation ========
def compute_batch_metrics(preds, masks, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()
    tp = (preds_bin * masks).sum((1,2,3))
    fp = (preds_bin * (1-masks)).sum((1,2,3))
    fn = ((1-preds_bin) * masks).sum((1,2,3))
    tn = ((1-preds_bin)*(1-masks)).sum((1,2,3))
    pixel_acc = ((tp+tn)/(tp+tn+fp+fn+eps)).mean().item()
    precision = (tp/(tp+fp+eps)).mean().item()
    recall = (tp/(tp+fn+eps)).mean().item()
    f1 = (2*tp/(2*tp+fp+fn+eps)).mean().item()
    iou = (tp/(tp+fp+fn+eps)).mean().item()
    return {'pixel_acc':pixel_acc,'precision':precision,'recall':recall,'f1_score':f1,'iou':iou}

# ======== Training Loops ========
def train_one_epoch_seg(model, loader, optimizer, criterion, device):
    model.train(); total_loss=0
    for imgs, masks in tqdm(loader, desc='Train', leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()*imgs.size(0)
    return total_loss/len(loader.dataset)

@torch.no_grad()
def validate_seg(model, loader, criterion, device):
    model.eval(); total_loss=0; metrics_sum={'pixel_acc':0,'precision':0,'recall':0,'f1_score':0,'iou':0}
    for imgs,masks in tqdm(loader, desc='Val', leave=False):
        imgs,masks=imgs.to(device),masks.to(device)
        preds=model(imgs); total_loss+=criterion(preds,masks).item()*imgs.size(0)
        batch_m=compute_batch_metrics(preds,masks)
        for k,v in batch_m.items(): metrics_sum[k]+=v*imgs.size(0)
    avg_loss=total_loss/len(loader.dataset)
    avg_m={k:metrics_sum[k]/len(loader.dataset) for k in metrics_sum}
    return avg_loss, avg_m

# Detection train/val

def train_one_epoch_det(model, loader, optimizer, device):
    model.train(); total_loss=0
    for imgs, targets in tqdm(loader, desc='TrainDet', leave=False):
        imgs=[im.to(device) for im in imgs]
        targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()*len(imgs)
    return total_loss/len(loader.dataset)

@torch.no_grad()
def validate_det(model, loader, device):
    model.eval(); total_loss=0; iou_sum=0; count=0
    for imgs, targets in tqdm(loader, desc='ValDet', leave=False):
        imgs=[im.to(device) for im in imgs]
        targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
        loss_dict=model(imgs, targets)
        total_loss+=sum(v.item() for v in loss_dict.values())*len(imgs)
        # compute IoU per box
        outputs=model(imgs)
        for t,o in zip(targets,outputs):
            if t['boxes'].numel()>0 and o['boxes'].numel()>0:
                iou_matrix=box_iou(t['boxes'],o['boxes'])
                iou_sum+=iou_matrix.diag().mean().item()
                count+=1
    avg_loss=total_loss/len(loader.dataset)
    avg_iou=(iou_sum/count if count>0 else 0)
    return avg_loss, {'det_iou':avg_iou}

# ======== Visualization ========
def visualize_samples(model, dataset, device, out_dir, n_samples=3, epoch=0, seg=True):
    os.makedirs(out_dir, exist_ok=True)
    mean, std = np.array([0.485,0.456,0.406]), np.array([0.229,0.224,0.225])
    idxs=random.sample(range(len(dataset)), n_samples)
    fig,axes=plt.subplots(n_samples,3,figsize=(9,3*n_samples))
    for i,idx in enumerate(idxs):
        img,mask=dataset[idx]
        img_np=img.cpu().numpy().transpose(1,2,0)*std+mean; img_np=np.clip(img_np,0,1)
        axes[i,0].imshow(img_np); axes[i,0].set_title('Image')
        axes[i,1].imshow(mask[0].cpu().numpy(),cmap='gray'); axes[i,1].set_title('GT')
        if seg:
            pred=(torch.sigmoid(model(img.unsqueeze(0).to(device))).detach().cpu().squeeze().numpy()>0.5).astype(np.uint8)
        else:
            out=model([img.to(device)])[0]
            boxes=out['boxes'].detach().cpu().numpy().astype(int)
            pred = img_np.copy()
            for b in boxes: plt.gca().add_patch(plt.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],
                edgecolor='r',facecolor='none',linewidth=2))
        axes[i,2].imshow(pred,cmap='gray' if seg else None); axes[i,2].set_title('Pred')
        for ax in axes[i]: ax.axis('off')
    plt.tight_layout(); plt.savefig(f"{out_dir}/{epoch:03d}.png",dpi=300); plt.close()

# ======== Main ========
def main():
    base='/home/featurize/work/newSEG'
    # paths
    timg,tlbl=f"{base}/split_dataset/Train_img",f"{base}/split_dataset/Train_label"
    vimg,vlbl=f"{base}/split_dataset/Test_img",f"{base}/split_dataset/Test_label"
    save=f"{base}/Results"; os.makedirs(save,exist_ok=True)

    # transforms
    train_tf=A.Compose([A.RandomResizedCrop((256,256),scale=(0.8,1),ratio=(0.9,1.1)),A.HorizontalFlip(),A.Normalize(),ToTensorV2()])
    val_tf=A.Compose([A.Resize(height=256, width=256),A.Normalize(),ToTensorV2()])

    # loaders
    seg_train_ds=LandslideDataset(timg,tlbl,train_tf)
    seg_val_ds  =LandslideDataset(vimg,vlbl,val_tf)
    det_train_ds=LandslideDetectionDataset(timg,tlbl,train_tf)
    det_val_ds  =LandslideDetectionDataset(vimg,vlbl,val_tf)
    seg_tr=DataLoader(seg_train_ds,16,shuffle=True,num_workers=4,pin_memory=True)
    seg_val=DataLoader(seg_val_ds,16,shuffle=False,num_workers=4,pin_memory=True)
    det_tr=DataLoader(det_train_ds,8,shuffle=True,collate_fn=lambda b: list(zip(*b)),num_workers=4)
    det_val=DataLoader(det_val_ds,8,shuffle=False,collate_fn=lambda b: list(zip(*b)),num_workers=4)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model zoo
    models = {}
    optim = {}
    sched = {}
    hist = {}
        # seg models
        # 定义各分割模型
    seg_model_defs = [
        ('convnext',  smp.Unet('tu-convnext_tiny', encoder_weights='imagenet', in_channels=3, classes=1)),
        ('unet',      smp.Unet('resnet34', encoder_weights='imagenet', in_channels=3, classes=1)),
        ('unet++',    smp.UnetPlusPlus('resnet34', encoder_weights='imagenet', in_channels=3, classes=1)),
        ('att_unet',  smp.Unet('resnet34', encoder_weights='imagenet', in_channels=3, classes=1, decoder_attention_type='scse')),
        ('pspnet',    smp.PSPNet('resnet34', encoder_weights='imagenet', in_channels=3, classes=1)),
        ('pan',       smp.PAN('resnet34', encoder_weights='imagenet', in_channels=3, classes=1)),
        ('hrnet',     smp.Unet('hrnet_w18', encoder_weights='imagenet', in_channels=3, classes=1)),
        ('segformer', smp.SegFormer('MiT-B0', encoder_weights='imagenet', in_channels=3, classes=1)),
        ('deeplabv3+',smp.DeepLabV3Plus('mobilenet_v2', encoder_weights='imagenet', in_channels=3, classes=1)),
        ('swinunet',  smp.Unet('swin_tiny_patch4_window7_224', encoder_weights='imagenet', in_channels=3, classes=1))
    ]
    for name, net in seg_model_defs:
        models[name] = net.to(device)
        optim[name]  = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-5)
        sched[name]  = torch.optim.lr_scheduler.CosineAnnealingLR(optim[name], T_max=200)
        hist[name]   = []

    # detection model
    models['fasterrcnn']=fasterrcnn_resnet50_fpn(pretrained=True,num_classes=2).to(device)
    optim['fasterrcnn']=torch.optim.AdamW(models['fasterrcnn'].parameters(),1e-4)
    sched['fasterrcnn']=torch.optim.lr_scheduler.StepLR(optim['fasterrcnn'],step_size=20,gamma=0.1)
    hist['fasterrcnn']=[]

    # loss for seg
    crit=lambda p,t: nn.BCEWithLogitsLoss()(p,t)+smp.losses.DiceLoss('binary')(p,t)
    EPOCHS=50
    for epoch in range(1,EPOCHS+1):
        print(f"Epoch {epoch}/{EPOCHS}")
        # segmentation
        for name in ['convnext','unet','deeplabv3+','segformer','swinunet']:
            trL=train_one_epoch_seg(models[name],seg_tr,optim[name],crit,device)
            vL,met=validate_seg(models[name],seg_val,crit,device)
            sched[name].step()
            hist[name].append({'epoch':epoch,'train_loss':trL,'val_loss':vL,**met})
            print(f"[{name}] tr {trL:.3f} vl {vL:.3f} iou {met['iou']:.3f}")
            visualize_samples(models[name],seg_val_ds,device,os.path.join(save,'viz',name),4,epoch,seg=True)
        # detection
        trLd=train_one_epoch_det(models['fasterrcnn'],det_tr,optim['fasterrcnn'],device)
        vLd,det_m=validate_det(models['fasterrcnn'],det_val,device)
        sched['fasterrcnn'].step()
        hist['fasterrcnn'].append({'epoch':epoch,'train_loss':trLd,'val_loss':vLd,**det_m})
        print(f"[fasterrcnn] tr {trLd:.3f} vl {vLd:.3f} det_iou {det_m['det_iou']:.3f}")
        visualize_samples(models['fasterrcnn'],det_val_ds,device,os.path.join(save,'viz','fasterrcnn'),4,epoch,seg=False)

    # save results
    for name,log in hist.items():
        df=pd.DataFrame(log)
        df.to_excel(os.path.join(save,f'history_{name}.xlsx'),index=False)
        plt.figure(); plt.plot(df.epoch,df.train_loss,label='tr'); plt.plot(df.epoch,df.val_loss,label='val'); plt.title(name+' loss'); plt.legend(); plt.savefig(os.path.join(save,f'loss_{name}.png'),dpi=300); plt.close()
        plt.figure();
        keys=[k for k in df.columns if k not in ['epoch','train_loss','val_loss']]
        for k in keys: plt.plot(df.epoch,df[k],label=k)
        plt.title(name+' metrics'); plt.legend(); plt.savefig(os.path.join(save,f'metrics_{name}.png'),dpi=300); plt.close()

if __name__=='__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
