import os
import shutil
import random
from pathlib import Path


def split_dataset(
        img_dir: str,
        label_dir: str,
        out_base_dir: str,
        train_ratio: float = 0.8,
        seed: int = 42
):
    """
    将 img_dir 和 label_dir 中同名文件（按文件名 stem 对应）按 train_ratio 随机拆分，
    并复制到 out_base_dir 下的四个子目录：
      - Train_img
      - Train_label
      - Test_img
      - Test_label
    """
    random.seed(seed)

    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    out_base = Path(out_base_dir)

    # 准备输出目录
    train_img_dir = out_base / "Train_img"
    train_label_dir = out_base / "Train_label"
    test_img_dir = out_base / "Test_img"
    test_label_dir = out_base / "Test_label"
    for d in (train_img_dir, train_label_dir, test_img_dir, test_label_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 收集所有图片文件（根据需要可扩展后缀列表）
    valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in valid_exts]
    if not img_files:
        raise RuntimeError(f"No image files found in {img_dir}")

    # 过滤出同时存在标签的图片
    paired = []
    for img_path in img_files:
        stem = img_path.stem
        # 假设标签文件也以同样的 stem 命名，但可能只支持 tif
        # 你可以按需修改后缀或增加更多后缀检查
        for lab_ext in ('.tif', '.png', '.jpg', '.jpeg'):
            lab_path = label_dir / (stem + lab_ext)
            if lab_path.exists():
                paired.append((img_path, lab_path))
                break
        else:
            print(f"WARNING: 找不到 {stem} 的 label 文件，跳过此样本。")

    if not paired:
        raise RuntimeError("No matching image–label pairs found.")

    # 随机打乱并划分
    random.shuffle(paired)
    n_train = int(len(paired) * train_ratio)
    train_pairs = paired[:n_train]
    test_pairs = paired[n_train:]

    # 复制文件
    def copy_pairs(pairs, dst_img_dir, dst_lab_dir):
        for img_path, lab_path in pairs:
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            shutil.copy2(lab_path, dst_lab_dir / lab_path.name)

    copy_pairs(train_pairs, train_img_dir, train_label_dir)
    copy_pairs(test_pairs, test_img_dir, test_label_dir)

    print(f"总样本数：{len(paired)}，训练集：{len(train_pairs)}，测试集：{len(test_pairs)}")
    print(f"保存到：{out_base.resolve()} 下的 Train_*/Test_* 文件夹。")


if __name__ == "__main__":
    split_dataset(
        img_dir="/Users/fengxiao/Desktop/img",
        label_dir="/Users/fengxiao/Desktop/四个区域合并二值化labels",
        out_base_dir="/Users/fengxiao/Desktop/split_dataset",
        train_ratio=0.8,
        seed=42
    )
