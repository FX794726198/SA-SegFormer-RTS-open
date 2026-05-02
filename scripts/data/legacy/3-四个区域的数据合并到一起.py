import os
import shutil
from pathlib import Path

# 定义根目录
base_dir = Path(r"/Users/fengxiao/Desktop/Work/语义分割/数据集0613")

# 区域文件夹及其对应前缀映射
regions = {
    "北1区域数据集": "北1区域_",
    "北2区域数据集": "北2区域_",
    "北3区域数据集": "北3区域_",
    "中1区域数据集": "中1区域_",
}

# 目标文件夹
output_img_dir = base_dir / "/Users/fengxiao/Desktop/img"
output_label_dir = base_dir / "/Users/fengxiao/Desktop/label"

# 创建目标文件夹（如果不存在）
output_img_dir.mkdir(parents=True, exist_ok=True)
output_label_dir.mkdir(parents=True, exist_ok=True)

for region_folder, prefix in regions.items():
    region_path = base_dir / region_folder
    img_src = region_path / "images"
    lbl_src = region_path / "labels"

    # 处理 images 文件夹
    if img_src.exists():
        for file_path in img_src.iterdir():
            if file_path.is_file():
                # 拼接新的文件名
                new_name = prefix + file_path.name
                dest = output_img_dir / new_name
                # 复制文件
                shutil.copy2(file_path, dest)
    else:
        print(f"警告：未找到 {img_src}")

    # 处理 labels 文件夹
    if lbl_src.exists():
        for file_path in lbl_src.iterdir():
            if file_path.is_file():
                new_name = prefix + file_path.name
                dest = output_label_dir / new_name
                shutil.copy2(file_path, dest)
    else:
        print(f"警告：未找到 {lbl_src}")

print("文件合并和重命名完成！")
