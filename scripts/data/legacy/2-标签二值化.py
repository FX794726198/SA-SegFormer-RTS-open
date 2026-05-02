import os
import cv2
import numpy as np

def binarize_tif_labels(src_dir: str, dst_dir: str, thresh: int = None, invert: bool = False) -> None:
    """
    Read all .tif/.tiff files from src_dir, binarize them, and save results to dst_dir as .tif.

    :param src_dir: Directory containing source label files.
    :param dst_dir: Directory where binarized .tif files will be saved.
    :param thresh: Fixed threshold value (0-255). If None, uses Otsu's method.
    :param invert: If True, inverts the binary output.
    """
    os.makedirs(dst_dir, exist_ok=True)

    for fname in os.listdir(src_dir):
        if not fname.lower().endswith(('.tif', '.tiff')):
            continue

        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: could not read {src_path}")
            continue

        # Ensure data is 8-bit or 16-bit
        if img.dtype not in (np.uint8, np.uint16):
            img = img.astype(np.uint8)

        # Convert to grayscale if color
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Debug: show min/max intensity
        min_val, max_val = int(gray.min()), int(gray.max())
        print(f"{fname}: gray range = [{min_val}, {max_val}]")

        # Choose thresholding method
        if thresh is None:
            # Otsu's automatic threshold
            ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print(f"  Otsu threshold = {ret}")
        else:
            ret = thresh
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            print(f"  Applied fixed threshold = {thresh}")

        # Optionally invert
        if invert:
            binary = cv2.bitwise_not(binary)
            print("  Inverted binary image")

        # Ensure 8-bit output
        binary = binary.astype(np.uint8)

        # Save as TIFF
        cv2.imwrite(dst_path, binary)
        print(f"Saved binarized label to {dst_path}\n")

if __name__ == '__main__':
    # 示例调用：
    src_folder = '/Users/fengxiao/Desktop/label'
    dst_folder = '/Users/fengxiao/Desktop/label2'
    # thresh=None 启用 Otsu，invert 可根据需要设为 True
    binarize_tif_labels(src_folder, dst_folder, thresh=None, invert=False)
