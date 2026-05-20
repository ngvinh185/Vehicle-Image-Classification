import os
import shutil
import random
from pathlib import Path

# Cấu hình
SOURCE_DIR = "Img"  # Thư mục gốc
OUTPUT_DIR = "dataset"                  # Thư mục output
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
SEED = 42

random.seed(SEED)

# Các class (folder)
classes = ["Auto Rickshaws", "Bikes", "Cars", "Motorcycles", "Planes", "Ships", "Trains"]

for cls in classes:
    src_folder = Path(SOURCE_DIR) / cls
    
    # Lấy tất cả file ảnh
    images = [f for f in src_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
    random.shuffle(images)
    
    split_idx = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Copy vào train/val
    for split, split_images in [("train", train_images), ("val", val_images)]:
        dest_folder = Path(OUTPUT_DIR) / split / cls
        dest_folder.mkdir(parents=True, exist_ok=True)
        
        for img in split_images:
            shutil.copy2(img, dest_folder / img.name)
    
    print(f"{cls}: {len(train_images)} train, {len(val_images)} val")

print("\nDone! Cấu trúc output:")
print(f"{OUTPUT_DIR}/")
print("  train/ -> Auto Rickshaws, Bikes, Cars, Motorcycles, Planes, Ships, Trains")
print("  val/   -> Auto Rickshaws, Bikes, Cars, Motorcycles, Planes, Ships, Trains")