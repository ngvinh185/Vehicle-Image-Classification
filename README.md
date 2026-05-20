# Vehicle Image Classification

Dự án phân loại ảnh phương tiện giao thông gồm **7 lớp**: Auto Rickshaws, Bikes, Cars, Motorcycles, Planes, Ships, Trains.

Sử dụng **EfficientNet-B3** pretrained trên ImageNet làm backbone, finetune với custom head trên bộ dữ liệu riêng.

---

## Cấu trúc thư mục

```
Vehicle-Image-Classification/
│
├── dataset/
│   ├── train/
│   │   ├── Auto Rickshaws/
│   │   ├── Bikes/
│   │   ├── Cars/
│   │   ├── Motorcycles/
│   │   ├── Planes/
│   │   ├── Ships/
│   │   └── Trains/
│   └── val/
│       ├── Auto Rickshaws/
│       ├── Bikes/
│       ├── Cars/
│       ├── Motorcycles/
│       ├── Planes/
│       ├── Ships/
│       └── Trains/
│
├── logs/                       # Log training (tự động tạo)
│   └── training.log
│
├── config.py                   # Cấu hình hyperparameter
├── dataset.py                  # Load và transform dữ liệu
├── model.py                    # Định nghĩa kiến trúc model
├── train.py                    # Training loop
├── evaluate.py                 # Vẽ biểu đồ loss/accuracy
├── test.py                     # Script chia train/val từ dữ liệu gốc
├── utils.py                    # Logger
└── index.py                    # Entry point
```

---

## Mô tả từng file

**`config.py`** — Chứa toàn bộ hyperparameter: image size (224), đường dẫn dataset, số class (7), device, learning rate của backbone (`1e-5`) và head (`1e-3`), batch size (16), số epoch tối đa (70).

**`test.py`** — Script tiền xử lý dữ liệu: đọc ảnh từ thư mục `Img/`, shuffle và chia 80/20 thành train/val, copy vào thư mục `dataset/` theo cấu trúc `ImageFolder`.

**`dataset.py`** — Định nghĩa transform cho train (resize, flip, color jitter, normalize) và val (resize, normalize). Khởi tạo `ImageFolder` và `DataLoader` cho cả hai tập.

**`model.py`** — Định nghĩa class `Model` gồm:
- **Backbone**: `efficientnet_b3.features` (pretrained ImageNet, giữ output 4D)
- **Head**: `ModuleDict` gồm `custom_conv` (Conv2d → BN → ReLU) và `fc` (Linear → BN → ReLU → Dropout → Linear)

**`utils.py`** — Hàm `get_logger()` tạo logger ghi log ra file `logs/training.log` và stdout.

**`train.py`** — Training loop chính: khởi tạo model, loss (CrossEntropy), optimizer (AdamW với differential lr), scheduler (CosineAnnealingLR). Mỗi epoch tính train/val loss và accuracy, log kết quả và lưu checkpoint.

**`evaluate.py`** — Hàm `plot_training_progress()` vẽ 2 biểu đồ Loss và Accuracy theo epoch (train vs val), lưu ra file `training_progress_separate.png`.

**`index.py`** — Entry point, import và chạy toàn bộ pipeline (train + evaluate).

---

## Cài đặt

```bash
git clone https://github.com/ngvinh185/Vehicle-Image-Classification.git
cd Vehicle-Image-Classification

pip install torch torchvision matplotlib
```

---

## Chạy

**Bước 1:** Cấu hình (tuỳ chỉnh nếu cần)

```python
# config.py
device = 'cuda:0'     # đổi GPU nếu cần
max_epoch = 70
batch_size = 16
```

**Bước 2:** Train

```bash
python index.py
```

Kết quả log sẽ in ra màn hình và lưu vào `logs/training.log`. Checkpoint được lưu tự động sau mỗi epoch tại `checkpoint.pth`.

---

## Kiến trúc Model

```
Input (224×224×3)
    ↓
EfficientNet-B3 features  
    ↓
Conv2d(1536→2048, k=3, s=1) + BN + ReLU  
    ↓
Conv2d(2048→2048, k=5, s=1) + BN + ReLU   
    ↓
AdaptiveAvgPool2d(1)
    ↓
Linear(2048→512) + BN + ReLU + Dropout(0.5)
    ↓
Linear(512→7)                # (N, 7)
```

---

## Kết quả

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1     | 0.3514    | 0.0972   | 89.17%    | 96.78%  |
| 2     | 0.2371    | 0.0518   | 93.25%    | 98.30%  |
| 3     | 0.1600    | 0.0616   | 95.39%    | 98.30%  |
| 4     | 0.1413    | 0.0426   | 96.26%    | 98.57%  |
| 5     | 0.1228    | 0.0373   | 96.82%    | 98.66%  |
| 6     | 0.0936    | 0.0436   | 97.34%    | 98.66%  |
| 7     | 0.0827    | 0.0365   | 97.83%    | 98.84%  |
| 8     | 0.0792    | 0.0466   | 97.99%    | 98.75%  |
| 9     | 0.1021    | 0.0476   | 97.36%    | 98.93%  |
| 10    | 0.0747    | 0.0418   | 97.94%    | 99.11%  |
| 11    | 0.0601    | 0.0516   | 98.59%    | 98.84%  |
| 12    | 0.0619    | 0.0291   | 98.43%    | 99.37%  |
| 13    | 0.0660    | 0.0348   | 98.30%    | 98.93%  |
| 14    | 0.0376    | 0.0464   | 99.04%    | 98.66%  |
| 15    | 0.0549    | 0.0369   | 98.57%    | 98.84%  |
| 16    | 0.0376    | 0.0285   | 99.15%    | 99.02%  |
| 17    | 0.0478    | 0.0308   | 98.84%    | 99.28%  |
| 18    | 0.0432    | 0.0277   | 98.95%    | 99.28%  |
| 19    | 0.0380    | 0.0387   | 99.04%    | 98.93%  |
| 20    | 0.0395    | 0.0432   | 99.13%    | 98.93%  |
| 21    | 0.0344    | 0.0292   | 99.11%    | 99.11%  |
| 22    | 0.0439    | 0.0496   | 99.06%    | 98.93%  |
| 23    | 0.0251    | 0.0332   | 99.42%    | 99.11%  |
| 24    | 0.0344    | 0.0346   | 99.19%    | 99.11%  |
| 25    | 0.0341    | 0.0157   | 99.24%    | 99.64%  |
| 26    | 0.0440    | 0.0426   | 98.97%    | 98.75%  |
| 27    | 0.0465    | 0.0281   | 99.04%    | 99.37%  |
| 28    | 0.0227    | 0.0488   | 99.49%    | 99.28%  |
| 29    | 0.0221    | 0.0229   | 99.40%    | 99.28%  |
| 30    | 0.0325    | 0.0223   | 99.19%    | 99.37%  |
| 31    | 0.0154    | 0.0522   | 99.51%    | 99.28%  |
| 32    | 0.0258    | 0.0215   | 99.44%    | 99.28%  |
| 33    | 0.0156    | 0.0319   | 99.60%    | 99.19%  |
| 34    | 0.0146    | 0.0498   | 99.60%    | 98.75%  |
| 35    | 0.0133    | 0.0214   | 99.58%    | 99.28%  |
| 36    | 0.0207    | 0.0613   | 99.64%    | 98.84%  |
| 37    | 0.0170    | 0.0529   | 99.60%    | 98.93%  |
| 38    | 0.0219    | 0.0515   | 99.60%    | 98.93%  |
| 39    | 0.0185    | 0.0394   | 99.55%    | 99.02%  |
| 40    | 0.0093    | 0.0387   | 99.71%    | 99.11%  |
| 41    | 0.0149    | 0.0432   | 99.62%    | 99.28%  |
| 42    | 0.0162    | 0.0424   | 99.60%    | 99.19%  |
| 43    | 0.0206    | 0.0643   | 99.58%    | 99.02%  |
| 44    | 0.0155    | 0.0708   | 99.58%    | 98.75%  |
| 45    | 0.0299    | 0.0527   | 99.53%    | 99.19%  |
| 46    | 0.0071    | 0.0449   | 99.80%    | 99.28%  |
| 47    | 0.0185    | 0.0402   | 99.69%    | 99.28%  |
| 48    | 0.0101    | 0.0489   | 99.69%    | 99.28%  |
| 49    | 0.0121    | 0.0334   | 99.69%    | 99.28%  |
| 50    | 0.0117    | 0.0255   | 99.75%    | 99.46%  |
 
