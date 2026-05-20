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
EfficientNet-B3 features     # (N, 1536, 7, 7)
    ↓
Conv2d(1536→2048, k=7, s=2) + BN + ReLU   # (N, 2048, 2, 2)
    ↓
Flatten                      # (N, 8192)
    ↓
Linear(8192→512) + BN + ReLU + Dropout(0.5)
    ↓
Linear(512→7)                # (N, 7)
```

---

## Kết quả

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1     | 0.3119    | 0.0785   | 90.09%    | 97.67%  |
