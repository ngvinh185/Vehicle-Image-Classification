# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("mohamedmaher5/vehicle-classification")

# print("Path to dataset files:", path)


from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from config import *

train_data_transform = transforms.Compose([
  transforms.Resize((img_size, img_size)), 
  transforms.RandomHorizontalFlip(),
  transforms.ColorJitter(brightness=0.2, contrast=0.2),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_data_transform = transforms.Compose([
  transforms.Resize((img_size, img_size)),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = ImageFolder(folder_data_train, transform = train_data_transform)
val_data = ImageFolder(folder_data_val, transform = val_data_transform)

train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True, num_workers=2)
dev_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle = False, num_workers=2)

