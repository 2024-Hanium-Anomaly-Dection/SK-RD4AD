import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
import os
from scipy.stats import zscore

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, z_thresh=3.0):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # Z-Score 계산 및 이상치 탐지
        numeric_data = self.data.iloc[:, 2:]  # 이미지 경로와 라벨을 제외한 나머지 데이터를 사용
        z_scores = zscore(numeric_data, axis=0)
        self.data['zscore'] = z_scores.max(axis=1)
        self.data['is_outlier'] = self.data['zscore'] > z_thresh
        
        # 이상치 제거
        self.data = self.data[self.data['is_outlier'] == False]
        self.data = self.data.drop(columns=['zscore', 'is_outlier'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.data.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        
        sample = {'image': image, 'label': torch.tensor(label, dtype=torch.long)}

        return sample

# 데이터 전처리 및 증강
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 데이터셋 및 DataLoader 생성
csv_file_path = 'C:/Users/your_username/Desktop/vad/data.csv'
dataset = CustomDataset(csv_file=csv_file_path, transform=transform)

# 학습 및 검증 데이터셋 분할
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

# 데이터 배치 확인
for i, batch in enumerate(train_loader):
    print(f"Batch {i+1}")
    print(f"Images: {batch['image'].shape}")
    print(f"Labels: {batch['label']}")
    break  # 첫 번째 배치만 출력
