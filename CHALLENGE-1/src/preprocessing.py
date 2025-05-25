# preprocessing.py


"""

Author: Annam.ai IIT Ropar
Team Name: Ice 'N' Dagger
Team Members: Barun Saha, Bibaswan Das
Leaderboard Rank: 70

"""

# Here you add all the preprocessing related details for the task completed from Kaggle.
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder

BATCH_SIZE = 32
IMG_SIZE = 224
TRAIN_DIR = "/kaggle/input/soil-classification/soil_classification-2025/train"
TEST_DIR = "/kaggle/input/soil-classification/soil_classification-2025/test"
TRAIN_CSV = "/kaggle/input/soil-classification/soil_classification-2025/train_labels.csv"
TEST_CSV = "/kaggle/input/soil-classification/soil_classification-2025/test_ids.csv"

# Label encoding
df = pd.read_csv(TRAIN_CSV)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['soil_type'])
df.to_csv("encoded_train.csv", index=False)  # Save for reuse in training and inference

# Save label encoder
import joblib
joblib.dump(label_encoder, "label_encoder.pkl")

# Test CSV is not modified, so no need to save

# Dataset class
class SoilDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'])
        image = Image.open(img_path).convert("RGB")
        label = row['label'] if 'label' in row else -1
        if self.transform:
            image = self.transform(image)
        return image, label, row['image_id']

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
