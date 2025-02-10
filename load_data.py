import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import os


class AOIData(Dataset):
    def __init__(self,df,path,transform=None):
        self.df=df
        self.transform=transform
        self.path=path
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        data_id=str(self.df.loc[idx,'ID'])
        data_path=os.path.join(self.path,data_id)
        image = Image.open(
            data_path
        ).convert("RGB")
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        enhanced_pil = Image.fromarray(enhanced_rgb)
        if self.transform:
            enhanced_pil = self.transform(enhanced_pil)
        label=self.df.loc[idx,"Label"]
        
        return {
            'image':enhanced_pil,
            'label':label
        }
    

train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')
train_dir='./train_images'
test_dir='./test_images'

def Split_train_valid(df):
    train_df_split, valid_df = train_test_split(
                df, test_size=0.2, random_state=42,stratify=df["Label"]
                )
    train_df_split=train_df_split.reset_index(drop=True)
    valid_df=valid_df.reset_index(drop=True)
    return train_df_split,valid_df


train_df_split, valid_df=Split_train_valid(train_df)
# print(len(train_df_split),len(valid_df))
    

train_transform = transforms.Compose(
    [
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)
valid_transform = transforms.Compose(
    [
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)


train_dataset = AOIData(train_df_split,train_dir, transform=train_transform)
valid_dataset = AOIData(valid_df, train_dir,transform=valid_transform)
test_dataset = AOIData(test_df, test_dir,transform=test_transform)



train_data_loader = torch.utils.data.DataLoader(train_dataset, 
    batch_size=128,
    shuffle=True,
    num_workers=2
)
val_data_loader = torch.utils.data.DataLoader(valid_dataset, 
    batch_size=128,
    shuffle=False,
    num_workers=2
)
test_data_loader = torch.utils.data.DataLoader(test_dataset, 
    batch_size=128,
    shuffle=False,
    num_workers=2
)
