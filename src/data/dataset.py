import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from src import config

class SketchToPhotoDataset(Dataset):
    def __init__(self, hed_dir, photo_dir, max_samples=None):
        self.hed_dir = hed_dir
        self.photo_dir = photo_dir
        self.hed_files = sorted(os.listdir(hed_dir))
        self.photo_files = sorted(os.listdir(photo_dir))
        
        if max_samples and len(self.hed_files) > max_samples:
            import random
            indices = random.sample(range(len(self.hed_files)), max_samples)
            self.hed_files = [self.hed_files[i] for i in indices]
            self.photo_files = [self.photo_files[i] for i in indices]
        
        self.condition_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()  
        ])
        self.target_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])
    
    def __len__(self):
        return len(self.hed_files)

    def __getitem__(self, idx):
        hed_path = os.path.join(self.hed_dir, self.hed_files[idx])
        photo_path = os.path.join(self.photo_dir, self.photo_files[idx])
        
        hed_image = Image.open(hed_path).convert("RGB")
        photo = Image.open(photo_path).convert("RGB")
        
        hed_image = self.condition_transform(hed_image)  
        photo = self.target_transform(photo) 
        
        return {"hed": hed_image, "photo": photo}

def get_dataloaders():
    train_dataset = SketchToPhotoDataset(
        hed_dir=os.path.join(config.train_data_dir, "train", "sketches"),
        photo_dir=os.path.join(config.train_data_dir, "train", "photos"),
        max_samples=None 
    )
    val_dataset = SketchToPhotoDataset(
        hed_dir=os.path.join(config.train_data_dir, "val", "sketches"),
        photo_dir=os.path.join(config.train_data_dir, "val", "photos"),
        max_samples=None 
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"Train steps: {len(train_dataloader)}, Val steps: {len(val_dataloader)}")
    return train_dataloader, val_dataloader