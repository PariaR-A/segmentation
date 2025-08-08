import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
from swin_unet_detection import SwinUnetDetection
from networks.vision_transformer import SwinUnet as ViT_seg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Dataset
# ------------------------------
class NPZSegDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        image = torch.tensor(data["image"], dtype=torch.float32)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        mask = torch.tensor(data["label"], dtype=torch.float32)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return image, mask

# ------------------------------
# Model
# ------------------------------
seg_model = ViT_seg(img_size=224, num_classes=1)
model = SwinUnetDetection(seg_model).to(device)
model.load_state_dict(torch.load("swin_det.pth", map_location=device))
model.eval()

test_dataset = NPZSegDataset("data/test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ------------------------------
# Inference
# ------------------------------
with torch.no_grad():
    for i, (images, masks) in enumerate(test_loader):
        images = images.to(device)
        boxes = model(images)  # list of boxes
        print(f"Image {i} -> {boxes}")
