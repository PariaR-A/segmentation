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
        image = torch.tensor(data["image"], dtype=torch.float32)  # (H,W) or (C,H,W)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        mask = torch.tensor(data["label"], dtype=torch.float32)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return image, mask

# ------------------------------
# Model
# ------------------------------
seg_model = ViT_seg(img_size=224, num_classes=1)  # adjust num_classes
model = SwinUnetDetection(seg_model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_dataset = NPZSegDataset("data/train")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# ------------------------------
# Training loop
# ------------------------------
epochs = 5
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        out = model(images, target_masks=masks)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "swin_det.pth")
