import torch
from models.FlowNetS import FlowNetS
from models.FlowNetC import FlowNetC

# Select the model type
model_type = 'FlowNetS'  # Change to 'FlowNetC' if needed
# Correct the path to the pre-trained model
pretrained_path = "D:/FISE A3/Semestre 6/UE_G - Computer Vision/Project/FlowNet 2025/ComputerVision-FlowNet-main/models_PyTorch/pytorch/flownets_EPE1.951.pth"

# Load model
if model_type == 'FlowNetS':
    model = FlowNetS(batchNorm=False)  # Set batchNorm=True if using batch normalization
elif model_type == 'FlowNetC':
    model = FlowNetC(batchNorm=False)

# Load pre-trained weights
checkpoint = torch.load(pretrained_path)
model.load_state_dict(checkpoint['state_dict'])

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.train()  # Set model to training mode

from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np

class OpticalFlowDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Separate real images and ground-truth images
        self.real_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.bmp')])
        self.gt_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

        # Ensure the number of real and ground-truth images match
        assert len(self.real_images) == len(self.gt_images), "Mismatch between real and ground-truth images"

    def __len__(self):
        return len(self.real_images) - 1  # Pairs of images

    def __getitem__(self, idx):
        # Get real image paths
        real_img1_path = os.path.join(self.image_dir, self.real_images[idx])
        real_img2_path = os.path.join(self.image_dir, self.real_images[idx + 1])

        # Get ground-truth image paths
        gt_img1_path = os.path.join(self.image_dir, self.gt_images[idx])
        gt_img2_path = os.path.join(self.image_dir, self.gt_images[idx + 1])

        # Load real images
        real_img1 = cv2.imread(real_img1_path)
        real_img2 = cv2.imread(real_img2_path)

        # Load ground-truth images
        gt_img1 = cv2.imread(gt_img1_path, cv2.IMREAD_GRAYSCALE)  # Assuming ground-truth is grayscale
        gt_img2 = cv2.imread(gt_img2_path, cv2.IMREAD_GRAYSCALE)

        # Check if images are loaded successfully
        if real_img1 is None or real_img2 is None or gt_img1 is None or gt_img2 is None:
            raise FileNotFoundError(f"Failed to load images: {real_img1_path}, {real_img2_path}, {gt_img1_path}, {gt_img2_path}")

        # Convert real images to RGB
        real_img1 = cv2.cvtColor(real_img1, cv2.COLOR_BGR2RGB)
        real_img2 = cv2.cvtColor(real_img2, cv2.COLOR_BGR2RGB)

        # Resize images to FlowNet input size
        real_img1 = cv2.resize(real_img1, (512, 384))
        real_img2 = cv2.resize(real_img2, (512, 384))
        gt_img1 = cv2.resize(gt_img1, (512, 384))
        gt_img2 = cv2.resize(gt_img2, (512, 384))

        # Normalize and transpose real images
        real_img1 = np.transpose(real_img1, (2, 0, 1)) / 255.0
        real_img2 = np.transpose(real_img2, (2, 0, 1)) / 255.0

        # Normalize ground-truth images
        gt_img1 = gt_img1 / 255.0
        gt_img2 = gt_img2 / 255.0

        # Convert to tensors
        real_img1 = torch.tensor(real_img1, dtype=torch.float32)
        real_img2 = torch.tensor(real_img2, dtype=torch.float32)
        gt_img1 = torch.tensor(gt_img1, dtype=torch.float32)
        gt_img2 = torch.tensor(gt_img2, dtype=torch.float32)

        return real_img1, real_img2, gt_img1, gt_img2

# =================================== For 80% training and 20% testing split ===================================
# from torch.utils.data import random_split
# # Load dataset
# dataset = OpticalFlowDataset("D:/FISE A3/Semestre 6/UE_G - Computer Vision/Project/FlowNet 2025/ComputerVision-FlowNet-main/sequences-train")
# # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# # Split dataset into 80% training and 20% testing
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# =========================================================================================================

# =================================== Choose several sequence classes for training ===================================
# Define the sequences for training and testing
train_sequences = ["bag", "bear", "camel", "swan"]
test_sequences = ["book", "rhino"]

# Load datasets
train_dataset = OpticalFlowDataset(
    "D:/FISE A3/Semestre 6/UE_G - Computer Vision/Project/FlowNet 2025/ComputerVision-FlowNet-main/sequences-train",
    train_sequences,
)
test_dataset = OpticalFlowDataset(
    "D:/FISE A3/Semestre 6/UE_G - Computer Vision/Project/FlowNet 2025/ComputerVision-FlowNet-main/sequences-train",
    test_sequences,
)
# =========================================================================================================

# Create DataLoaders for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class EPELoss(nn.Module):
    def __init__(self):
        super(EPELoss, self).__init__()

    def forward(self, pred_flow, gt_flow):
        return torch.norm(pred_flow - gt_flow, p=2, dim=1).mean()

# Define loss & optimizer
criterion = EPELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 40  # Adjust based on dataset size

# for epoch in range(num_epochs):
#     running_loss = 0.0

#     for real_img1, real_img2, gt_img1, gt_img2 in dataloader:
#         real_img1, real_img2 = real_img1.to(device), real_img2.to(device)
#         gt_img1, gt_img2 = gt_img1.to(device), gt_img2.to(device)

#         optimizer.zero_grad()
        
#         # Forward pass (FlowNet expects concatenated inputs)
#         inputs = torch.cat([real_img1, real_img2], dim=1)  # Concatenate along channel dimension
#         pred_flow = model(inputs)

#         # If pred_flow is a tuple, extract the first element
#         if isinstance(pred_flow, tuple):
#             pred_flow = pred_flow[0]

#         # Use ground-truth flow as the target
#         gt_flow = torch.cat([gt_img1.unsqueeze(1), gt_img2.unsqueeze(1)], dim=1)  # Combine ground-truth flows

#         # Resize ground-truth flow to match the dimensions of pred_flow
#         gt_flow = F.interpolate(gt_flow, size=pred_flow.shape[-2:], mode='bilinear', align_corners=False)

#         # Compute loss
#         loss = criterion(pred_flow, gt_flow)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# # Save fine-tuned model
# torch.save(model.state_dict(), "flownets_EPE1.951_finetuned.pth")

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for real_img1, real_img2, gt_img1, gt_img2 in train_dataloader:
        real_img1, real_img2 = real_img1.to(device), real_img2.to(device)
        gt_img1, gt_img2 = gt_img1.to(device), gt_img2.to(device)

        optimizer.zero_grad()
        
        # Forward pass (FlowNet expects concatenated inputs)
        inputs = torch.cat([real_img1, real_img2], dim=1)  # Concatenate along channel dimension
        pred_flow = model(inputs)

        # If pred_flow is a tuple, extract the first element
        if isinstance(pred_flow, tuple):
            pred_flow = pred_flow[0]

        # Use ground-truth flow as the target
        gt_flow = torch.cat([gt_img1.unsqueeze(1), gt_img2.unsqueeze(1)], dim=1)  # Combine ground-truth flows

        # Resize ground-truth flow to match the dimensions of pred_flow
        gt_flow = F.interpolate(gt_flow, size=pred_flow.shape[-2:], mode='bilinear', align_corners=False)

        # Compute loss
        loss = criterion(pred_flow, gt_flow)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f}")

# Save fine-tuned model
torch.save(model.state_dict(), "flownets_EPE1.951_finetuned_selected_sequences.pth")

# Testing loop
model.eval()  # Set model to evaluation mode
test_loss = 0.0

with torch.no_grad():
    for real_img1, real_img2, gt_img1, gt_img2 in test_dataloader:
        real_img1, real_img2 = real_img1.to(device), real_img2.to(device)
        gt_img1, gt_img2 = gt_img1.to(device), gt_img2.to(device)

        # Forward pass
        inputs = torch.cat([real_img1, real_img2], dim=1)
        pred_flow = model(inputs)

        if isinstance(pred_flow, tuple):
            pred_flow = pred_flow[0]

        gt_flow = torch.cat([gt_img1.unsqueeze(1), gt_img2.unsqueeze(1)], dim=1)
        gt_flow = F.interpolate(gt_flow, size=pred_flow.shape[-2:], mode='bilinear', align_corners=False)

        # Compute loss
        loss = criterion(pred_flow, gt_flow)
        test_loss += loss.item()

print(f"Test Loss: {test_loss/len(test_dataloader):.4f}")