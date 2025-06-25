project executable files 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 1. Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 2. Load Dataset
train_dataset = ImageFolder(root='data/train', transform=transform)
val_dataset = ImageFolder(root='data/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 3. Load Pretrained Model
model = models.resnet18(pretrained=True)

# Replace final layer for our classes (e.g., 6 classes of waste)
num_classes = 6
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 4. Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
for epoch in range(5):  # You can increase epochs
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
