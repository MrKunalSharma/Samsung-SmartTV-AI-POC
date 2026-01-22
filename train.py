import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleSceneCNN
import os

# Settings
BATCH_SIZE = 16
EPOCHS = 5 # Transfer learning is fast; 5 epochs is plenty
LEARNING_RATE = 0.001
DATA_DIR = './data/train'

# 1. Advanced Data Augmentation (Makes the AI smarter)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # AI sees 'left-to-right' and 'right-to-left'
    transforms.RandomRotation(10),     # AI learns even if the camera is tilted
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SimpleSceneCNN(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.base_model.classifier[1].parameters(), lr=LEARNING_RATE)

    print(f"Found {len(dataset.classes)} classes: {dataset.classes}")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(loader):.4f}")

    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), './saved_models/tv_scene_model.pth')
    print("Success! Professional MobileNet model saved.")

if __name__ == "__main__":
    train()