import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

# 1. Data Loading
def load_data(classes, data_path):
    all_data, all_labels = [], []
    for idx, label in enumerate(classes):
        data = np.load(f"{data_path}/{label}.npy")
        data = data.astype(np.float32) / 255.0
        all_data.append(data)
        all_labels.extend([idx] * data.shape[0])
    return np.vstack(all_data), np.array(all_labels)

classes = ['cat', 'car', 'apple']
X, y = load_data(classes, 'path_to_numpy_bitmap_data')

# 2. Data Preprocessing
transform = transforms.Compose([transforms.ToTensor()])
X_tensor, y_tensor = torch.Tensor(X), torch.LongTensor(y)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 3. Model Selection
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# If you have GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 4. Training
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Flatten the data from (BATCH, 1, 28, 28) to (BATCH, 28*28) because QuickDraw is in grayscale and ResNet expects 3 channels
        inputs = inputs.view(inputs.size(0), -1)
        
        # Since ResNet expects 3 channels, repeat the single grayscale channel 3 times.
        inputs = inputs.repeat(1, 3).view(-1, 3, 28, 28)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    avg_train_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_train_loss:.4f}")

print("Training Completed!")

# Save the model
torch.save(model.state_dict(), 'quick_draw_model.pth')