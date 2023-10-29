import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
import numpy as np
import pandas as pd
import os
import Processor
import ast

class ConvLSTMModel(nn.Module):
    def __init__(self, input_channels, hidden_size, num_classes):
        super(ConvLSTMModel, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # LSTM Layers
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

        # Fully Connected Layer with Softmax Activation
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax along the class dimension


    def forward(self, x):
        # Convolutional Layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        # Reshape for LSTM
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, height, width * channels)

        # LSTM Layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Select the output at the last time step
        x = x[:, -1, :]

        # Fully Connected Layer
        x = self.fc(x)

        return x

class DatasetWrapper(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        row = self.base_dataset.iloc[index]
        
        image = torch.tensor(row["Image"], dtype=torch.float32)
        label = row["Label"]

        return image, label


def get_mappings(): 
    current_directory = os.getcwd()
    data_file = current_directory + "/data.csv"

    data = pd.read_csv(data_file)
    
    processor = Processor.Processor()
    
    # Iterate over the rows using iterrows()
    for index, row in data.iterrows():
        # print(row)
        
        # Access the "Image" column from the current row
        image = row["Image"]
        image = ast.literal_eval(image)

        # Process the image using the Processor instance
        process_result = processor.strokes_to_image(image)
        process_result = processor.image_to_array(process_result)

        # Do something with the process_result if needed
        label = row["Label"]

        data.iloc[index] = {"Image": process_result, "Label": label}
    
    # Define the size of the training and test sets
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size

    data = DatasetWrapper(data)

    # Use random_split to create training and test datasets
    train_dataset, test_dataset = random_split(data, [train_size, test_size])
   
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_dataloader, test_dataloader


def train(train_dataloader):
    model = ConvLSTMModel(64, 64, 345)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10  # Adjust as needed

    for epoch in range(num_epochs):

        for inputs in train_dataloader:
            # Assuming images is a batch of image tensors and labels is a batch of corresponding labels
            labels = inputs[1] 
            images = inputs[0]
            #print(labels)
            #print(images)
            
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

        # Print the loss for each epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Optionally, save the trained model
    torch.save(model.state_dict(), 'conv_lstm_model.pth')




def predict():
    pass

"""
# Create an instance of the model
input_channels = 1  # Adjust based on your input data
hidden_size = 64
num_classes = 345  # Adjust based on your task
model = ConvLSTMModel(input_channels, hidden_size, num_classes)

# Print the model architecture
print(model)
"""

train_dataloader, test_dataloader = get_mappings()
train(train_dataloader)