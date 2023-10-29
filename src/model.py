import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import pandas as pd
import os
import Processor
import ast
from sklearn.preprocessing import LabelEncoder

class ConvLSTMModel(nn.Module):
    def __init__(self, input_channels, hidden_size, num_classes):
        super(ConvLSTMModel, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(64)


        # LSTM Layers
        #self.lstm1 = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=1, batch_first=True)
        #self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

        # Fully Connected Layer with Softmax Activation
        self.fc = nn.Linear(65536, 512)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax along the class dimension


    def forward(self, x):
        # Convolutional Layers

        x = x.unsqueeze(1)

        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        # Reshape for LSTM
        #batch_size, _, features, time_steps = x.size()
        #x = x.view(batch_size, time_steps, features)  # Reshape to (batch_size, time_steps, features)

        # LSTM Layers
        #x, _ = self.lstm1(x)
        #x, _ = self.lstm2(x)


        # Select the output at the last time step
        # x = x[:, -1, :]
        # print(x.size())
        x = x.view(x.size(0), -1)


        # Fully Connected Layer
        x = self.fc(x)
        x = self.relu4(x)
        x = self.dropout1(x)
        x = self.fc2(x)

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
        strokes = row["Strokes"]

        return image, label, strokes


def get_mappings(): 
    current_directory = os.getcwd()
    data_file = current_directory + "/data2.csv"

    data = pd.read_csv(data_file)

    label_encoder = LabelEncoder()
    encoded_outputs = label_encoder.fit_transform(data["Label"])
    data["Label"] = encoded_outputs
    data['Strokes'] = None

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

        data.iloc[index] = {"Image": process_result, "Label": label, 'Strokes': row['Image']}
    
    # Define the size of the training and test sets
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size

    data = DatasetWrapper(data)

    # Use random_split to create training and test datasets
    train_dataset, test_dataset = random_split(data, [train_size, test_size])
   
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    return train_dataloader, test_dataloader


def train(train_dataloader):
    num_classes = 50
    hidden_size = 64
    model = ConvLSTMModel(1, hidden_size, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 10  # Adjust as needed
    total = 0
    correct = 0

    for epoch in range(num_epochs):

        for inputs in train_dataloader:
            # Assuming images is a batch of image tensors and labels is a batch of corresponding labels
            labels = inputs[1].long()
            images = inputs[0]
            #print(labels)
            #print(images)
            
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

        # Print the loss for each epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        accuracy = correct / total 
        print(accuracy)

    # Optionally, save the trained model
    torch.save(model.state_dict(), 'conv_lstm_model.pth')

    return model


def predict(model, data):
    
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs in data:
            # Assuming inputs[0] is the batch of image tensors
            images = inputs[0]
            labels = inputs[1]
            strokes = inputs[2]

            # Get the model's predictions
            outputs = model(images)

            # Get the predicted class for each item in the batch
            _, predicted = torch.max(outputs, 1)

            #print(predicted)
            #print(labels.size(0))
            #print(labels)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print(correct)

    accuracy = correct / total 

    return accuracy


def load_model():

    # Load the model from the .pth file
    model = ConvLSTMModel(1, 64, 50)  # Create an instance of your model
    model_path = 'conv_lstm_model.pth'  # Replace with the path to your .pth file

    # Load the state dictionary
    state_dict = torch.load(model_path)

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    return model


retrain = True
train_dataloader, test_dataloader = get_mappings()

if retrain:
    model = train(train_dataloader)
else:
    model = load_model()

accuracy = predict(model, test_dataloader)
print(f'Test Accuracy: {accuracy * 100:.2f}%')