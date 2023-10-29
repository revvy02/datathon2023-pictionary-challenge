import os
import torch

from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from model import ConvLSTMModel
from util import create_data_object
from config import TRAINING_BATCH_SIZE, TRAINING_SET_SIZE, RAW_DATA_SOURCE

def process_strokes(strokes):
    MAX_STROKES = 100
    MAX_LENGTH = 50

    processed = torch.zeros(MAX_STROKES, 2, MAX_LENGTH)

    for i, stroke in enumerate(strokes[:MAX_STROKES]):
        for j in range(2):
            processed[i, j, :len(stroke[j])] = torch.tensor(stroke[j][:MAX_LENGTH])

    return processed

class QuickDrawDataset(Dataset):
	def __init__(self, root_dir):
		self.data = []
		self.labels = []
        
		self.label_to_int = {}  # Mapping from label to integer
		self.int_to_label = {}  # Mapping from integer to label
        
		self.file_list = sorted([f for f in os.listdir(root_dir) if f.endswith('.ndjson')])
        
		for idx, file_name in enumerate(self.file_list):
			file_path = os.path.join(root_dir, file_name)
			label_str = file_name.rsplit('.', 1)[0]  
            
			self.label_to_int[label_str] = idx
			self.int_to_label[idx] = label_str
            
			with open(file_path, 'r') as f:
				class_data = [f.readline() for _ in range(TRAINING_SET_SIZE)]
                
				self.data.extend(class_data)
				self.labels.extend([idx] * len(class_data))

	def __len__(self):
		return len(self.labels)
	
	def __getitem__(self, idx):
		data_object = create_data_object(self.data[idx])
		
		image = torch.tensor(data_object["image"], dtype=torch.float32)
		strokes = process_strokes(data_object["strokes"]).float()
		label = torch.tensor(self.labels[idx], dtype=torch.long)

		return image, strokes, label
		
def train_model(model, train_loader, optimizer, loss_function, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0

        for i, (image, strokes, label) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(image, strokes)
            loss = loss_function(outputs, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    print("Finished Training")


def main():
    # Hyperparameters
    num_epochs = 10
    learning_rate = 0.001
    input_channels = 1  # assuming grayscale images
    hidden_size = 256  # you can adjust this
    num_classes = len(os.listdir(RAW_DATA_SOURCE))  # replace with your directory path

    # Model, Loss, Optimizer
    model = ConvLSTMModel(input_channels, hidden_size, num_classes)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_function = CrossEntropyLoss()

    # DataLoader
    dataset = QuickDrawDataset(RAW_DATA_SOURCE)  # replace with your directory path
    train_loader = DataLoader(dataset, batch_size=TRAINING_BATCH_SIZE, shuffle=True)

    # Train the model
    train_model(model, train_loader, optimizer, loss_function, num_epochs)

    # Save the model
    torch.save(model.state_dict(), "trained_model.pth")

if __name__ == '__main__':
    main()