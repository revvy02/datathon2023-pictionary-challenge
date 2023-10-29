import torch
import torch.nn as nn

class ConvLSTMModel(nn.Module):
	def __init__(self, input_channels, hidden_size, num_classes):
		super(ConvLSTMModel, self).__init__()

		# Convolutional Layers
		self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
		self.relu1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.relu2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.relu3 = nn.ReLU()
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

		# Fully Connected Layer with Softmax Activation
		self.fc = nn.Linear(65536, 1024)
		self.relu4 = nn.ReLU()
		self.dropout1 = nn.Dropout(0.3)

		self.fc2 = nn.Linear(1024, num_classes)
		self.softmax = nn.Softmax(dim=1)  # Apply softmax along the class dimension

	def forward(self, x):
		# Convolutional Layers
		x = x.unsqueeze(1)

		x = self.pool1(self.relu1(self.conv1(x)))
		x = self.pool2(self.relu2(self.conv2(x)))
		x = self.pool3(self.relu3(self.conv3(x)))

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

				print(predicted)
				print(labels.size(0))
				print(labels)

				total += labels.size(0)
				correct += (predicted == labels).sum().item()

				print(correct)

		accuracy = correct / total 

		return accuracy