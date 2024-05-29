import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import DisinformationDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from vocab import Vocab
import numpy as np

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, dummy = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# TODO: add filepath for data
filepath = ''
disinfo_dataset = DisinformationDataset(filepath=filepath)

# TODO: dummy dimensions; replace with real ones after finalizing dataset
batch_size = 32
input_features = 20
num_classes = 2

inputs = disinfo_dataset.data
labels = torch.randint(0, num_classes, (batch_size,))

model = SimpleRNN(input_dim=input_features, hidden_dim=50, output_dim=num_classes)
criterion = nn.BCELoss()  # binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

outputs = model(inputs)
loss = criterion(outputs, labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f'Loss: {loss.item()}')