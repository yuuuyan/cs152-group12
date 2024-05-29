import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datasets import DisinformationDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

X = disinfo_dataset.texts
y = disinfo_dataset.labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

model = SimpleRNN(input_dim=input_features, hidden_dim=50, output_dim=num_classes)
loss_fn = nn.BCELoss()  # binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# hyperparams
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    for curr_X, curr_y in train_loader:
        curr_X, curr_y = curr_X.to(DEVICE), curr_y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(curr_X)
        loss = loss_fn(outputs, curr_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

torch.save(model.state_dict(), 'simple_rnn_model/model_params.pth')
model.to(DEVICE)