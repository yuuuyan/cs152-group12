# Import Required Packages
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
# import time
import DiscordBot.datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# TODO input filepath
filepath = "DiscordBot/datasets/train.csv"
df = pd.read_csv(filepath)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

X = sequences.view(sequences.shape[0], -1)
y = targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)
eval_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_test, y_test),
    batch_size=64,
    shuffle=False
)

num_epochs = 10
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for curr_X, curr_y in train_loader:
        curr_X, curr_y = curr_X.to(DEVICE), curr_y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(curr_X)
        loss = criterion(outputs, curr_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Evaluation Loop
model.eval()
all_predictions = []
all_targets = []
total_loss = 0.0
with torch.no_grad():
    for curr_X, curr_y in eval_loader:
        curr_X, curr_y = curr_X.to(DEVICE), curr_y.to(DEVICE)
        outputs = model(curr_X)
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(curr_y.cpu().numpy())
        loss = criterion(outputs, curr_y)
        total_loss += loss.item()

# Concatenate predictions and targets
all_predictions = np.concatenate(all_predictions)
all_targets = np.concatenate(all_targets)

# Calculate evaluation metrics
mae = mean_absolute_error(all_targets, all_predictions)
rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
r2 = r2_score(all_targets, all_predictions)

end_time = time.time()

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")

avg_loss = total_loss / len(eval_loader)
print(f"Average Loss: {avg_loss}")

print(f"Total Time: {end_time - start_time} seconds")


"""import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

dataset_filepath = ""
dataset = load_dataset(dataset_filepath)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")"""
