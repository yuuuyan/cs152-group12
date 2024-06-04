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
train_filepath = "DiscordBot/datasets/train.csv"
test_filepath = "DiscordBot/datasets/test.csv"

df_train = pd.read_csv(train_filepath)
df_test = pd.read_csv(test_filepath)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

training_tokenized_dataset = df_train.map(preprocess_function, batched=True)
testing_tokenized_dataset = df_test.map(preprocess_function, batched=True)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_tokenized_dataset["train"],
    eval_dataset=testing_tokenized_dataset["test"],
)

trainer.train()

results = trainer.evaluate()
print(results)