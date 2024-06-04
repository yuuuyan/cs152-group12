# Import Required Packages
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
# import time
import datasets
# import DiscordBot.datasets.datasets as ds
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# TODO input filepath
train_filepath = "DiscordBot/datasets/train.csv"
test_filepath = "DiscordBot/datasets/test.csv"

data_files = {"train": train_filepath, "test": test_filepath}
dataset = datasets.load_dataset("csv", data_files=data_files)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_data(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_dataset = dataset.map(preprocess_data, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("misinformation", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(DEVICE)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

print(tokenized_dataset["train"])
trainer.train()

results = trainer.evaluate()
print(results)