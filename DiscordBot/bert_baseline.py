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
import evaluate

# adapted from https://huggingface.co/docs/transformers/en/training
metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# TODO input filepath
train_filepath = "./Datasets/train.csv"
test_filepath = "./Datasets/test.csv"

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

# freezing pretrained BERT encoder
for param in model.base_model.parameters():
    param.requires_grad = False

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics
)


print("Trainng model")
trainer.train()

print("Running evaluation")
results = trainer.evaluate()
print(results)