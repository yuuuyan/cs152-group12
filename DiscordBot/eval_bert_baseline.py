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
import os
from safetensors.torch import load_file
from collections import defaultdict
import json
from tqdm import tqdm

ckpt_dir = "./results/checkpoint-4000"
output_file = "classifier_outputs.json"
num_save = 50

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

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
model.classifier = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2))
model.to(DEVICE)

model_path = os.path.join(ckpt_dir, "model.safetensors")
weights = load_file(model_path, "cuda")

# loading weights from checkpoint
print("Loading weights from checkpoint %s" % (model_path))
for name, param in model.named_parameters():
    to_load_weight = weights[name]
    model.state_dict()[name].data.copy_(to_load_weight)

outputs = defaultdict(dict)
tot = 0.0
correct = 0.0
for i in tqdm(range(len(tokenized_dataset["test"]))):
    outputs[i]["message"] = tokenized_dataset["test"]["text"][i]
    outputs[i]["gt"] = int(tokenized_dataset["test"]["labels"][i].cpu())

    input_ids = tokenized_dataset["test"]["input_ids"][i].unsqueeze(0).to(device)
    attention_mask = tokenized_dataset["test"]["attention_mask"][i].unsqueeze(0).to(device)
    labels =  tokenized_dataset["test"]["labels"][i].unsqueeze(0).to(device)
    token_type_ids = torch.tensor(tokenized_dataset["test"]["token_type_ids"][i]).unsqueeze(0).to(device)

    outputs[i]["prediction"] = int(torch.argmax(model(input_ids, attention_mask, token_type_ids)["logits"]).cpu())

    correct += (outputs[i]["gt"] == outputs[i]["prediction"])
    tot += 1.0

    if i % num_save == 0 and i != 0:
        with open(output_file, 'w') as f:
            json.dump(outputs, f, indent=6)      

with open(output_file, 'w') as f:
    json.dump(outputs, f, indent=6)

print("Accuracy: %.2f" % (correct / tot))