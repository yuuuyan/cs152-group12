import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from datasets import DisinformationDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from vocab import Vocab
from DiscordBot.train_simple_rnn import SimpleRNN
import numpy as np

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def evaluate_model(eval_loader, model):
    # Eval loop
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for curr_X, curr_y in eval_loader:
            curr_X, curr_y = curr_X.to(DEVICE), curr_y.to(DEVICE)
            outputs = model(curr_X)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(curr_y.cpu().numpy())
            
    # Concatenate predictions and targets
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    accuracy = accuracy_score(all_predictions, all_targets)
    print(f'Validation Accuracy: {accuracy}')

if __name__ == "__main__":
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

    eval_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=True)
    vocab_size = len(disinfo_dataset.vocab)

    # hyperparams: open to change
    embedding_dim = 10
    hidden_dim = 10
    output_dim = 1
    
    model = SimpleRNN(vocab_size, embedding_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load('simple_rnn_model/model_params.pth'))
    
    evaluate_model(eval_loader, model)


