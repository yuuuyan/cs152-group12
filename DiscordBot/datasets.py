import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

class DisinformationDataset(Dataset):
    def __init__(self, filepath):
        """
        Initialize the dataset.
        :param filepath: Path to the CSV file containing the disformation data.

        TODO: 
        - add any other necessary initialization pieces. add vocab, length management?
        """
        # sep='\t', header(=0), names may not be right
        self.data = pd.read_csv(filepath, sep='\t')
        self.prepare_data()

    def preprocess_data(self):
        """
        TODO:
        - finish coding data preparation
        - determine what X and y are
        - 
        """
        df = pd.DataFrame(self.data)

        # TODO: adjust this to fit format of final dataset
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.vocab = self.build_vocab(self.texts)
    
    def build_vocab(self, texts):
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for text in texts:
            for word in text.split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab    

def get_loader(filepath, batch_size=32, shuffle=False):
    """
    Create data loader.
    :param filepath: Path to the CSV data file.
    :param batch_size: Batch size.
    :param seq_length: Length of the sequence.
    :param shuffle: Whether to shuffle the data.
    """
    dataset = DisinformationDataset(filepath)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return loader

# Example usage
if __name__ == '__main__':
    datasets_home_dir = ''
    filepath = ''
    loader = get_loader(filepath, batch_size=64)
    

