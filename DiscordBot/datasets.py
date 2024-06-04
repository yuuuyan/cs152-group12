import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

class MisinformationDataset(Dataset):
    def __init__(self, true_filepath, false_filepath, filepath=None):
        """
        Initialize the dataset.
        :param filepath: Path to the CSV file containing the disformation data.

        TODO: 
        - add any other necessary initialization pieces. add vocab, length management?
        """
        # sep='\t', header(=0), names may not be right
        if filepath:
            self.data = pd.read_csv(filepath, sep='\t')
            self.prepare_data()
        else:
            self.true_df = pd.DataFrame(pd.read_csv(true_filepath, sep='\t'))
            self.false_df = pd.DataFrame(pd.read_csv(false_filepath, sep='\t'))
            self.prepare_data_dual_input()

    def prepare_data_dual_input(self):
        """
        TODO:
        - finish coding data preparation
        - determine what X and y are
        - 
        """
        self.true_df["misinformation"] = 0
        self.false_df["misinformation"] = 1
        self.df = pd.concat(self.true_df, self.false_df)

    def prepare_data_single_input(self):
        """
        TODO:
        - finish coding data preparation
        - determine what X and y are
        - 
        """
        self.df = pd.DataFrame(self.data)
    
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
    dataset = MisinformationDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader

# Example usage
if __name__ == '__main__':
    datasets_home_dir = ''
    true_filepath = 'datasets/DataSet_Misinfo_TRUE.csv'
    fake_filepath = 'datasets/DataSet_Misinfo_FALSE.csv'
    loader = get_loader(true_filepath, fake_filepath, batch_size=64)
    

