from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

class MisinformationDataset(Dataset):
    def __init__(self, true_filepath, false_filepath, filepath=None):
        """
        Initialize the dataset.
        :param filepath: Path to the CSV file containing the disformation data.
        """

        if filepath:
            self.data = pd.read_csv(filepath, sep='\t')
            self.prepare_data()
        else:
            self.true_df = pd.DataFrame(pd.read_csv(true_filepath))
            self.false_df = pd.DataFrame(pd.read_csv(false_filepath))
            self.prepare_data_dual_input()

    def prepare_data_dual_input(self):
        self.true_df.drop(columns = 'Unnamed: 0', axis = 1, inplace = True)
        self.false_df.drop(columns = 'Unnamed: 0', axis = 1, inplace = True)
        self.true_df["misinformation"] = 0
        self.false_df["misinformation"] = 1
        self.df = pd.concat((self.true_df, self.false_df))
        self.df.dropna(how = 'any', inplace = True)

    def prepare_data_single_input(self):
        self.df = pd.DataFrame(self.data)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        sample = self.df.iloc[index]
        label = torch.tensor(sample["misinformation"])
        text = sample["text"]
        return {"text": text, "label": label}

def get_loader(filepath, batch_size=32, shuffle=True):
    """
    Create data loader.
    :param filepath: Path to the CSV data file.
    :param batch_size: Batch size.
    :param shuffle: Whether to shuffle the data.
    """
    dataset = MisinformationDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader

# Example usage
# if __name__ == '__main__':
#     datasets_home_dir = ''
#     true_filepath = 'datasets/DataSet_Misinfo_TRUE.csv'
#     fake_filepath = 'datasets/DataSet_Misinfo_FALSE.csv'
#     loader = get_loader(true_filepath, fake_filepath, batch_size=64)    dataset = MisinformationDataset()
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # return loader

# Example usage
# if __name__ == '__main__':
#     datasets_home_dir = ''
#     true_filepath = 'datasets/DataSet_Misinfo_TRUE.csv'
#     fake_filepath = 'datasets/DataSet_Misinfo_FALSE.csv'
#     loader = get_loader(true_filepath, fake_filepath, batch_size=64)
    

