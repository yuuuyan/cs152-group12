# adapted from https://stackoverflow.com/questions/43697240/how-can-i-split-a-dataset-from-a-csv-file-for-training-and-testing
import pandas as pd
import numpy as np
from datasets import DisinformationDataset
from sklearn.model_selection import train_test_split


dataset = DisinformationDataset(true_filepath="./Datasets/DataSet_Misinfo_TRUE.csv", false_filepath="./Datasets/DataSet_Misinfo_FAKE.csv")
shuffled_data = dataset.df.sample(frac=1, random_state=42).reset_index(drop=True)
train, test = train_test_split(shuffled_data, test_size=0.05)

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
