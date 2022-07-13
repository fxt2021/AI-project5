import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

DATA_TRAIN_PATH = './dataset/train.txt'
DATA_TEST_PATH = './dataset/test_without_label.txt'

TAG_DICT = {'negative': 0,
            'neutral': 1,
            'positive': 2}


class PictureTextDataset(Dataset):
    def __init__(self, data, tokenizer, test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.test = test

    def __getitem__(self, index):
        text = self.tokenizer(self.data.loc[index]['text'], max_length=128, padding="max_length", truncation=True)
        item = (self.data.loc[index]['picture'],
                torch.tensor(text['input_ids']),
                torch.tensor(text['token_type_ids']),
                torch.tensor(text['attention_mask']))
        if not self.test:
            return *item, self.data.loc[index]['tag']
        else:
            return *item, self.data.loc[index]['guid']

    def __len__(self):
        return len(self.data)


def load_(filepath):
    data = pd.read_csv(filepath)
    data['tag'] = data['tag'].map(TAG_DICT)
    tqdm.pandas(desc='load picture data')
    data['picture'] = data['guid'].progress_apply(
        lambda x: np.asarray(Image.open(f'./dataset/data/{x}.jpg').resize((224, 224)),
                             dtype=np.float32).transpose((2, 0, 1)))
    tqdm.pandas(desc='load text data')
    data['text'] = data['guid'].progress_apply(
        lambda x: open(f'./dataset/data/{x}.txt', 'rb').readline().decode('UTF－8', errors='ignore'))
    return data


def load_train():
    return load_(DATA_TRAIN_PATH)


def load_test():
    return load_(DATA_TEST_PATH)


def get_data(tokenizer, test_size=0.2, batch_size=64):
    data = load_train()
    data_train, data_val = train_test_split(data, test_size=test_size)
    data_train.reset_index(inplace=True)
    data_val.reset_index(inplace=True)
    train_dataset = PictureTextDataset(data_train, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size)
    val_dataset = PictureTextDataset(data_val, tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size)
    return train_dataloader, val_dataloader


def get_test(tokenizer, batch_size=64):
    data = load_test()
    test_dataset = PictureTextDataset(data, tokenizer, test=True)
    test_dataloader = DataLoader(test_dataset, batch_size)
    return test_dataloader
