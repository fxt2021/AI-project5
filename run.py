import argparse
import os
import random
import numpy as np
import torch
from torch import nn

from model.bert import load_bert_tokenizer
from model.mutil_model import MutilModelClassifier
from utils.data import get_data, get_test
from utils.train import train_model
from utils.inference import inference

parser = argparse.ArgumentParser(description='choose fusion level and hyper parameters for mutil model classification')
parser.add_argument('--train', action='store_true', default=False, help='train and then infer or infer directly')
parser.add_argument('--fusion_level', default='feature', help='choose model fusion level, feature or decision',
                    choices=['feature', 'decision'])
parser.add_argument('--learning_rate', '-lr', default=1e-3, type=float, help='choose learning rate')
parser.add_argument('--batch_size', '-bs', default=64, type=int, help='choose batch size')
parser.add_argument('--num_epochs', '-n', default=20, type=int, help='choose number of epochs')
parser.add_argument('--gpu', '-g', action='store_true', default=False, help='use gpu or not')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='save model parameters for inference or not')
args = parser.parse_args()

fusion_level = args.fusion_level
learning_rate = args.learning_rate
batch_size = args.batch_size
num_epochs = args.num_epochs
device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
save = args.save_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train():
    set_seed(seed=1566)
    tokenizer = load_bert_tokenizer()
    print('start loading train data')
    train_dataloader, val_dataloader = get_data(tokenizer, batch_size)
    print(f'finish loading train data\nstart training {fusion_level} fusion model')
    model = MutilModelClassifier(fusion_level)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_model(model, train_dataloader, val_dataloader, loss_fn, optimizer, num_epochs, device, save)
    print('finish training')


def infer():
    tokenizer = load_bert_tokenizer()
    model = MutilModelClassifier(fusion_level)
    model.to(device)
    model.load_state_dict(torch.load(f'./trained_model/{fusion_level}_fusion/model.pth', map_location=device))
    print('start loading test data')
    test_dataloader = get_test(tokenizer)
    print(f'finish loading test data\nstart inferring through {fusion_level} fusion model')
    result_path = './result.txt'
    guids, predicts = inference(model, test_dataloader, device)
    with open(result_path, 'w') as f:
        f.write('guid,tag\n')
        for i in range(len(guids)):
            f.write(str(guids[i]) + ',' + str(predicts[i]) + '\n')
    print(f'finish inferring, result saved to {result_path}')


def main():
    if args.train:
        train()
    infer()


if __name__ == '__main__':
    main()
