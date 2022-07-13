import torch
from tqdm.auto import tqdm

from utils.data import TAG_DICT

IDX2TAG = dict([val, key] for key, val in TAG_DICT.items())


def inference(model, test_dataloader, device):
    guids = []
    predicts = []
    with torch.no_grad():
        loop = tqdm(test_dataloader, total=len(test_dataloader))
        for batch_x1, *batch_x2, guid in loop:
            model.eval()
            batch_x1 = batch_x1.to(device)
            batch_x2 = [i.to(device) for i in batch_x2]
            pred = model(batch_x1, *batch_x2).argmax(axis=1).cpu().numpy().tolist()
            guids.extend(guid.cpu().numpy().tolist())
            predicts.extend([IDX2TAG[i] for i in pred])
            loop.set_description('test finished percentage')
    return guids, predicts
