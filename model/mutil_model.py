import torch
from torch import nn

from model.vgg import *
from model.bert import *


class MutilModelClassifier(nn.Module):
    def __init__(self, fusion_level):
        super(MutilModelClassifier, self).__init__()
        self.fusion_level = fusion_level
        if self.fusion_level == 'feature':
            self.bert = load_bert()
            self.vgg = VGG16Feature()
            self.fc = nn.Sequential(
                nn.Linear(768 + 4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 3),
            )
        elif self.fusion_level == 'decision':
            self.bert = BertClassifier()
            self.vgg = VGG16Classifier()
        self.freeze_bert_param()

    def forward(self, pic, input_ids, token_type_ids, attention_mask, mode='all', device=None):
        if self.fusion_level == 'feature':
            batch_size = pic.shape[0]
            if mode == 'all':
                bert_out = self.bert(input_ids, token_type_ids, attention_mask)
                vgg_feature = self.vgg(pic)
                return self.fc(torch.cat((bert_out.last_hidden_state[:, 0], vgg_feature), dim=1))
            elif mode == 'picture':
                vgg_feature = self.vgg(pic)
                return self.fc(torch.cat((torch.zeros(size=(batch_size, 768)).to(device), vgg_feature), dim=1))
            elif mode == 'text':
                bert_out = self.bert(input_ids, token_type_ids, attention_mask)
                return self.fc(
                    torch.cat((bert_out.last_hidden_state[:, 0], torch.zeros(size=(batch_size, 4096)).to(device)),
                              dim=1))
        elif self.fusion_level == 'decision':
            if mode == 'all':
                return self.bert(input_ids, token_type_ids, attention_mask) + self.vgg(pic)
            elif mode == 'picture':
                return self.vgg(pic)
            elif mode == 'text':
                return self.bert(input_ids, token_type_ids, attention_mask)

    def freeze_bert_param(self):
        if self.fusion_level == 'feature':
            for param in self.bert.parameters():
                param.requires_grad_(False)
        if self.fusion_level == 'decision':
            for param in self.bert.feature.parameters():
                param.requires_grad_(False)
