from transformers import BertModel, BertTokenizer
from torch import nn


def load_bert_tokenizer():
    try:
        bert_tokenizer = BertTokenizer.from_pretrained('./pretrained-model/bert-base-uncased/tokenizer/')
    except OSError:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_tokenizer.save_pretrained('./pretrained-model/bert-base-uncased/tokenizer/')
    return bert_tokenizer


def load_bert():
    try:
        pretrained_model = BertModel.from_pretrained('./pretrained-model/bert-base-uncased/model/')
    except OSError:
        pretrained_model = BertModel.from_pretrained('bert-base-uncased')
        pretrained_model.save_pretrained('./pretrained-model/bert-base-uncased/model/')
    return pretrained_model


class BertClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(BertClassifier, self).__init__()
        self.feature = load_bert()
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, token_type_ids, attention_mask):
        features = self.feature(input_ids, token_type_ids, attention_mask)
        classifier = self.classifier(features.last_hidden_state[:, 0])
        return classifier
