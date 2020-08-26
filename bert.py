# This file is adapted from from May et al. 2019: "On Measuring Bias in Sentence Encoders"
# File: https://github.com/W4ngatang/sent-bias/blob/master/sentbias/encoders/bert.py


import torch
import pytorch_pretrained_bert as bert
from transformers import AutoTokenizer


MODELS = {
    'bert': [
        'bert-base-german-cased',
        'bert-base-german-dbmdz-cased',
        'bert-base-german-dbmdz-uncased',
        'distilbert-base-german-cased'
    ],
    'xlm': [
        'xlm-mlm-ende-1024',
        'xlm-clm-ende-1024'
    ]
}



TOKENIZERS = {
    'bert-base-german-cased': 'bert-base-german-cased',
    'dbmdz/bert-base-german-cased',
    'bert-base-german-dbmdz-cased',
    'severinsimmler/literary-german-bert',
    'oliverguhr/german-sentiment-bert'
}


def load_tokenizer(version):
    return AutoTokenizer.from_pretrained(TOKENIZERS[version])


def load_model(version):
    """Load BERT model and corresponding tokenizer."""
    tokenizer = bert.BertTokenizer.from_pretrained(version)
    model = bert.BertModel.from_pretrained(version)
    model.eval()
    return model, tokenizer


def encode(model, tokenizer, texts):
    """Use tokenizer and model to encode texts."""
    encs = {}
    for text in texts:
        tokenized = tokenizer.tokenize(text)
        indexed = tokenizer.convert_tokens_to_ids(tokenized)
        segment_idxs = [0] * len(tokenized)
        tokens_tensor = torch.tensor([indexed])
        segments_tensor = torch.tensor([segment_idxs])
        enc, _ = model(tokens_tensor, segments_tensor, output_all_encoded_layers=False)

        enc = enc[:, 0, :]  # extract the last rep of the first input
        encs[text] = enc.detach().view(-1).numpy()
    return encs