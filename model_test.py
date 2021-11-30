#/usr/bin/env python
# -*- coding: utf-8 -*-
import os, pathlib, sys,logging,string,csv,json,tempfile,re

sys.path.append(os.getcwd())

parent_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(parent_path))

master_path = parent_path.parent
sys.path.append(str(master_path))

project_path = master_path.parent
sys.path.append(str(project_path))


from itertools import chain
from typing import Union,Dict,List,Tuple,Optional
from collections import OrderedDict,Counter

import time
from datetime import date,datetime,timedelta
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import cleantext


####-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def words2indices(origin, vocab):
    if isinstance(origin[0], list):
        result = [[vocab[w] for w in sent] for sent in origin]
    else:
        result = [vocab[w] for w in origin]
    return result

def indices2words(origin, vocab):
    if isinstance(origin[0], list):
        result = [[vocab.id2word(w) for w in sent] for sent in origin]
    else:
        result = [vocab.id2word(w) for w in origin]
    return result

####----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Vocab:
    def __init__(self, word2id, id2word):
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'
        self.START = '<START>'
        self.END = '<END>'
        self.__word2id = word2id
        self.__id2word = id2word

    def get_word2id(self):
        return self.__word2id

    def get_id2word(self):
        return self.__id2word

    def __getitem__(self, item):
        if self.UNK in self.__word2id:
            return self.__word2id.get(item, self.__word2id[self.UNK])
        return self.__word2id[item]

    def __len__(self):
        return len(self.__word2id)

    def id2word(self, idx):
        return self.__id2word[idx]

    @staticmethod
    def build(data, max_dict_size, freq_cutoff, is_tags):

        word_counts = Counter(chain(*data))
        valid_words = [w for w, d in word_counts.items() if d >= freq_cutoff]
        valid_words = sorted(valid_words, key=lambda x: word_counts[x], reverse=True)
        valid_words = valid_words[: max_dict_size]
        valid_words += ['<PAD>']
        word2id = {w: idx for idx, w in enumerate(valid_words)}
        if not is_tags:
            word2id['<UNK>'] = len(word2id)
            valid_words += ['<UNK>']
        return Vocab(word2id=word2id, id2word=valid_words)

    def save(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({'word2id': self.__word2id, 'id2word': self.__id2word}, f, ensure_ascii=False)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            entry = json.load(f)
        return Vocab(word2id=entry['word2id'], id2word=entry['id2word'])

####----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def pad(data, padded_token, device):
    lengths = [len(sent) for sent in data]
    max_len = lengths[0]
    padded_data = []
    for s in data:
        padded_data.append(s + [padded_token] * (max_len - len(s)))
    return torch.tensor(padded_data, device=device), lengths

####----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class BiLSTMCRF(nn.Module):
    def __init__(self, sent_vocab, tag_vocab, dropout_rate=0.5, embed_size=256, hidden_size=256):
        super(BiLSTMCRF, self).__init__()
        self.dropout_rate = dropout_rate
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.sent_vocab = sent_vocab
        self.tag_vocab = tag_vocab
        self.embedding = nn.Embedding(len(sent_vocab), embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, bidirectional=True)
        self.hidden2emit_score = nn.Linear(hidden_size * 2, len(self.tag_vocab))
        self.transition = nn.Parameter(torch.randn(len(self.tag_vocab), len(self.tag_vocab)))  # shape: (K, K)

    def forward(self, sentences, tags, sen_lengths):
        mask = (sentences != self.sent_vocab[self.sent_vocab.PAD]).to(self.device)  # shape: (b, len)
        sentences = sentences.transpose(0, 1)  # shape: (len, b)
        sentences = self.embedding(sentences)  # shape: (len, b, e)
        emit_score = self.encode(sentences, sen_lengths)  # shape: (b, len, K)
        loss = self.cal_loss(tags, mask, emit_score)  # shape: (b,)
        return loss

    def encode(self, sentences, sent_lengths):
        padded_sentences = pack_padded_sequence(sentences, sent_lengths)
        hidden_states, _ = self.encoder(padded_sentences)
        hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True)  # shape: (b, len, 2h)
        emit_score = self.hidden2emit_score(hidden_states)  # shape: (b, len, K)
        emit_score = self.dropout(emit_score)  # shape: (b, len, K)
        return emit_score

    def cal_loss(self, tags, mask, emit_score):
        batch_size, sent_len = tags.shape
        # calculate score for the tags
        score = torch.gather(emit_score, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)  # shape: (b, len)
        score[:, 1:] += self.transition[tags[:, :-1], tags[:, 1:]]
        total_score = (score * mask.type(torch.float)).sum(dim=1)  # shape: (b,)
        # calculate the scaling factor
        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, sent_len):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  # shape: (uf, 1, K)
            emit_and_transition = emit_score[: n_unfinished, i].unsqueeze(dim=1) + self.transition  # shape: (uf, K, K)
            log_sum = d_uf.transpose(1, 2) + emit_and_transition  # shape: (uf, K, K)
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)  # shape: (uf, 1, K)
            log_sum = log_sum - max_v  # shape: (uf, K, K)
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)  # shape: (uf, 1, K)
            d = torch.cat((d_uf, d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)  # shape: (b, K)
        max_d = d.max(dim=-1)[0]  # shape: (b,)
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)  # shape: (b,)
        llk = total_score - d  # shape: (b,)
        loss = -llk  # shape: (b,)
        return loss

    def predict(self, sentences, sen_lengths):
        batch_size = sentences.shape[0]
        mask = (sentences != self.sent_vocab[self.sent_vocab.PAD])  # shape: (b, len)
        sentences = sentences.transpose(0, 1)  # shape: (len, b)
        sentences = self.embedding(sentences)  # shape: (len, b, e)
        emit_score = self.encode(sentences, sen_lengths)  # shape: (b, len, K)
        tags = [[[i] for i in range(len(self.tag_vocab))]] * batch_size  # list, shape: (b, K, 1)
        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, sen_lengths[0]):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  # shape: (uf, 1, K)
            emit_and_transition = self.transition + emit_score[: n_unfinished, i].unsqueeze(dim=1)  # shape: (uf, K, K)
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition  # shape: (uf, K, K)
            d_uf, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()  # list, shape: (nf, K)
            tags[: n_unfinished] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)]
            d = torch.cat((torch.unsqueeze(d_uf, dim=1), d[n_unfinished:]), dim=0)  # shape: (b, 1, K)
        d = d.squeeze(dim=1)  # shape: (b, K)
        _, max_idx = torch.max(d, dim=1)  # shape: (b,)
        max_idx = max_idx.tolist()
        tags = [tags[b][k] for b, k in enumerate(max_idx)]
        return tags

    def save(self, filepath):
        params = { 'sent_vocab': self.sent_vocab,  'tag_vocab': self.tag_vocab, 'args': dict(dropout_rate=self.dropout_rate, embed_size=self.embed_size, hidden_size=self.hidden_size),
            'state_dict': self.state_dict() }
        torch.save(params, filepath)

    @staticmethod
    def load(filepath, device_to_load):
        params = torch.load(filepath, map_location=lambda storage, loc: storage)
        model = BiLSTMCRF(params['sent_vocab'], params['tag_vocab'], **params['args'])
        model.load_state_dict(params['state_dict'])
        model.to(device_to_load)
        return model

    @property
    def device(self):
        return self.embedding.weight.device
####----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_chunk_type(tag_name):

   # tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type

def get_chunks(seq):

    default = "O"#tags["O"]
   # idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None

    for i, tok in enumerate(seq):
        if tok == default:
            if chunk_type is not None:
                chunk_type, chunk_start = None, None
            else:
                pass
        else:
            if re.search('\-',tok):
                tok_class, tok_chunk_type = tok.split('-')[0],tok.split('-')[1]
                if tok_class == 'S':
                    chunk = (tok_chunk_type, i, i+1)
                    chunks.append(chunk)
                    chunk_type, chunk_start = None, None
                if tok_class == 'B':
                    chunk_start = i
                    chunk_type = tok_chunk_type
                if tok_class == 'I':
                    if chunk_type is not None:
                        if chunk_type == tok_chunk_type:
                            pass
                        else:
                            chunk_type, chunk_start = None, None
                    else:
                        pass
                if tok_class == 'E':
                    if chunk_type is not None:
                        if chunk_type == tok_chunk_type:
                            chunk = (chunk_type, chunk_start, i+1)
                            chunks.append(chunk)
                            chunk_type, chunk_start = None, None
                        else:
                            chunk_type, chunk_start = None, None
                    else:
                        pass
    return chunks


###--------------------------------------------------


ner_eng_model_file=r'C:\Users\marcus\PycharmProjects\ipo_sponsor_NER\models\model.pth'
ner_eng_sen_vocabfile=r'C:\Users\marcus\PycharmProjects\ipo_sponsor_NER\models\sent_vocab.json'
ner_eng_tag_vocabfile=r'C:\Users\marcus\PycharmProjects\ipo_sponsor_NER\models\tag_vocab.json'
eng_max_len=100


eng_sent_vocab = Vocab.load(ner_eng_sen_vocabfile)
eng_tag_vocab = Vocab.load(ner_eng_tag_vocabfile)

device = torch.device('cpu' )
ner_eng_model = BiLSTMCRF.load(filepath=ner_eng_model_file,device_to_load=device)
#
print('start testing...')
ner_eng_model.eval()
print('using device', device)


def scan_ner(text:str):
    with torch.no_grad():
        text=text.replace('. .','')
        text=text.replace('*','')
        ori_tokens = [w for w in text.split(' ') if w !='']

        tokens = ['<START>'] + ori_tokens + ['<END>']

        tokens_idx = words2indices([tokens], eng_sent_vocab)[0]

        lengths = len(tokens_idx)

        padded_data = tokens_idx + [eng_sent_vocab[eng_sent_vocab.PAD]] * (eng_max_len - len(tokens_idx))
        padded_tokens_idx, tokens_idx_len = torch.tensor([padded_data], device=device), [lengths]
        pred_tag_idx = ner_eng_model.predict(padded_tokens_idx, tokens_idx_len)[0][1:-1]
        pred_tags = [eng_tag_vocab.id2word(p) for p in pred_tag_idx]
        return ori_tokens,pred_tags

# ###---------------------------------------------------------------------------------------------------------------------------------

