import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from transformers import BertTokenizer

from utils import OrderedCounter

from common.file_handler import save2pickle, load_pickle
from joblib import load as job_load

MAX_TOKEN_LEN = 100 + 1

class Tokenizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __call__(self, text):
        return self.tokenize(text)

    def tokenize(self, text):
        return self.tokenizer.encode_plus(
            text=text,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_TOKEN_LEN,                  # Max length to truncate/pad
            truncation=True,
            padding='max_length',
            # pad_to_max_length=True,         # Pad sentence to max length
            return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
            )
    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)

    def convert_tokens_to_string(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens)

class VAETokenizer(Tokenizer):
    def __call__(self, text, **kwargs):
        # discard the kwargs

        # single input
        if isinstance(text, str):
            encoded_sent = self.tokenize(text)
            words = encoded_sent['input_ids']
            input_seq = words[:, :-1]
            length = (encoded_sent['attention_mask'] == 1).sum(axis=-1) - 1
            return {
                'input': input_seq
                , 'length': length
            }

        # batch input
        input_seqs, lengths = self.batch_tokenize(text)

        return {
            'input': input_seqs
            , 'length': lengths
        }
    def batch_tokenize(self, batch_text):
        encoded_sents = self.tokenizer.batch_encode_plus(
            batch_text,
            add_special_tokens=True,
            max_length=MAX_TOKEN_LEN,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True
        )
        input_seqs = encoded_sents['input_ids'][:,:-1]
        lengths = (encoded_sents['attention_mask'] == 1).sum(dim=1) - 1
        
        return input_seqs, lengths

class InputDataset(Dataset):

    def __init__(self, data_dir=None, raw_data_filename=None, split=None, create_data=None, online=False, **kwargs):

        super().__init__()
        if online:
            self.max_sequence_length = kwargs.get('max_sequence_length', 50)
            self.tokenizer = Tokenizer()

        else:
            self.data_dir = data_dir
            self.split = split
            self.max_sequence_length = kwargs.get('max_sequence_length', 50)
            # self.min_occ = kwargs.get('min_occ', 3)

            self.raw_data_path = os.path.join(data_dir, f"{raw_data_filename}.{split}.pickle")
            self.data_file = os.path.join(data_dir, 'processed_'+f"{raw_data_filename}.{split}.pickle")
            # self.vocab_file = 'ptb.vocab.json'
            
            self.tokenizer = Tokenizer()

            if create_data:
                print("Creating new %s ptb data."%split.upper())
                self._create_data()

            elif not os.path.exists(self.data_file):
                print("%s preprocessed file not found at %s. Creating new."%(split.upper(), os.path.join(self.data_dir, self.data_file)))
                self._create_data()

            else:
                self._load_data()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # idx = str(idx)
        # print(self.data[idx])

        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length']
        }

    @property
    def vocab_size(self):
        return self.tokenizer.tokenizer.vocab_size

    @property
    def pad_idx(self):
        return self.tokenizer.tokenizer.pad_token_id

    @property
    def sos_idx(self):
        return self.tokenizer.tokenizer.cls_token_id

    @property
    def eos_idx(self):
        return self.tokenizer.tokenizer.sep_token_id

    @property
    def unk_idx(self):
        return self.tokenizer.tokenizer.unk_token_id

#     def get_w2i(self):
#         return self.w2i

#     def get_i2w(self):
#         return self.i2w


    def _load_data(self):
        self.data = job_load(self.data_file)
        # if vocab:
        #     with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
        #         vocab = json.load(file)
        #     self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

#     def _load_vocab(self):
#         with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
#             vocab = json.load(vocab_file)

#         self.w2i, self.i2w = vocab['w2i'], vocab['i2w']
        
    def _tokenize_sentence(self, sentence):
        "Make sure sentence is truncated before input to this func"
        encoded_sent = self.tokenizer.tokenize(sentence)
        words = encoded_sent['input_ids'][0]
        input_seq = words[:-1]
        target = words[1:]
        length = (encoded_sent['attention_mask'] == 1).sum().item() - 1
        return input_seq, target, length


    def _construct_data(self, file):
        data = defaultdict(dict)
        
        for i, line in enumerate(file):
            if len(line) <= 0:
                continue
            assert len(line) > 0

            # encoded_sent = self.tokenizer.tokenize(line)
            
            # words = encoded_sent['input_ids'][0]
            
            # input_seq = words[:-1]
            # target = words[1:]


#                 input = ['<sos>'] + words
#                 input = input[:self.max_sequence_length]

#                 target = words[:self.max_sequence_length-1]
#                 target = target + ['<eos>']

            # assert len(input) == len(target), "%i, %i"%(len(input), len(target))
            # length = (encoded_sent['attention_mask'] == 1).sum().item() - 1
            input_seq, target, length = self._tokenize_sentence(line[:self.max_sequence_length - 1])  # -1 for <eos>/<sos>

            # input.extend(['<pad>'] * (self.max_sequence_length-length))
            # target.extend(['<pad>'] * (self.max_sequence_length-length))

            # input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
            # target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

            data_id = len(data)
            data[data_id]['input'] = input_seq
            data[data_id]['target'] = target
            data[data_id]['length'] = length
        return data

    def _create_data(self):

        # if self.split == 'train':
        #     self._create_vocab()
        # else:
        #     self._load_vocab()

        # tokenizer = TweetTokenizer(preserve_case=False)
        file = job_load(self.raw_data_path)
        
        save2pickle(self._construct_data(file), self.data_file)
        # with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
        #     data = json.dumps(data, ensure_ascii=False)
        #     data_file.write(data.encode('utf8', 'replace'))
        

        self._load_data()

#     def _create_vocab(self):

#         assert self.split == 'train', "Vocablurary can only be created for training file."

#         # tokenizer = TweetTokenizer(preserve_case=False)

#         w2c = OrderedCounter()
#         w2i = dict()
#         i2w = dict()

#         special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
#         for st in special_tokens:
#             i2w[len(w2i)] = st
#             w2i[st] = len(w2i)

#         with open(self.raw_data_path, 'r') as file:

#             for i, line in enumerate(file):
#                 words = self.tokenizer.tokenize(line)
#                 w2c.update(words)

#             for w, c in w2c.items():
#                 if c > self.min_occ and w not in special_tokens:
#                     i2w[len(w2i)] = w
#                     w2i[w] = len(w2i)

#         assert len(w2i) == len(i2w)

#         print("Vocablurary of %i keys created." %len(w2i))

#         vocab = dict(w2i=w2i, i2w=i2w)
#         with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
#             data = json.dumps(vocab, ensure_ascii=False)
#             vocab_file.write(data.encode('utf8', 'replace'))

#         self._load_vocab()
        
        

if __name__ == '__main__':
    ptb = PTB('data', 'self', False)

    
