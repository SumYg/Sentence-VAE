# create a dataset using load_dataset
from torch.utils.data import Dataset
from datasets import load_dataset


class STSDataset(Dataset):
    def __init__(self, tokenizer, split):
        self.dataset = load_dataset('glue', 'stsb')[split].map(lambda x: {
            'label': x['label'] / 5,
            'sentence1': {k: v[0] for k, v in tokenizer(x['sentence1'], padding='max_length', truncation=True, return_tensors='pt').items()},
            'sentence2': {k: v[0] for k, v in tokenizer(x['sentence2'], padding='max_length', truncation=True, return_tensors='pt').items()},
            # , 'sentence1': tokenizer(x['sentence1'], padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
            # , 'sentence2': tokenizer(x['sentence2'], padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
        })
        # tokenize the sentences
        # dataset = dataset.map(lambda x: tokenizer(x['sentence1'], x['sentence2'], padding='max_length', truncation=True, max_length=128, return_tensors='pt'))
        # self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class SST2Dataset(Dataset):
    def __init__(self, tokenizer, split):
        
        # normalize the labels to be in the range [0, 1]
        # dataset = dataset.map(lambda x: {'label': x['label'] / 5
        #     # , 'sentence1': tokenizer(x['sentence1'], padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
        #     # , 'sentence2': tokenizer(x['sentence2'], padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
        # })
        # tokenize the sentences
        # dataset = dataset.map(lambda x: tokenizer(x['sentence1'], x['sentence2'], padding='max_length', truncation=True, max_length=128, return_tensors='pt'))
        # def index():
        #     global i
        #     i += 1
        #     return i
        self.dataset = load_dataset('glue', 'sst2')[split].map(lambda x: {
            'sentence': {k: v[0] for k, v in tokenizer(x['sentence'], padding='max_length', truncation=True, return_tensors='pt').items()},
            # 'sentence': {**tokenizer(x['sentence'], padding='max_length', truncation=True, return_tensors='pt'), 'index': index()},
        })


        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
        
class SNLIDataset(Dataset):
    def __init__(self, tokenizer, split):
        self.dataset = load_dataset('snli', split=split).map(lambda x: {
            'sentence1': {k: v[0] for k, v in tokenizer(x['premise'], padding='max_length', truncation=True, return_tensors='pt').items()},
            'sentence2': {k: v[0] for k, v in tokenizer(x['hypothesis'], padding='max_length', truncation=True, return_tensors='pt').items()},
        }).filter(lambda x: x['label'] != -1)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
