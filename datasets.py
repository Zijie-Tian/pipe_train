from concurrent.futures import process
import torch

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader

# For Datasets !!
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"]) 

def encoding_data(raw_text_iter):
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data

def get_batch(source, i):
    bptt = 25
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    # Need batch dimension first for pipeline parallelism.
    return data.t(), target

class WT2_Dataset2(torch.utils.data.Dataset):
    def __init__(self, data, bsz, bptt):
        self.bptt = bptt
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, train_iter), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"]) 

        _train_data = self.encoding_data(data)
        train_data = self.batchify(_train_data, bsz)

        self.inputs = []
        self.labels = []
        # -------------------
        # Set the batch is 10000
        self.nbatch = 10000
        # -------------------
        print("train data size: ", train_data.size())
        print("raw data size: ", _train_data.size())
        for i in range(self.nbatch):
            input_ids, tgt = self.get_batch(train_data, i)
            self.inputs.append(input_ids)
            self.labels.append(tgt)

    def encoding_data(self, raw_text_iter):
        data = [torch.tensor(self.vocab(self.tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def batchify(self, data, bsz):
        self.nbatch = data.size(0) // bsz
        # print("before: ", data.size(), bsz, nbatch)
        batched_data = torch.narrow(data, 0, 0, self.nbatch * bsz)
        # print("after: ", batched_data.size(), bsz, nbatch)
        ret = batched_data.view(bsz, -1).t().contiguous()
        return ret

    def get_batch(self, source, i):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len].view(-1)
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

    def __len__(self):
        return self.nbatch
    
    def __getitem__(self, idx): 
        return self.inputs[idx], self.labels[idx]

class WT2_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, bptt):
        self.data = batchify(encoding_data(data), 1)
        self.nbatch = self.data.size(0) // 1
        self.bptt = bptt

    def __len__(self):
        return self.nbatch
    
    def __getitem__(self, idx): 
        bptt = 25
        seq_len = min(bptt, len(self.data) - 1 - idx)
        data = self.data[idx : idx + seq_len]
        target = self.data[idx + 1 : idx + 1 + seq_len].view(-1)
        return data, target

if __name__ == '__main__':
    train_iter = WikiText2(split='train')
    train_set = WT2_Dataset(train_iter, 25)
    for data, target in train_set :
        print(data, target)
        break
    # print(train_data.shape)


if __name__ == '__main__':
    train_iter = WikiText2(split='train')
    train_set = WT2_Dataset2(train_iter, 4, 512)
    for data, target in train_set :
        print(data.shape, target.shape)
        break
    # print(train_data.shape)