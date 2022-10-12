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
