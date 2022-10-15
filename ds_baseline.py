import os
from pickletools import optimize
import sys
import signal
import tqdm
import math

from model import get_gptj_model 
sys.path.append("./")
sys.path.append("../")
sys.path.append("../MobiusNeuron-Benchmark")
from multiprocessing import Process,freeze_support

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
# import torch.distributed.mpu

import time
import subprocess 
import argparse

import deepspeed
import deepspeed.runtime.utils as ds_utils
from deepspeed.ops.adam import DeepSpeedCPUAdam # teh optimizer of ds 
from deepspeed.runtime.pipe import ProcessTopology
from deepspeed.runtime.pipe.topology import PipelineParallelGrid

# model
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model
from transformers import GPTJConfig

from torchtext.datasets import WikiText2
import pandas as pd
from GPTJ.modeling_gptj import GPTJForCausalLM
# from transformers import GPTJForCausalLM

from torchmobius.utils import print_memory_usage, model_config, setup_seed, debug_condition, clip_grad_norm_

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

my_mpu = None

def init_model(model_config, dataset, config_path='./ds_config_gpt_j.json'):
    my_mpu = PipelineParallelGrid()
    
    model = GPTJForCausalLM.from_pretrained("gpt-j-6B", config=model_config)
    model = nn.Sequential(*(model.to_layers()))
    model = model.half()
    
    deepspeed.zero.Init(module=model,
                        mpu=my_mpu,
                        remote_device='cpu', # initial device to store weights
                        enabled=True, # if F, this context has no effect
                        pin_memory=True, # potentially increase performance
                        config_dict_or_path=config_path)
    
    # optimizer = DeepSpeedCPUAdam(model.parameters())
    # lr = 5 # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=3e-7)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = deepspeed.runtime.lr_schedules.OneCycle(optimizer, cycle_min_lr=0.0005, cycle_max_lr=0.1)

    model, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=config_path,
        training_data=dataset,
        lr_scheduler=scheduler,
        dist_init_required=False
    )

    return model, optimizer, train_dataloader, lr_scheduler

train_iter, val_iter, test_iter = WikiText2()
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
    bptt = 512
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].view(-1)
    target = source[i+1:i+1+seq_len].view(-1)
    # Need batch dimension first for pipeline parallelism.
    return data, target

class WT2_Dataset2(torch.utils.data.Dataset):
    def __init__(self, data, bsz, bptt):
        # self.data = batchify(encoding_data(data), 1024)
        # Here the data is a sequence of encoded text.
        # Pretend to have a batch dimension.
        _train_data = encoding_data(data)
        train_data = batchify(_train_data, bsz)

        self.inputs = []
        self.labels = []
        self.nbatch = 600
        for i in range(self.nbatch):
            input_ids, tgt = get_batch(train_data, i)
            self.inputs.append(input_ids)
            self.labels.append(tgt)

    def __len__(self):
        return self.nbatch
    
    def __getitem__(self, idx): 
        return self.inputs[idx], self.labels[idx]

def train(model, train_dl, optimizer, lr_scheduler, config, criterion, epochs=1):
    print("Start training...")
    # criterion = nn.CrossEntropyLoss()
    # num_of_batches = len(trainloader)

    # for batch_id, data in enumerate(train_dl):
    #     input_ids, tgt = data
    #     break

    model.train()
    for epoch in tqdm.tqdm(range(epochs)):
        for batch_id, data in enumerate(train_dl):
            input_ids, tgt = data

            input_ids = input_ids.to(torch.cuda.current_device())
            tgt = tgt.to(torch.cuda.current_device())

            outputs = model(input_ids)
            torch.cuda.synchronize()
            loss = criterion(outputs.view(-1, config.vocab_size), tgt.view(-1))
            
            print('| epoch {:3d} | step {:3d} | loss {:5.2f} '.format(epoch, batch_id, loss))
            
            model.backward(loss)
            torch.cuda.synchronize()
            
            # ds_utils.clip_grad_norm_(model.parameters(), max_norm=0.1, mpu=my_mpu)
            
            model.step()
            
            lr_scheduler.step()

    print('Finished Training')

if __name__ == '__main__':
    setup_seed(2021)
    deepspeed.init_distributed(verbose=False)

    train_iter = WikiText2(split='train')
    dataset = WT2_Dataset2(train_iter, 1, 512)

    CONFIG = GPTJConfig.from_pretrained('gpt-j-6B')
    model, optimizer, train_dl, lr_scheduler = init_model(CONFIG, dataset)

    criterion = nn.CrossEntropyLoss()
    
    train(model, train_dl, optimizer, lr_scheduler, CONFIG, criterion, epochs=30)

