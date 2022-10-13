from pickletools import optimize
from symbol import parameters
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributed.pipeline.sync import Pipe
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader

from model import Encoder, Decoder, PositionalEncoding, get_gptj_gpipemodel, get_gptj_mobiusmodel, get_gptj_pipemodel
import torchmobius
from torchmobius.utils import mobius_logger, setup_seed, clip_grad_norm_, has_overflow_serial, has_overflow_params
import torchmobius.attribute

# For Datasets !!
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from GPTJ.modeling_gptj import GPTJBlock
from transformers import AutoTokenizer
from GPTJ.modeling_gptj import GPTJBlock, GPTJConfig

from deepspeed.ops.adam import DeepSpeedCPUAdam
import deepspeed.runtime

import wandb

wandb.init(project="test", entity="thu-storage")
# define our custom x axis metric
wandb.define_metric("time_step")
# define which metrics will be plotted against it
wandb.define_metric("train_loss", step_metric="time_step")
start_time = time.time()


# ------------------- Attention -------------------
# Set the random seed manually for reproducibility.
setup_seed(2022)
# ------------------- Attention -------------------

train_iter, val_iter, test_iter = WikiText2()

class WT2_Dataset2(torch.utils.data.Dataset):
    def __init__(self, data, bsz, bptt):
        self.bppt = bptt
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, train_iter), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"]) 

        _train_data = self.encoding_data(data)
        train_data = self.batchify(_train_data, bsz)

        self.inputs = []
        self.labels = []
        self.nbatch = 600
        for i in range(self.nbatch):
            input_ids, tgt = self.get_batch(train_data, i)
            self.inputs.append(input_ids)
            self.labels.append(tgt)

    def encoding_data(self, raw_text_iter):
        data = [torch.tensor(self.vocab(self.tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def batchify(self, data, bsz):
        nbatch = data.size(0) // bsz
        data = data.narrow(0, 0, nbatch * bsz)
        data = data.view(bsz, -1).t().contiguous()
        return data

    def get_batch(self, source, i):
        seq_len = min(self.bppt, len(source) - 1 - i)
        data = source[i:i+seq_len].view(-1)
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

    def __len__(self):
        return self.nbatch
    
    def __getitem__(self, idx): 
        return self.inputs[idx], self.labels[idx]

def train_step(model, train_dl, start_step, optimizer, criterion, scheduler, epoch=1, log_interval = 1, fp16=False):

    for batch_id, data in enumerate(train_dl):
        sample_ids, sample_labels = data

        if fp16:
            # ------------------- Step --------------------
            optimizer.zero_grad()
            output = model(sample_ids.to(model.get_first_device()))
                        
            loss = criterion(output.view(-1, config.vocab_size), sample_labels.view(-1).cuda(output.device))
            mobius_logger(f"Loss : {loss.item()} ")
            
            loss.backward()
            torch.cuda.synchronize()
            
            # if torchmobius.attribute == False:
            #     mobius_logger("GRADIENT OVERFLOW!")
            #     continue
            
            # if has_overflow_serial(model.parameters()):
            #     mobius_logger("GRADIENT OVERFLOW!")
            #     continue
            
            
            # clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()

        else:
            assert False, "Not implemented yet"

        log_dict = {
            "train_loss": loss,
        }
        wandb.log(log_dict)
        mobius_logger('| epoch {:3d} | step {:3d} | loss {:5.2f} '.format(epoch, batch_id, loss))
        # break

    return step

if __name__ == '__main__':
    config = GPTJConfig.from_pretrained('EleutherAI/gpt-j-6B')

    FP16_mode = True
    DEVICES = "0,1,2,3"
    # Build the pipeline.
    # model = get_gptj_pipemodel(config, 32)
    model = get_gptj_mobiusmodel(config, DEVICES, 32, fp16=FP16_mode)
    # model = get_gptj_gpipemodel(config, 32, fp16=FP16_mode)

    criterion = nn.CrossEntropyLoss()
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=3e-7)
    optimizer = DeepSpeedCPUAdam(model.parameters(), lr=0.001, betas=[0.8, 0.999])

    best_val_loss = float("inf")
    epochs = 30 # The number of epochs
    scheduler = deepspeed.runtime.lr_schedules.OneCycle(optimizer, cycle_min_lr=0, cycle_max_lr=0.1)

    train_set = WT2_Dataset2(train_iter, 1, 512)
    train_dl = DataLoader(dataset=train_set, batch_size=8, num_workers=1, shuffle=False)


    start_time = time.time()
    step = 0
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        step = train_step(model, train_dl, step, optimizer, criterion, scheduler, epoch=epoch, fp16=FP16_mode)
        scheduler.step()

