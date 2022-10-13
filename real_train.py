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

# For Datasets !!
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from GPTJ.modeling_gptj import GPTJBlock
from transformers import AutoTokenizer
from GPTJ.modeling_gptj import GPTJBlock, GPTJConfig

import wandb

wandb.init(project="test", entity="thu-storage")
# define our custom x axis metric
wandb.define_metric("time_step")
# define which metrics will be plotted against it
wandb.define_metric("train_loss", step_metric="time_step")
start_time = time.time()

if sys.platform == 'win32':
    print('Windows platform is not supported for pipeline parallelism')
    sys.exit(0)
if torch.cuda.device_count() < 2:
    print('Need at least two GPU devices for this tutorial')
    sys.exit(0)

# ------------------- Attention -------------------
# Set the random seed manually for reproducibility.
setup_seed(2022)
# ------------------- Attention -------------------

train_iter, val_iter, test_iter = WikiText2()

# def encoding_data(raw_text_iter):
#     print("Raw data : ", raw_text_iter)
#     data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
#     return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

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
        # bptt = 512
        seq_len = min(self.bppt, len(source) - 1 - i)
        data = source[i:i+seq_len].view(-1)
        target = source[i+1:i+1+seq_len].view(-1)
        # Need batch dimension first for pipeline parallelism.
        return data, target

    def __len__(self):
        return self.nbatch
    
    def __getitem__(self, idx): 
        return self.inputs[idx], self.labels[idx]

def train_step(model, train_dl, start_step, optimizer, criterion, scheduler, epoch=1, log_interval = 1, fp16=False):
    # model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    # ntokens = len(vocab)

    # Train only for 50 batches to keep script execution time low.
    # nbatches = min(50 * bptt, train_data.size(0) - 1)

    sample = None
    for batch_id, data in enumerate(train_dl):
        sample_ids, sample_labels = data
        break

    # for batch_id, i in enumerate(range(0, nbatches, bptt)):
    for batch_id, rawdata in enumerate(train_dl):
        ids, targets = rawdata
        # data, targets = get_batch(train_data, i)
        # print("Data shape: ", sample_ids.shape)
        # print("Targets shape: ", sample_labels.shape)
        if fp16:
            # ------------------- Step --------------------
            # output = model(data)
            optimizer.zero_grad()
            output = model(sample_ids.to(model.get_first_device()))

            loss = criterion(output.view(-1, config.vocab_size), sample_labels.view(-1).cuda(output.device))
            mobius_logger(f"Loss : {loss.item()} ")
            loss.backward()
            
            torch.cuda.synchronize()
            if has_overflow_serial(model.parameters()):
                mobius_logger("OVERFLOW!")
                continue
            
            # clip_grad_norm_(model.parameters(), 0.01)
            # 
            # if has_overflow_params(model.parameters()):
            #     mobius_logger("OVERFLOW before update")
            
            # optimizer.step()
            
            # if has_overflow_params(model.parameters()):
            #     mobius_logger("OVERFLOW after update")

        else:
            assert False, "Not implemented yet"
            # ------------------- Step --------------------
            # output = model(data)
            output = model(sample_ids)

            # loss = criterion(output.view(-1, config.vocab_size), targets.cuda(output.device))
            loss = criterion(output.view(-1, config.vocab_size), sample_labels.cuda(output.device))
            loss.backward()
            
            # fyy's advice
            torch.cuda.synchronize()
            
            # print(model.parameters()[321].grad)
    
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()
            # ----------------------------------------------
        log_dict = {
            "train_loss": loss,
        }
        wandb.log(log_dict)
        mobius_logger('| epoch {:3d} | step {:3d} | loss {:5.2f} '.format(epoch, batch_id, loss))
        # break

    return step

if __name__ == '__main__':
    # -----------------------------------------------------
    # Create RPC backend and initialize distributed training.
    # -----------------------------------------------------
    from torch.distributed import rpc
    tmpfile = tempfile.NamedTemporaryFile()
    rpc.init_rpc(
        name="worker",
        rank=0,
        world_size=1,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="file://{}".format(tmpfile.name),
            # Specifying _transports and _channels is a workaround and we no longer
            # will have to specify _transports and _channels for PyTorch
            # versions >= 1.8.1
            _transports=["ibv", "uv"],
            _channels=["cuda_ipc", "cuda_basic"],
        )
    )
    config = GPTJConfig.from_pretrained('EleutherAI/gpt-j-6B')

    # ntokens = len(vocab) # the size of vocabulary
    # emsize = 4096 # embedding dimension
    # nhid = 4096 # the dimension of the feedforward network model in nn.TransformerEncoder
    # nlayers = 32 # the number of nn.Transformer
    # nhead = 16 # the number of heads in the multiheadattention models
    # dropout = 0.2 # the dropout value

    FP16_mode = True
    DEVICES = "0,1,2,3"
    # Build the pipeline.
    # model = get_gptj_pipemodel(config, 32)
    model = get_gptj_mobiusmodel(config, DEVICES, 32, fp16=FP16_mode)
    # model = get_gptj_gpipemodel(config, 32, fp16=FP16_mode)

    criterion = nn.CrossEntropyLoss()
    lr = 5 # learning rate
    data_len = 200 # The number of batches
    batch_size = 4 # batch size

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=3e-7)

    best_val_loss = float("inf")
    epochs = 64 # The number of epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=data_len // batch_size, epochs=epochs)

    train_set = WT2_Dataset2(train_iter, 1, 512)
    train_dl = DataLoader(dataset=train_set, batch_size=4, num_workers=1, shuffle=False)

    # for data in train_dl:
    #     sample_ids, sample_labels = data
    #     print(sample_ids.shape)
    #     print(sample_labels.shape)
    #     print(data)
    #     break

    start_time = time.time()
    step = 0
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        step = train_step(model, train_dl, step, optimizer, criterion, scheduler, epoch=epoch, fp16=FP16_mode)
        
        ''' TODO(fyy) NOT IMPL INFERENCE
        val_loss = evaluate(model, config, val_data, criterion)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
        '''

        scheduler.step()

