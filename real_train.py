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
from torchmobius.utils import setup_seed, clip_grad_norm_

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

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"]) 

def data_process(raw_text_iter):
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

def batchify(data, bsz):
    device = torch.device("cuda:0")
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 4
eval_batch_size = 1
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

bptt = 512
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    # Need batch dimension first for pipeline parallelism.
    return data.t(), target

class WT2_Dataset2(torch.utils.data.Dataset):
    def __init__(self, data, bsz, bptt):
        # self.data = batchify(encoding_data(data), 1024)
        # Here the data is a sequence of encoded text.
        # Pretend to have a batch dimension.
        _train_data = data_process(data)
        train_data = batchify(_train_data, bsz)

        self.inputs = []
        self.labels = []
        self.nbatch = 200
        for i in range(self.nbatch):
            input_ids, tgt = get_batch(train_data, i)
            self.inputs.append(input_ids)
            self.labels.append(tgt)

    def __len__(self):
        return self.nbatch
    
    def __getitem__(self, idx): 
        return self.inputs[idx], self.labels[idx]

def train_step(model, train_dl, start_step, optimizer, criterion, scheduler, num_gpus=8, epoch=1, log_interval = 1, fp16=False):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(vocab)

    # Train only for 50 batches to keep script execution time low.
    nbatches = min(50 * bptt, train_data.size(0) - 1)

    step = start_step
    # for batch_id, i in enumerate(range(0, nbatches, bptt)):
    for batch_id, data in enumerate(train_dl):
        step = step + 1
        data, targets = get_batch(train_data, i)
        # print("Data shape: ", data.shape)
        # print("Targets shape: ", targets.shape)
        if fp16:
            # ------------------- Step --------------------
            # output = model(data).local_value().cuda(num_gpus - 1)
            output = model(data)
            # Need to move targets to the device where the output of the
            # pipeline resides.
            # ----------------------------------------------
            # print("Output Shape : ", output.shape)
            # print("After View Shape : ", output.view(-1, config.vocab_size).shape)
            # print("Target Shape : ", targets.shape)
            loss = criterion(output.view(-1, config.vocab_size), targets.cuda(output.device))
            loss.backward()
            
            # fyy's advice
            torch.cuda.synchronize()
            
            # print(model.parameters()[321].grad)
            clip_grad_norm_(model.parameters(), 0.1)
            
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)      
                  
            optimizer.step()
            optimizer.zero_grad()
            
        else:
            # ------------------- Step --------------------
            # output = model(data).local_value().cuda(num_gpus - 1)
            output = model(data)
            # Need to move targets to the device where the output of the
            # pipeline resides.
            print(targets.shape)
            loss = criterion(output.view(-1, config.vocab_size), targets.to(output.device))
            loss.backward()
            
            # fyy's advice
            torch.cuda.synchronize()
            
            # print(model.parameters()[321].grad)
    
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()
            # ----------------------------------------------

        # print(loss)
        log_dict = {
            "train_loss": loss,
            "time_step": step,
        }
        wandb.log(log_dict)
        print('| epoch {:3d} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch_id, loss, math.exp(loss)))

        # total_loss += loss.item()
        # if batch_id % log_interval == 0 and batch > 0:
        #     cur_loss = total_loss / log_interval
        #     elapsed = time.time() - start_time
        #     print('| epoch {:3d} | {:5d}/{:5d} batches | '
        #           'lr {:02.2f} | ms/batch {:5.2f} | '
        #           'loss {:5.2f} | ppl {:8.2f}'.format(
        #             epoch, batch, nbatches // bptt, scheduler.get_lr()[0],
        #             elapsed * 1000 / log_interval,
        #             cur_loss, math.exp(cur_loss)))
        #     total_loss = 0
        #     start_time = time.time()
        print('lr {} | loss {:5.2f}'.format(scheduler.get_lr()[0], loss.item()))
        break

    return step
                        
def evaluate(eval_model, config, data_source, criterion, num_gpus=8):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    # ntokens = len(vocab)
    # Evaluate only for 50 batches to keep script execution time low.
    nbatches = min(50 * bptt, data_source.size(0) - 1)
    with torch.no_grad():
        for i in range(0, nbatches, bptt):
            data, targets = get_batch(data_source, i)
            # output = eval_model(data).local_value().cuda(num_gpus - 1)
            output = eval_model(data).cuda(num_gpus - 1)
            output_flat = output.view(-1, config.vocab_size)
            # Need to move targets to the device where the output of the
            # pipeline resides.
            total_loss += len(data) * criterion(output_flat, targets.cuda(num_gpus - 1)).item()
    return total_loss / (len(data_source) - 1)

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
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    config = GPTJConfig.from_pretrained('EleutherAI/gpt-j-6B')

    # ntokens = len(vocab) # the size of vocabulary
    # emsize = 4096 # embedding dimension
    # nhid = 4096 # the dimension of the feedforward network model in nn.TransformerEncoder
    # nlayers = 32 # the number of nn.Transformer
    # nhead = 16 # the number of heads in the multiheadattention models
    # dropout = 0.2 # the dropout value

    FP16_mode = True
    # Build the pipeline.
    # model = get_gptj_pipemodel(config, 32)
    model = get_gptj_mobiusmodel(config, 32, fp16=FP16_mode)
    # model = get_gptj_gpipemodel(config, 32, fp16=FP16_mode)

    criterion = nn.CrossEntropyLoss()
    lr = 5 # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float("inf")
    epochs = 64 # The number of epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=train_data.size(0) // batch_size, epochs=epochs)
    best_model = None

    train_set = WT2_Dataset2(train_iter, 1, 512)
    train_dl = DataLoader(dataset=train_set, batch_size=4, num_workers=1, shuffle=False)

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

