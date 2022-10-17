from sched import scheduler
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from model import Encoder, Decoder, PositionalEncoding, get_gptj_gpipemodel, get_gptj_mobiusmodel, get_gptj_pipemodel
import torchmobius
from torchmobius.utils import mobius_logger, setup_seed, clip_grad_norm_, has_overflow_serial, has_overflow_params
import torchmobius.attribute

# For Datasets !!
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from GPTJ.modeling_gptj import GPTJConfig

from deepspeed.ops.adam import DeepSpeedCPUAdam
import deepspeed.runtime

import wandb

epochs      =   15  # The number of epochs
bsz         =   16   # Batch size for training

WANDB = True

if WANDB:
    wandb.init(project="mobius", entity="thu-storage",
    config={"epochs": epochs, "batch_size": bsz},
    )
    # wandb.define_metric("train_loss", step_metric="time_step")
start_time = time.time()


# ------------------- Attention -------------------
# Set the random seed manually for reproducibility.
setup_seed(2021)
# ------------------- Attention -------------------

train_iter, val_iter, test_iter = WikiText2()

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
        # Fix the batch is 10000
        self.nbatch = 10000
        # -------------------
        # print("train data size: ", train_data.size())
        # print("raw data size: ", _train_data.size())
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

def train(model, train_dl, optimizer, criterion, scheduler, epochs=1, log_interval = 1, fp16=False):

    
    _step = 0
    for epoch in range(1, epochs + 1):
        for batch_id, data in enumerate(train_dl):
            
            sample_ids, sample_labels = data
            
            optimizer.zero_grad()
            
            output = model(sample_ids.to("cuda:0"))
            torch.cuda.synchronize()
            
            loss = criterion(output.view(-1, config.vocab_size), sample_labels.view(-1).cuda(output.device))
            
            loss.backward()
            torch.cuda.synchronize()
            
            mobius_logger('| epoch {:3d} | step {:3d} | loss {:5.2f} '.format(epoch, batch_id * bsz, loss))
                                            
            if fp16 and has_overflow_serial(model.parameters()):
                mobius_logger("GRADIENT OVERFLOW!")
                continue
                
                # clip_grad_norm_(model.parameters(), 0.01)
                # print(model.parameters()[3].data)
                # A parameter for DeepSpeedCPUAdam.
                # fp16_param_groups=model.parameters()]
            
            if fp16:
                import time
                # A synchronization point for the CUDA stream.
                # torch.cuda.synchronize()
                # clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                # torch.cuda.synchronize()
                # optimizer.step(fp16_param_groups=model.parameters())
            else:
                optimizer.step()
            
            scheduler.step()

            log_dict = {
                    "train_loss": loss,
                    "learning_rate": scheduler.get_last_lr()[0],
            }  
                
            if WANDB:
                wandb.log(log_dict, step=_step)
                _step += bsz

from torch.utils.data import Sampler
class CustomSampler(Sampler):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        import pickle
        filename = 'training_preload_indices'
        try:
            with open (filename, 'rb') as f: 
                indices = pickle.load(f) 
            f.close()
            return iter(indices)
        except IOError:
                pass
        
        
        # data_sampler = RandomSampler(train_set)
        # indices = [i for i in data_sampler]
        
        with open (filename, 'wb') as f:
            pickle.dump(indices, f)
            
        return iter(indices)

    def __len__(self):
        return len(self.data)
    


if __name__ == '__main__':
    FP16_mode   = False
    DEVICES     = "0,1,4,5"
    # DEVICES     = "0,1,4,5"
    
    config      = GPTJConfig.from_pretrained('gpt-j-6B')

    # Build the pipeline.
    model = get_gptj_mobiusmodel(config, DEVICES, nlayers=-1, fp16=FP16_mode)
    # model = get_gptj_gpipemodel(config, DEVICES, nlayers=28, fp16=FP16_mode)

    criterion = nn.CrossEntropyLoss()
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=0, weight_decay=3e-7)
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = DeepSpeedCPUAdam(model.parameters())
    scheduler = deepspeed.runtime.lr_schedules.OneCycle(optimizer, cycle_min_lr=0, cycle_max_lr=0.0001)
    # scheduler = None

    if WANDB:
        wandb.watch(
            model,
            criterion=criterion,
            log="all"
        )

    train_set = WT2_Dataset2(train_iter, 1, 512)
    data_sampler = CustomSampler(train_set)
    train_dl = DataLoader(dataset=train_set, 
                          batch_size=bsz, 
                          sampler=data_sampler,
                          num_workers=1)
    
    train(model, train_dl, optimizer, criterion, scheduler, epochs=epochs, fp16=FP16_mode)