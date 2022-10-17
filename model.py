from lib2to3.pgen2 import token
import sys
import math
from numpy import half
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.pipeline.sync import Pipe
import tempfile
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from transformers import AutoTokenizer
from GPTJ.modeling_gptj import GPTJBlock, GPTJConfig, GPTJModel, GPTJForCausalLM

import sys
from torchmobius.kernel import MobiusKernel

from torchgpipe import GPipe

class Encoder(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(Encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # Need (S, N) format for encoder.
        src = src.t()
        src = self.encoder(src) * math.sqrt(self.ninp)
        return self.pos_encoder(src)

class Decoder(nn.Module):
    def __init__(self, ntoken, ninp):
        super(Decoder, self).__init__()
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp):
        # Need batch dimension first for output of pipeline.
        return self.decoder(inp).permute(1, 0, 2)


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def split_devices(raw_devices):
    up = []
    low = []
    for dev in raw_devices:
        if dev <= 3:
            up.append(dev)
        else :
            low.append(dev)
    ret = []
    if len(up) != 0:
        ret.append(up)
    if len(low) != 0:
        ret.append(low)
    return ret 

def get_gptj_model(config, nlayers, dropout=0.5):
    num_gpus = 8
    partition_len = ((nlayers - 1) // num_gpus) + 1

    module_list = []
    tmp_list = [nn.Embedding(config.vocab_size, config.n_embd).cuda(0)]
    tmp_list.append(nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon).cuda(0))

    for i in range(nlayers):
        block = GPTJBlock(config=config)
        if i != 0 and i % (partition_len) == 0:
            module_list.append(nn.Sequential(*tmp_list))
            tmp_list = []
        device = i // (partition_len)
        tmp_list.append(block.to(device))
    
    tmp_list.append(nn.Linear(config.n_embd, config.vocab_size).cuda(num_gpus - 1))
    module_list.append(nn.Sequential(*tmp_list))
    seq = torch.nn.Sequential(*module_list)

    return seq

def get_gptj_pipemodel(config, nlayers, dropout=0.5):
    num_gpus = 8
    partition_len = ((nlayers - 1) // num_gpus) + 1

    module_list = []
    tmp_list = [nn.Embedding(config.vocab_size, config.n_embd).cuda(0)]
    tmp_list.append(nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon).cuda(0))

    for i in range(nlayers):
        block = GPTJBlock(config=config)
        if i != 0 and i % (partition_len) == 0:
            module_list.append(nn.Sequential(*tmp_list))
            tmp_list = []
        device = i // (partition_len)
        tmp_list.append(block.to(device))
    
    tmp_list.append(nn.Linear(config.n_embd, config.vocab_size).cuda(num_gpus - 1))
    module_list.append(nn.Sequential(*tmp_list))
    
    chunks = 8
    model = Pipe(torch.nn.Sequential(*module_list), chunks = chunks)

    return model

def get_gptj_mobiusmodel(config, DEVICES, nlayers=-1, dropout=0.5, fp16=False):
    devices = [int(dev) for dev in DEVICES.split(',')]
    N_GPU = len(devices)
    tmp_devices = split_devices(devices)
    num_gpus = N_GPU

    if nlayers != -1:
        # tmp_list = [nn.Embedding(config.vocab_size, config.n_embd)]
        # tmp_list.append(nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon))

        # for i in range(nlayers):
        #     block = GPTJBlock(config=config)
        #     tmp_list.append(block)
        
        # tmp_list.append(nn.Linear(config.n_embd, config.vocab_size))
        # seq = torch.nn.Sequential(*tmp_list)
        
        gptj_model = GPTJForCausalLM(config)
        seq = nn.Sequential(*(gptj_model.to_layers()))
        
    else:
        # pretrain model
        gptj_model = GPTJForCausalLM.from_pretrained("gpt-j-6B", config=config)
        seq = nn.Sequential(*(gptj_model.to_layers()))


    # Mobius configuration
    GPU_SIZE_RATIO = 0.44
    PARTITION_RATIO = 0.8
    PARTITION_NUM = 4
    CROSSMAP = False

    sample_ids = torch.rand(4, 512)
    sample_tgt = torch.rand(4, 512)
    
    if fp16:
        seq = seq.half()

    mobius_model = MobiusKernel(
                    module=seq, 
                    devices=tmp_devices,
                    gpu_size_ratio=GPU_SIZE_RATIO,         
                    partition_ratio=PARTITION_RATIO,
                    chunks=8,
                    sample=sample_ids,
                    max_microbatch_param=1,
                    model_name="Mobius_test",
                    ilp=False,
                    partition_num=PARTITION_NUM,       
                    cross_map=CROSSMAP,
                    half_mode=fp16,      
                )
    
    return mobius_model

def get_gptj_gpipemodel(config, DEVICES, nlayers, dropout=0.5, fp16=False):
    devices = [int(dev) for dev in DEVICES.split(',')]
    num_gpus = len(devices)
    
    partition_len = ((nlayers) // num_gpus)
    if partition_len % num_gpus != 0:
        partition_len += 1

    total_layers = 0

    balance = [4, 4, 4, 4, 4, 4, 4, 4]
    # for i in range(nlayers):
    #     # Let ME !!!! decide the placement.
    #     if i != 0 and i % (partition_len) == 0:
    #         balance.append(i + 2 - total_layers)
    #         total_layers = i + 2
    
    # balance.append(nlayers + 4 - total_layers)
    # total_layers = nlayers + 4
    # print(balance)
    
    gptj_model = GPTJForCausalLM.from_pretrained("gpt-j-6B", config=config)
    seq = nn.Sequential(*(gptj_model.to_layers()))
        
    # module_list = []
    # tmp_list = [nn.Embedding(config.vocab_size, config.n_embd)]
    # tmp_list.append(nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon))

    # for i in range(nlayers):
    #     block = GPTJBlock(config=config)
    #     tmp_list.append(block)
        
    # tmp_list.append(nn.Linear(config.n_embd, config.vocab_size))
    # seq = torch.nn.Sequential(*tmp_list)

    if fp16:
        seq = seq.half()

    # -- chunks = 4 is pipeline setting.
    model = GPipe(seq, balance=balance, devices=devices, chunks=num_gpus, checkpoint='always')

    return model
