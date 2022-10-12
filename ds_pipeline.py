#!/usr/bin/env python3

import os
import sys
import time
import argparse

import tqdm

import numpy as np
import random

import torch
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms
from torchvision.models import AlexNet
from torchvision.models import vgg19

from transformers import GPT2Config, GPT2Tokenizer, GPT2Model
from gpt import get_GPT_model, get_GPTLayers

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader

sys.path.append("/home/tzj/Code/MobiusNeuron/Experiment/3rd-party/ptflops")
from ptflops.utils import flops_to_string, params_to_string
from ptflops.pytorch_engine import add_flops_counting_methods, CUSTOM_MODULES_MAPPING

from torchtext.datasets import WikiText2
from datasets import WT2_Dataset

def get_args():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def get_fake_input(batch_size = 32, seq_num = 512):
    x = torch.ones(batch_size, seq_num)
    x = x.long()    
    return x

class Fake_dataset(torch.utils.data.Dataset):
    def __init__(self, size, seq_num, batch_size = 1):
        self.size = size
        self.seq_num = seq_num
        self.batch_size = batch_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        '''
            1 is batch size.
        '''
        return get_fake_input(batch_size=self.batch_size, seq_num=self.seq_num)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

'''
    Same to loss.mean.
'''
def loss_fn(output, label):
    return output.mean()

'''
    Train Pipeline main function.
'''
def train_pipe(model, dataset,
                warm_up=10, 
                n_steps=10,
                part="parameters",
                data_to_dev=None,   # None means cannot determin by caller
                name="",            # Name of this experiment(useless)
                verbose=False, 
                ost=sys.stdout, 
                no_profiler=False,
                log_st=sys.stdout):  
    torch.manual_seed(2021)
    deepspeed.runtime.utils.set_random_seed(2021)

    net = PipelineModule(layers=model,
                         loss_fn=loss_fn,
                         num_stages=4,
                         partition_method=part,
                         activation_checkpoint_interval=0)

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=dataset)

    for step in tqdm.tqdm(range(warm_up)):
        loss = engine.train_batch()
        # print(f'step: {step:3d} / {warm_up:3d} loss: {loss}')

    start = time.time() # Start timer .
    model.train()
    for step in tqdm.tqdm(range(n_steps)):
        loss = engine.train_batch()
        # print(f'step: {step:3d} / {n_steps:3d} loss: {loss}')
    end = time.time()   # End timer

    try :
        ost.write("Time: {}\n".format(end - start))
    except Exception as e:
        print("IO Error : ", e)
        exit(-1)

if __name__ == '__main__':
    try :
        HIDDEN_DIM  =       int(os.environ.get('HIDDEN_DIM'))
        SEQ_LEN     =       int(os.environ.get('SEQ_LEN'))
        NUM_LAYER   =       int(os.environ.get('NUM_LAYER'))
        NUM_HEAD    =       int(os.environ.get('NUM_HEAD'))
        BATCH_SIZE  =       int(os.environ.get('BATCH_SIZE'))
        MODEL_NAME  =       str(os.environ.get('MODEL_NAME'))
        DEVICES     =       str(os.environ.get('DEVICES'))
        GPU_SIZE_RATIO  =   float(os.environ.get('GPU_SIZE_RATIO'))
        PARTITION_RATIO =   float(os.environ.get('PARTITION_RATIO'))
        PARTITION_NUM =     int(os.environ.get('PARTITION_NUM'))
        MP =                str(os.environ.get('MP'))
        CROSSMAP =          True if (os.environ.get('CROSSMAP') == "True") else False
        
        # Follow are single process env.
        FLOPS_NAME =        str(os.environ.get('FLOPS_NAME'))
        N_STEP    =         int(os.environ.get('N_STEP'))
        N_WARMUP   =        int(os.environ.get('N_WARMUP'))
    except Exception as e:
        print("Remember to run with environment. (you can try test_exp.sh)")
        exit(-1)

    # init the rpc environment.
    deepspeed.init_distributed(dist_backend='nccl')

    args = get_args()

    train_iter = WikiText2(split='train')
    trainset = WT2_Dataset(train_iter, SEQ_LEN, batch_size=BATCH_SIZE)
    # trainset = Fake_dataset(BATCH_SIZE * (N_STEP + N_WARMUP + 512), 512)

    CONFIG = GPT2Config(n_embd=HIDDEN_DIM, n_head=NUM_HEAD, n_layer=NUM_LAYER)
    model, _ = get_GPT_model(CONFIG, 3e-4, (0.9, 0.999))

    with open("./flops_time/{}.txt".format(FLOPS_NAME), 'w') as ft_out:
        train_pipe(model, trainset, warm_up=N_WARMUP, n_steps=N_STEP, part='parameters', ost=ft_out, name=MODEL_NAME)
