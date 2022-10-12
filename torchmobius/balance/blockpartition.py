"""Implements "Block Partitions of Sequences" by Imre Bárány et al.

Paper: https://arxiv.org/pdf/1308.2452.pdf

"""

from audioop import avg
from cgi import print_environ
from ctypes import cast
from re import S
from tkinter.tix import Tree
import cvxpy as cp
import pickle
from statistics import mean


from tkinter.messagebox import NO
from unicodedata import mirrored
from requests import get
import torch
from torch import Tensor
import torch.nn as nn

from torchmobius.balance.profile import mobius_profile

from ast import Param
import enum
from turtle import delay
from typing import Dict, Iterator, List, Tuple, Union

from torchmobius.utils import mobius_logger

__all__ = ['solve']

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]


def solve(sequence: List[int], partitions: int = 1) -> List[List[int]]:
    """Splits a sequence into several partitions to minimize variance for each
    partition.

    The result might not be optimal. However, it can be done only in O(kn³),
    where k is the number of partitions and n is the length of the sequence.

    """
    if partitions < 1:
        raise ValueError(f'partitions must be a positive integer ({partitions} < 1)')

    n = len(sequence)
    if n < partitions:
        raise ValueError(f'sequence is shorter than intended partitions ({n} < {partitions})')

    # Normalize the sequence in [0, 1].
    minimum = min(sequence)
    maximum = max(sequence) - minimum

    normal_sequence: List[float]
    if maximum == 0:
        normal_sequence = [0 for _ in sequence]
    else:
        normal_sequence = [(x-minimum)/maximum for x in sequence]

    splits = [n//partitions * (x+1) for x in range(partitions-1)] + [n]

    def block_size(i: int) -> float:
        start = splits[i-1] if i > 0 else 0
        stop = splits[i]
        return sum(normal_sequence[start:stop])

    def leaderboard() -> Iterator[Tuple[float, int]]:
        return ((block_size(i), i) for i in range(partitions))

    while True:
        """
        (1) Fix p ∈ [k] with M(P) = bp. So Bp is a maximal block of P.
        """
        # max_size: M(P)
        max_size, p = max(leaderboard())

        while True:
            """
            (2) If M(P) ≤ m(P) + 1, then stop.
            """
            # min_size: m(P)
            min_size, q = min(leaderboard())

            if max_size <= min_size + 1:
                
                return [[k for k in range(i, j)] for i, j in zip([0]+splits[:-1], splits)]
                # return [sequence[i:j] for i, j in zip([0]+splits[:-1], splits)]

            """
            (3) If M(P) > m(P) + 1, then let m(P) = bq for the q ∈ [k] which is
            closest to p (ties broken arbitrarily). Thus Bq is a minimal block
            of P. Let Bh be the block next to Bq between Bp and Bq. (Note that
            Bh is a non-empty block: if it were, then m(P) = 0 and we should
            have chosen Bh instead of Bq.)
            """
            if p < q:
                """
                So either p < q and then h = q−1 and we define P ∗ by moving
                the last element from Bh = Bq−1 to Bq,
                """
                h = q - 1
                splits[h] -= 1
            else:
                """
                or q < p, and then h = q + 1 and P ∗ is obtained by moving the
                first element of Bh = Bq+1 to Bq.
                """
                h = q + 1
                splits[q] += 1

            """
            Set P = P ∗ . If p = h, then go to (1), else go to (2).
            """
            if p == h:
                break


# backup
def mobius_solver_partition(sequence: nn.Sequential, 
                     device: torch.device, 
                     sample: TensorOrTensors, 
                     device_n: int, 
                     repeat_layer=[1, -2],
                     repeat_number: int=1,
                     device_memory_ratio: float=1):
    
    # repeat
    assert repeat_layer[1] <= 0
    
    compress_sequence_list = []
    for i in range(0, repeat_layer[0]):
        compress_sequence_list.append(sequence[i])
        
    for j in range(repeat_number):
        compress_sequence_list.append(sequence[repeat_layer[0] + j])
    
    for i in range(len(sequence) + repeat_layer[1] + 1, len(sequence)):
        compress_sequence_list.append(sequence[i])
    
    repeat_times = int((len(sequence) + repeat_layer[1] + 1 - repeat_layer[0]) / repeat_number)
    compress_sequence = nn.Sequential(*compress_sequence_list)
    
    device_total_memory = torch.cuda.get_device_properties(torch.device(device)).total_memory * device_memory_ratio
    t_fwd, t_bwd, s_param, s_act, s_fwd, s_bwd = mobius_profile(compress_sequence, sample, device)
    print(t_fwd)
    t_fwd = t_fwd[:repeat_layer[0]] + t_fwd[repeat_layer[0]: repeat_layer[1] + 1] * repeat_times + t_fwd[repeat_layer[1] + 1:] 
    t_bwd = t_bwd[:repeat_layer[0]] + t_bwd[repeat_layer[0]: repeat_layer[1] + 1] * repeat_times + t_bwd[repeat_layer[1] + 1:] 
    s_act = s_act[:repeat_layer[0]] + s_act[repeat_layer[0]: repeat_layer[1] + 1] * repeat_times + s_act[repeat_layer[1] + 1:] 
    s_fwd = s_fwd[:repeat_layer[0]] + s_fwd[repeat_layer[0]: repeat_layer[1] + 1] * repeat_times + s_fwd[repeat_layer[1] + 1:] 
    s_bwd = s_bwd[:repeat_layer[0]] + s_bwd[repeat_layer[0]: repeat_layer[1] + 1] * repeat_times + s_bwd[repeat_layer[1] + 1:] 
    s_param = s_param[:repeat_layer[0]] + s_param[repeat_layer[0]: repeat_layer[1] + 1] * repeat_times + s_param[repeat_layer[1] + 1:] 
    
    t_fwd = list(map(int, t_fwd))
    
    partitions = None
    for i in range(device_n, len(sequence) + 1, device_n):
        
        partitions = solve(t_fwd[:repeat_layer[0]:repeat_layer[1] + 1], i)
        for i, partition in enumerate(partitions):
            partitions[i] = [i + repeat_layer[0] for i in partition]

        # add head and tail of the model
        for i in reversed(range(repeat_layer[0])):
            partitions[0].insert(0, i)
        for i in range(len(sequence) + repeat_layer[1], len(sequence)):
            partitions[-1].append(i)
        
        bwd_reserved_memory_l = []
        fwd_reserved_memory_l = []
        partition_param_size_l = []
        t_partition_fwd_l = []
        t_partition_bwd_l = []
        
        # collect submodule information
        for _, sub_partition in enumerate(partitions):
            bwd_reserved_memory = device_total_memory
            fwd_reserved_memory = device_total_memory
            partition_param_size = 0
            t_partition_fwd = 0
            t_partition_bwd = 0
            
            for m_idx in sub_partition:
                bwd_reserved_memory -= s_param[m_idx] 
                bwd_reserved_memory -= s_bwd[m_idx]
                fwd_reserved_memory -= s_param[m_idx]
                fwd_reserved_memory -= s_fwd[m_idx]
                
                partition_param_size += s_param[m_idx]
                t_partition_fwd += t_fwd[m_idx]
                t_partition_bwd += t_bwd[m_idx]
                
            bwd_reserved_memory_l.append(bwd_reserved_memory)
            fwd_reserved_memory_l.append(fwd_reserved_memory)
            partition_param_size_l.append(partition_param_size)
            t_partition_fwd_l.append(t_partition_fwd)
            t_partition_bwd_l.append(t_partition_bwd)
            
        # ensure backward there is a double buffer
        for p_idx in range(len(partitions) - 1, 0, -1):
            if bwd_reserved_memory_l[p_idx] < partition_param_size_l[p_idx - 1]:
                partitions = None
                break
            
        if partitions != None:
            break
            
    assert(partitions != None)     
    # print(partitions)
    return partitions, len(partitions) // device_n


# def fake_paratition(module, device_n, num=1):
#     partition = []
#     for i in range(0, len(module), num):
#         partition.append([])
#         for j in range(num):
#             partition[-1].append(i + j)
#     return partition, len(partition) // device_n + len(partition) % device_n

def fake_paratition(module, device_n, num=4):
    tmp_list = [0, 1]
    partition = []
    for i in range(2, len(module) - 1, num):
        for j in range(num):
            if i + j < len(module) - 1:
                tmp_list.append(i + j)
        partition.append(tmp_list.copy())
        tmp_list = []
    partition[-1].append(len(module) - 1)
    print("[+++] ", partition)
    return partition, len(partition) // device_n + len(partition) % device_n
                
def mobius_partition(sequence: nn.Sequential,
               device: torch.device,
               sample: TensorOrTensors, 
               device_n: int, 
               bandwidth: float,
               repeat_layer=[1, -2],
               repeat_number: int=1,
               device_memory_ratio: float=1,
               ilp_name='',
               use_prev_solution=False,
               overwrite_pre_solution=False):
    
    if use_prev_solution:
        filename = ilp_name + '.ilp'
        try:
            f = open(filename)
            f.close()
            with open (filename, 'rb') as f: 
                mobius_time, partitions = pickle.load(f) 
            mobius_logger(f"use previous solution from {filename}")
            return partitions, len(partitions) // device_n + len(partitions) % device_n
        except IOError:
            pass
    
    def get_non_repeat_model():
        assert repeat_layer[1] <= 0
    
        compress_sequence_list = []
        for i in range(0, repeat_layer[0]):
            compress_sequence_list.append(sequence[i])
            
        for j in range(repeat_number):
            compress_sequence_list.append(sequence[repeat_layer[0] + j])
        
        for i in range(len(sequence) + repeat_layer[1] + 1, len(sequence)):
            compress_sequence_list.append(sequence[i])
    
        repeat_times = int((len(sequence) + repeat_layer[1] + 1 - repeat_layer[0]) / repeat_number)
        compress_sequence = nn.Sequential(*compress_sequence_list)
        
        return compress_sequence
    
    repeat_model = get_non_repeat_model()
    repeat_times = int((len(sequence) + repeat_layer[1] + 1 - repeat_layer[0]) / repeat_number)
        
    device_total_memory = torch.cuda.get_device_properties(torch.device(device)).total_memory * device_memory_ratio
    t_fwd, t_bwd, s_param, s_act, s_fwd, s_bwd = mobius_profile(repeat_model, sample, device)
    print(t_fwd)
    print(t_bwd)
    print(s_param)
    print(s_fwd)
    print(s_bwd)
    print(s_act)
    print(device_total_memory)
    
    t_fwd = t_fwd[:repeat_layer[0]] + t_fwd[repeat_layer[0]: repeat_layer[1] + 1] * repeat_times + t_fwd[repeat_layer[1] + 1:] 
    t_bwd = t_bwd[:repeat_layer[0]] + t_bwd[repeat_layer[0]: repeat_layer[1] + 1] * repeat_times + t_bwd[repeat_layer[1] + 1:] 
    s_act = s_act[:repeat_layer[0]] + s_act[repeat_layer[0]: repeat_layer[1] + 1] * repeat_times + s_act[repeat_layer[1] + 1:] 
    s_fwd = s_fwd[:repeat_layer[0]] + s_fwd[repeat_layer[0]: repeat_layer[1] + 1] * repeat_times + s_fwd[repeat_layer[1] + 1:] 
    s_bwd = s_bwd[:repeat_layer[0]] + s_bwd[repeat_layer[0]: repeat_layer[1] + 1] * repeat_times + s_bwd[repeat_layer[1] + 1:] 
    s_param = s_param[:repeat_layer[0]] + s_param[repeat_layer[0]: repeat_layer[1] + 1] * repeat_times + s_param[repeat_layer[1] + 1:] 
    
    s_f = []
    s_b = []
    
    for i in range(len(s_fwd)):
        s_f.append(s_fwd[i] + s_param[i])
        s_b.append(s_bwd[i] + s_param[i])
    
    t, partitions = mobius_ilp(n_gpu=device_n,
                              microbatch=device_n,
                              memory_size=device_total_memory,
                              bandwidth=bandwidth,
                              s_f=s_f,
                              s_b=s_b,
                              s_a=s_act,
                              s_g=s_act,
                              T_f=t_fwd,
                              T_b=t_bwd,
                              ilp_name=ilp_name,
                              use_prev_solution=use_prev_solution,
                              overwrite_pre_solution=overwrite_pre_solution,
                              seq=sequence)
    mobius_logger(f"Predict time: {t}")
    return partitions, len(partitions) // device_n + len(partitions) % device_n
    
def mobius_ilp( n_gpu: int,         # number of GPU
                microbatch: int,    # number of microbatch
                memory_size: int,   # memory siz of GPU
                bandwidth: float,     # avarage bandwidth of PCIe without contention
                s_f: list,          # memory capacity require of each layer in forward
                s_b: list,          # memory capacity require of each layer in backward
                s_a: list,          # size of activation
                s_g: list,          # size of gradient
                T_f: list,          # time cost of each layer in each batch in forward 
                T_b: list,          # time cost of each layer in each batch in backward
                verbose=True,
                ilp_name='',
                use_prev_solution=False,
                overwrite_pre_solution=False,
                seq=None):
    
    if use_prev_solution:
        filename = ilp_name + '.ilp'
        try:
            f = open(filename)
            f.close()
            with open (filename, 'rb') as f: 
                mobius_time, partitions = pickle.load(f) 
            mobius_logger(f"use previous solution from {filename}")
            return mobius_time, partitions
        except IOError:
            pass
    
    mobius_logger("start ILP solving")
    s_a = mean(s_a)
    s_g = mean(s_g)
    n_layer = len(s_b)
    n_block = int(int(n_layer / n_gpu) * n_gpu)

    # if layer i is in jth block, B[i][j] = True
    B       = [[cp.Variable(boolean=True) for _ in range(n_block)] for _ in range(n_layer)]
    
    # start time of jth block in forward and backward
    tb_f    = [[cp.Variable(pos = True) for _ in range(microbatch)] for _ in range(n_block)]
    tb_b    = [[cp.Variable(pos = True) for _ in range(microbatch)] for _ in range(n_block)]
    
    constraints = []
    
    ##################### B constraints #####################
    
    # the first layer must be in first block 
    constraints.append(B[0][0] == True)
    
    # the block assignment must be increasing
    BS = []       
    for i in range(n_layer):
        sum = 0
        for j in range(n_block):
            sum += B[i][j] * j
        BS.append(sum)
        
    for i in range(n_layer - 1):
        # BA is increasing 
        constraints.append(BS[i] <= BS[i + 1])  
        constraints.append(BS[i + 1] - BS[i] <= 1)  
    
    # each layer belongs to only one block
    for i in range(n_layer):
        sum = 0
        for j in range(n_block):
            sum += B[i][j]
        constraints.append(sum == 1)
    
    # for balance
    for j in range(0, n_block, n_gpu):
        k = []
        for x in range(n_gpu):
            sum = 0
            for i in range(n_layer):
                sum += B[i][j + x]
            k.append(sum)
        
        for ki in range(1, n_gpu):
            constraints.append(k[0] <= k[ki])
            constraints.append(k[ki] - k[0] <= 1)
            
    # time cost of each layer and each batch
    Tb_f = []
    Tb_b = []

    # time cost of each layer
    C_f  = []
    C_b  = []

    # size of each layer
    S_f  = []
    S_b  = []

    
    for j in range(n_block):
        tmp = 0
        for i in range(n_layer):
            tmp += B[i][j] * T_f[i]
        Tb_f.append(tmp)
        
        tmp = 0
        for i in range(n_layer):
            tmp += B[i][j] * T_b[i]     
        Tb_b.append(tmp)
        
        C_f.append((Tb_f[j] + tb_f[j][microbatch - 1]) - tb_f[j][0])
        C_b.append((Tb_b[j] + tb_b[j][microbatch - 1]) - tb_b[j][0])

        size_f = 0
        size_b = 0
        for i in range(n_layer):
            size_f += B[i][j] * s_f[i]
            size_b += B[i][j] * s_b[i]
            
        S_f.append(size_f)
        S_b.append(size_b)
        
    ##################### memory constraints #####################
    
    # each block's size must less or equal than the gpu memory size 
    for j in range(n_block):
        constraints.append(S_b[j] <= memory_size)
        constraints.append(S_f[j] <= memory_size)
        
        # FIXME double buffer ??
        if j + n_gpu < n_block:
            constraints.append(S_f[j] + S_f[j + n_gpu] <= memory_size)
            
        if j - n_gpu >= 0:
            constraints.append(S_b[j] + S_b[j - n_gpu] <= memory_size)
        
    ##################### prefetch constraints #####################
    
    # the prefetch size of each block in forward and backward
    p_f = [cp.Variable(pos = True) for _ in range(n_block)]
    p_b = [cp.Variable(pos = True) for _ in range(n_block)]
    
    for j in range(n_block):
        # forward
        if j >= n_gpu:
            # less than reserved memory
            constraints.append( p_f[j] <= memory_size - S_f[j - n_gpu] )
            # limited by the bandwidth and the computation time of last block
            constraints.append( p_f[j] <= bandwidth * C_f[j - n_gpu])
        # less than the size of the block
        constraints.append( p_f[j] <= S_f[j])
        
        # backward
        if j + n_gpu < n_block:
             # less than reserved memory
            constraints.append( p_b[j] <= memory_size - S_b[j + n_gpu] )
             # limited by the bandwidth and the computation time of last block
            constraints.append( p_b[j] <= bandwidth * C_b[j + n_gpu])
         # less than the size of the block
        constraints.append( p_b[j] <= S_f[j])

    ##################### pipeline constraints #####################
    
    for j in range(n_block):
        for i in range(microbatch):
            # the first microbatch
            if i == 0:
                
                # forward
                # FIXME activation and gradient size is not right
                if j > 0:
                    # waiting for the last block's activation
                    constraints.append( tb_f[j][0] >= tb_f[j-1][0] + Tb_f[j-1] + s_a / bandwidth)
                
                if j > n_gpu:
                    # waiting for the data of the block transferred into the gpu memory
                    constraints.append( tb_f[j][0] >= \
                        tb_f[j-n_gpu][0] + Tb_f[j-n_gpu] + (S_f[j] - p_f[j]) / bandwidth)

                # backward
                if j + 1 < n_block:
                    # waiting for the last block's gradients
                    constraints.append( tb_b[j][0] >= tb_b[j+1][0] + Tb_b[j+1] + s_g / bandwidth )
                
                if j + n_gpu < n_block:
                    # waiting for the data of the blokc transferred into the gpu memory
                    constraints.append( tb_b[j][0] >= \
                        tb_b[j+n_gpu][0] + Tb_b[j+n_gpu] + (S_b[j] - p_b[j]) / bandwidth )
                    
            # non-first microbatches
            else: 
                # forward
                
                # microbatch order
                constraints.append( tb_f[j][i] >= tb_f[j][i-1] + Tb_f[j] )
                if j != 0:
                    # waiting for last activation
                    constraints.append( tb_f[j][i] >= tb_f[j-1][i] + Tb_f[j-1] + s_a / bandwidth )

                # backward
                # microbatch order
                constraints.append( tb_b[j][i] >= tb_b[j][i-1] + Tb_b[j] )
                if j < n_gpu - 1:
                    # waiting for last gradient
                    constraints.append( tb_b[j][i] >= tb_b[j+1][i] + Tb_b[j+1] + s_g / bandwidth )     
                    
    constraints.append(tb_f[0][0] == 0)
    constraints.append(tb_b[n_block - 1][0] >= tb_f[n_block - 1][microbatch - 1])  
    
    mobius_time = tb_b[0][microbatch-1] + Tb_b[0]
    objective = cp.Minimize(mobius_time)
    prob = cp.Problem(objective, constraints)
    
    try:
        # ref: https://www.gurobi.com/documentation/9.1/refman/parameters.html
        result = prob.solve(solver=cp.GUROBI, MIPGapAbs=0.1, reoptimize=True, verbose=verbose)
    
        partitions = []    
        for j in range(n_block):
            sub_partition = []
            for i in range(n_layer):
                if B[i][j].value == True:
                    sub_partition.append(i)
            if len(sub_partition) != 0:
                partitions.append(sub_partition)
    
    except Exception as e:
        print(e)
        mobius_logger("ilp fails")
        partitions, virtual_number =  fake_paratition(seq, device_n=n_gpu)
    

    mobius_logger("finish ILP solving")      
    if ilp_name != '':
        filename = ilp_name + '.ilp'
        mobius_logger(f"saving ILP solving to {filename}")
        with open (filename, 'wb') as f:
            pickle.dump((mobius_time.value, partitions), f)
        
    return mobius_time.value, partitions


def get_device_order(device_group:list):
    
    shared_dic = {}
    devices = []
    for i, group in enumerate(device_group):
        for device in group:
            shared_dic[device] = (i, len(group))
            devices.append(device)
            
    
    def shared(i, j):
        if shared_dic[i][0] == shared_dic[j][0]:
            return shared_dic[i][1]
        else:
            return 1
        
    def get_contention(arrange):
        sum = 0
        for i in range(len(arrange)):
            for j in range(i + 1, len(arrange)):
                sum += shared(arrange[i], arrange[j]) / (j - i)
        return sum
        
    import itertools
    dis = 1e10
    rc = None
    for arrange in itertools.permutations(devices, len(devices)):
        tmp_dis = get_contention(arrange)
        if tmp_dis < dis:
            rc = arrange
            dis = tmp_dis
            
    return list(rc)
        
    