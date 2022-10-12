import torch 
from typing import (TYPE_CHECKING, Callable, Dict, Generator, List, Optional, Tuple, Type, Union,
                    cast)

from torchmobius.attribute import MobiusModuleAttribute, MobiusTensorAttribute

"""
class MobiusSchedulerMeta(object):
    def __init__(self, 
                 batch_number: int,
                 ):
        self.batch_number = batch_number

        # the key module should be transfered when caculating the value[0] at value[1] microbatch 
        self.transfer_fwd_future_param: Dict[torch.nn.modules, Tuple[torch.nn.modules, int]] = {}
        self.transfer_bwd_future_param: Dict[torch.nn.modules, Tuple[torch.nn.modules, int]] = {}

        # the key[0] module's key[1] activation should be transfered 
        # when caculating the value[0] at value[1] microbatch 
        self.transfer_future_activation: Dict[Tuple[torch.nn.modules, int], Tuple[torch.nn.modules, int]] = {}

        self.device_mapping: Dict[torch.nn.modules, torch.nn.device] = {}
        self.heads: List[torch.nn.modules] = []
        self.tails: List[torch.nn.modules] = []

    def add_head(self, module: torch.nn.modules):
        self.heads.append(module)

    def add_tail(self, module: torch.nn.modules):
        self.tails.append(module)

    def add_device(self, 
                   module: torch.nn.modules,
                   device: torch.device):
        self.device_mapping[module] = device

    def add_transfer_fwd_future_param(self, 
                                      module: torch.nn.modules,
                                      future_module: torch.nn.modules,
                                      micro_batch_number: int
                                    ):
        self.transfer_fwd_future_param[future_module] = tuple([module, micro_batch_number])

    def add_transfer_bwd_future_param(self, 
                                      module: torch.nn.modules,
                                      future_module: torch.nn.modules,
                                      micro_batch_number: int
                                    ):
        self.transfer_bwd_future_param[future_module] = tuple([module, micro_batch_number])

    def add_transfer_future_activation(self, 
                                       module: torch.nn.modules,
                                       micro_batch_number: int,
                                       future_module: int,
                                       future_micro_batch_number: int
                                    ):
        self.transfer_future_activation[tuple([future_module, future_micro_batch_number])] = tuple([module, micro_batch_number])

    

class MobiusSchedulerManager:
    def __init__(self, 
                 model,
                 devices,
                 prefetch_num,
                 virtual_gpu_ratio,
                 layer,
                 ):
        self.model = model

        self.devices = [torch.device(d) for d in devices]
        self.devices = cast(List[torch.device], self.devices)
        
        self.virtual_gpu_ratio = virtual_gpu_ratio
        self.virtual_device = []
        for _ in range(self.virtual_gpu_ratio):
            self.virtual_device += self.devices

        self.gpu_n             = len(self.devices)
        self.virtual_gpu_n     = self.gpu_n * virtual_gpu_ratio
        self.virtual_gpu_ratio = virtual_gpu_ratio
        self.prefetch_num      = prefetch_num
        self.layer             = layer

        self.schedule           = MobiusSchedulerMeta(self.gpu_n)
        self.modules_on_GPU     = None 
        self.microbatch_num     = self.gpu_n
        self._init_schedule()

        self.act_prof_info: Dict[Tuple[torch.nn.modules, int], Dict] = {}
        self.fwd_prof_info: Dict = {}
        self.param_prof_info: Dict = {}

    def _init_schedule(self):
        per_virtual_block_number = [0 for _ in range(self.virtual_gpu_n)]

        # FIXME(fyy) for gpt, 2 means attention and mlp
        for i in range(self.layer):
            per_virtual_block_number[i % self.virtual_gpu_n] += 2

        per_virtual_block_number[0]     += 1
        per_virtual_block_number[-1]    += 1
        
        # assign the module to each device 
        count = 0
        virtual_device_idx = 0
        virtual_device_module = [[] for i in range(self.virtual_gpu_n)]
        for _, child in self.model.named_children():
            physic = self.devices[virtual_device_idx % self.gpu_n]
            virtual_device_module[virtual_device_idx].append(child)
            self.schedule.add_device(child, physic)
            count += 1

            if count == per_virtual_block_number[virtual_device_idx]:
                virtual_device_idx += 1
                count = 0

        # get the sequential of the model
        self.modules_on_GPU = [[] for _ in range(self.gpu_n)]
        modules_on_GPU_idx  = [[] for _ in range(self.gpu_n)]
        for i in range(self.gpu_n):
            for j in range(self.virtual_gpu_ratio):
                self.modules_on_GPU[i].append(virtual_device_module[i + j * self.gpu_n])
                modules_on_GPU_idx[i].append(i + j * self.gpu_n)
        
        # init
        for j in range(self.gpu_n):
            for i in range(self.prefetch_num):
                heads = self.modules_on_GPU[j][i]
                for head in heads:
                    self.schedule.add_head(head)

            for i in range(1, self.prefetch_num + 1):
                tails = self.modules_on_GPU[j][-i]
                for tail in tails:
                    self.schedule.add_tail(tail)

            
            # NOTE(fyy) init prefetch 
            for i in range(self.virtual_gpu_ratio - self.prefetch_num):
                for module in self.modules_on_GPU[j][i + self.prefetch_num]:
                    self.schedule.add_transfer_fwd_future_param(self.modules_on_GPU[j][i][0], module, self.gpu_n - 1)

            for i in range(1, self.virtual_gpu_ratio - self.prefetch_num + 1):
                for module in self.modules_on_GPU[j][-(i + self.prefetch_num)]:
                    self.schedule.add_transfer_bwd_future_param(self.modules_on_GPU[j][-i][0], module, self.gpu_n - 1)

                for bm in range(self.gpu_n):
                    self.schedule.add_transfer_future_activation(module=self.modules_on_GPU[j][-i][0], 
                                                                 micro_batch_number=self.gpu_n - 1, 
                                                                 future_module=modules_on_GPU_idx[j][-(i + self.prefetch_num)], 
                                                                 future_micro_batch_number=bm)


    def _modify_transfer_fwd_future_param(self,
                                         module: torch.nn.modules,
                                         future_module: torch.nn.modules,
                                         micro_batch_number: int):
        
        post_module, post_microbatch = self.schedule.transfer_fwd_future_param[future_module]
        self.schedule.add_transfer_fwd_future_param(module, future_module, micro_batch_number)

        for _, tensor in future_module.named_parameters(recurse=True):
            post_module.mobius_module_attr.transfer_fwd_tensor.remove(tuple([tensor, post_microbatch]))
            module.mobius_module_attr.bind_fwd_tensor(tensor, micro_batch_number)


    def _modify_transfer_bwd_future_param(self,
                                         module: torch.nn.modules,
                                         future_module: torch.nn.modules,
                                         micro_batch_number: int):

        post_module, post_microbatch = self.schedule.transfer_bwd_future_param[future_module]
        self.schedule.add_transfer_bwd_future_param(module, future_module, micro_batch_number)

        for _, tensor in future_module.named_parameters(recurse=True):
            post_module.mobius_module_attr.transfer_bwd_tensor.remove(tuple([tensor, post_microbatch]))
            module.mobius_module_attr.bind_bwd_tensor(tensor, micro_batch_number)


    def _modify_transfer_future_activation( self,
                                           module: torch.nn.modules,
                                           micro_batch_number: int,
                                           future_module: int,
                                           future_micro_batch_number: int):
        self.schedule.add_transfer_future_activation(module, micro_batch_number, future_module, future_micro_batch_number)


    def prof_fwd(self):
        for _, param in self.model.named_parameters(recurse=True):
            param.mobius_tensor_attr.record_profile_info('fwd')
    

    def prof_bwd(self):
        for _, param in self.model.named_parameters(recurse=True):
            param.mobius_tensor_attr.record_profile_info('bwd')
            
    def proc_collect(self):
        for _, param in self.model.named_parameters(recurse=True):
            self.param_prof_info[param] = param.mobius_tensor_attr.prof_info 
    
    
    def apply_schedule(self):
        pass 
"""
