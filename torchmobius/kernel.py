"""The MobiusNeuron GPipe interface"""
from ast import mod
from collections import OrderedDict
import enum
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union, cast, Dict
import time

from numpy import clip 

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda

import torchmobius
from torchmobius import microbatch
from torchmobius.batchnorm import DeferredBatchNorm
from torchmobius.pipeline import MobiusPipeline
from torchmobius.skip.layout import inspect_skip_layout
from torchmobius.stream import AbstractStream
from torchmobius.memory import ContiguousMemoryAllocator
from torchmobius.balance.blockpartition import mobius_partition, fake_paratition, get_device_order
from torchmobius.attribute import (MobiusModuleAttribute, MobiusTensorAttribute,
                                  register_mobius_module, register_mobius_param_in_cpu, register_mobius_param_in_gpu)
from torchmobius.hook import register_hooks_on_module
from torchmobius.utils import mobius_logger

Device = Union[torch.device, int, str]
Devices = Union[Iterable[Device], List[Device]]

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

if TYPE_CHECKING:
    Module = nn.Module[TensorOrTensors]
    NamedModules = OrderedDict[str, Module]
else:
    Module = nn.Module
    NamedModules = OrderedDict

MOVING_DENIED = TypeError('denied to move parameters and buffers, '
                          'because Mobius should manage device placement')

class MobiusBalanceError(ValueError):
    pass

def verify_module(module: nn.Sequential) -> None:
    # TODO
    pass    

class MobiusKernel(Module):

    def __init__(self,
                 module: nn.Sequential,
                 devices: List[List],
                 gpu_size_ratio: float,
                 chunks: int,
                 sample: TensorOrTensors,
                 # TODO
                 partition_ratio=0.5,
                 checkpoint: str = 'always',
                 deferred_batch_norm: bool = False,
                 max_microbatch_param = 1,
                 model_name='',
                 cross_map=True,
                 ilp=True,
                 partition_num=1,
                 noprefetch=False,
                 half_mode=False,
                 # TODO(fyy)
                 ) :
        super().__init__()
        
        module = module
        # for profile
        # self.module.mobius_backup = True 
        
        # for design
        self.cross_map = cross_map
        self.ilp = ilp
        self.partition_num = partition_num
        self.noprefetch = noprefetch
        self.half_mode = half_mode
        
        if cross_map:
            devices = get_device_order(devices)
        else:
            d = []
            for i in devices:
                for j in i:
                    d.append(j)
            devices = d
            
        self.devices = [torch.device(d) for d in devices]
        sample = sample.to(self.devices[0])
        
        self.chunks = chunks
        self.checkpoint = checkpoint
        
        self.max_microbatch_param = max_microbatch_param

        # set pyhsic fo physic device for offloading param 
        self.act_upload_streams_dic = { d : torch.cuda.Stream(d) for d in self.devices}
        self.act_offload_streams_dic = { d : torch.cuda.Stream(d) for d in self.devices}

        # get the memory buffer for physic devices
        self.device_mems: Dict[torch.device, ContiguousMemoryAllocator] = {}

        for device in self.devices:
            if half_mode:
                gpu_size = int(torch.cuda.get_device_properties(torch.device(device)).total_memory * gpu_size_ratio / 2) 
                self.device_mems[device] = ContiguousMemoryAllocator(gpu_size, torch.float16, device)
            else:
                gpu_size = int(torch.cuda.get_device_properties(torch.device(device)).total_memory * gpu_size_ratio / 4) 
                self.device_mems[device] = ContiguousMemoryAllocator(gpu_size, torch.float32, device)
        
        ########################### partition
        
        # get the partition 
        mobius_logger("start partition")
        try:
            if not self.ilp:
                assert False
                
            self.partition_idx, virtual_number = mobius_partition(sequence=module, 
                                                                device=self.devices[0], 
                                                                sample=sample, 
                                                                bandwidth=10,
                                                                device_n=len(self.devices), device_memory_ratio=partition_ratio,
                                                                ilp_name=model_name,
                                                                use_prev_solution=True,
                                                                overwrite_pre_solution=False)
        except Exception as e:
            print(e)
            mobius_logger("ilp fails")
            self.partition_idx, virtual_number =  fake_paratition(module, device_n=len(self.devices), num=self.partition_num)
        mobius_logger("finish partition")
        print(self.partition_idx, virtual_number)
        
        
        # get the virtual devices
        self.virtual_devices = []
        for _ in range(virtual_number):
            self.virtual_devices += self.devices
        
        # move the submodel to the gpu
        self.partitions, self.sub_partitions = self._move_model(module)
        
        # add hook of the model
        
        # TODO(fyy) how much cpu mems?
        if half_mode:
            self.cpu_mem = ContiguousMemoryAllocator(1024 ** 3 * 5, torch.float16, 'cpu')
        else:
            self.cpu_mem = ContiguousMemoryAllocator(1024 ** 3 * 5, torch.float32, 'cpu')
        self.cpu_mem.pin_memory()
        self.transfer_future_activation = self._register(module)
                
        self._copy_streams: List[List[AbstractStream]] = []
        self._skip_layout = inspect_skip_layout(self.partitions)
        
        ########################### multible fwd
        self.n_multi_fwd_stable = False
        self.n_multi_fwd = 1
        self.last_run_time = None
        self.last_n_multi_fwd = 1
        
        self.compute_stream: Dict[torch.device, List] = {}
        for device in self.devices:
            self.compute_stream[device] = []
            for _ in range(self.chunks):
                self.compute_stream[device].append(torch.cuda.Stream(device))   
                
        mobius_logger("Mobius finish initialization.")
        
    def get_first_device(self):
        return self.devices[0]
    
    def _move_model(self, init_module):
        partitions = []
        sub_partitions = [[] for _ in self.devices]
        
        layers: NamedModules = OrderedDict()
        n_layers = 0
        virtual_device_idx = 0
        for name, layer in init_module.named_children():
            layers[name] = layer
            n_layers += 1
            
            if n_layers == len(self.partition_idx[virtual_device_idx]):
                partition = nn.Sequential(layers)
                partitions.append(partition)
                
                sub_partitions[virtual_device_idx % len(self.devices)].append(partition)
                
                layers.clear()
                virtual_device_idx += 1
                n_layers = 0
                
        if n_layers != 0:
            partition = nn.Sequential(layers)
            partitions.append(partition)
            sub_partitions[virtual_device_idx % len(self.devices)].append(partition)
                
        return cast(List[nn.Sequential], nn.ModuleList(partitions)), sub_partitions
    
    def _register(self, init_module):
        
        transfer_future_activation = {}       
        
        for j, gpu_partitions in enumerate(self.sub_partitions):
            
            device = self.devices[j]
            upload_stream = torch.cuda.Stream(device)
            offload_stream = torch.cuda.Stream(device)
            device_mem = self.device_mems[device]
            
            # for each sub partition in a single gpu
            for i, gpu_partition in enumerate(gpu_partitions):
                first_child = None
                for num_child, child in gpu_partition.named_children():
                    
                    if first_child == None:
                        first_child = child
                    
                    # registe the module
                    register_mobius_module(
                               module=child, 
                               microbatch_num=self.chunks, 
                               upload_stream=upload_stream, 
                               offload_stream=offload_stream, 
                               device_mem=device_mem, 
                               cpu_mem=self.cpu_mem,
                               device_mems=self.device_mems)
                    
                    # registe the tensors
                    for name_tensor, tensor in child.named_parameters(recurse=True):
                            
                        register_mobius_param_in_cpu(
                                         tensor, 
                                         upload_stream, 
                                         offload_stream, 
                                         device_mem, 
                                         device,
                                         i == 0,
                                         i == len(gpu_partitions) - 1,
                                         self.half_mode,)
                        
                    child.to(device)
                    
                # add prefetch
                # backward
                prefetch_num = 1
                if self.noprefetch:
                    prefetch_num = 0
                    torchmobius.hook.NO_PREFTECH = True
                    
                if i - prefetch_num >= 0:
                    for _, next_tensor in gpu_partitions[i - prefetch_num].named_parameters(recurse=True):
                        first_child.mobius_module_attr.bind_bwd_tensor(next_tensor, 0)
                        
                    for _, future_module in gpu_partitions[i - prefetch_num].named_children():
                        # FIXME(fyy) multiple microbatch forward ???
                        for batch_i in range(self.chunks):
                            # NOTE(fyy) when to upload activation is a problem
                            # if too early, there is not enough space
                            transfer_future_activation[tuple([j * i, batch_i])] = tuple([child, 0])
                
                # forward
                if i + prefetch_num < len(gpu_partitions):
                    for _, next_tensor in gpu_partitions[i + prefetch_num].named_parameters(recurse=True):
                        first_child.mobius_module_attr.bind_fwd_tensor(next_tensor, 0)
                        
                first_child.mobius_module_attr.sort_tensor_by_micro_batch()
                    
        
        register_hooks_on_module(init_module, transfer_future_activation)
        return transfer_future_activation
    
    
    
    def parameters(self):
        params = []
        
        for param in self.partitions.parameters():
            params.append(param.mobius_tensor_attr.cpu_param_tensor)
            param.mobius_tensor_attr.cpu_param_tensor.requires_grad = True
        return params

    
    def DEBUG(self):
        count = 0
        for param in self.partitions.parameters():
            count += 1
            if count == 20:
                print(param.mobius_tensor_attr.cpu_param_tensor[0])
                print(param.mobius_tensor_attr.cpu_param_tensor.grad[0])
                break
    
    def zero_grad(self, set_to_none: bool = False):
        for param in self.partitions.parameters():
            if param.mobius_tensor_attr.cpu_param_tensor.grad is not None:
                if set_to_none:
                    param.mobius_tensor_attr.cpu_param_tensor.grad = None
                else:
                    if param.mobius_tensor_attr.cpu_param_tensor.grad.grad_fn is not None:
                        param.mobius_tensor_attr.cpu_param_tensor.grad.detach_()
                    else:
                        param.mobius_tensor_attr.cpu_param_tensor.grad.requires_grad_(False)
                    param.mobius_tensor_attr.cpu_param_tensor.grad.zero_()
        

    # Mobius manages the data and model moving between kinds of device
    # Any moving behaviors should be denied.
    def cuda(self, device: Optional[Device] = None) -> 'Mobius':
        raise MOVING_DENIED

    def cpu(self) -> 'Mobius':
        raise MOVING_DENIED

    def to(self, *args: Any, **kwargs: Any) -> 'Mobius':
        raise MOVING_DENIED

    def _ensure_copy_streams(self) -> List[List[AbstractStream]]:
        if not self._copy_streams:
            # activation copy have higher priority
            for device in self.devices:
                self._copy_streams.append([torch.cuda.Stream(device, priority=1) for _ in range(self.chunks * 2)])
        return self._copy_streams

            
    def forward(self, input: TensorOrTensors, _labels=None, fix_head_number = 1) -> TensorOrTensors:  # type: ignore
        
        torchmobius.hook.FORWARD_FLAG = True

        microbatch.check(input)

        if not self.devices:
            # Empty sequential module is not illegal.
            return input

        # Separate CUDA streams for copy.
        copy_streams = self._ensure_copy_streams()

        # The micro-batch index where the checkpointing stops.
        if self.training:
            checkpoint_stop = {
                'always': self.chunks,
                'except_last': self.chunks-1,
                'never': 0,
            }[self.checkpoint]
        else:
            checkpoint_stop = 0
            
        if fix_head_number != -1:
            # Run pipeline parallelism.
            pipeline = MobiusPipeline(  input,
                                        self.partitions,
                                        self.compute_stream,
                                        self.devices,
                                        self.virtual_devices,
                                        copy_streams,
                                        self._skip_layout,
                                        checkpoint_stop,
                                        self.transfer_future_activation,
                                        self.act_upload_streams_dic, 
                                        self.act_offload_streams_dic,
                                        fix_head_number)
            
            self.n_multi_fwd = fix_head_number
            
            torch.cuda.nvtx.range_push("fwd")
            tick = time.time()
            batches = pipeline.run()
            tock = time.time()
            torch.cuda.nvtx.range_pop()
            
            run_time = tock - tick
            # mobius_logger(f"Mobius forward time: {run_time}")
            
        else:
            # Run pipeline parallelism.
            pipeline = MobiusPipeline(  input,
                                        self.partitions,
                                        self.compute_stream,
                                        self.devices,
                                        self.virtual_devices,
                                        copy_streams,
                                        self._skip_layout,
                                        checkpoint_stop,
                                        self.transfer_future_activation,
                                        self.act_upload_streams_dic, 
                                        self.act_offload_streams_dic,
                                        self.n_multi_fwd)
            
            if not self.n_multi_fwd_stable:
                tick = time.time()
                batches = pipeline.run()
                tock = time.time()
                
                run_time = tock - tick
                if self.last_run_time == None:
                    self.last_run_time = run_time
                    self.last_n_multi_fwd = self.n_multi_fwd
                    self.n_multi_fwd += 1
                else:
                    if self.last_run_time > run_time:
                        self.last_run_time = run_time
                        self.last_n_multi_fwd = self.n_multi_fwd
                        self.n_multi_fwd += 1
                    else:
                        self.n_multi_fwd -= 1
                        self.n_multi_fwd_stable = True

                if self.max_microbatch_param <= self.n_multi_fwd:
                    self.n_multi_fwd_stable = True
                    self.n_multi_fwd = self.max_microbatch_param
                    
                mobius_logger(f"Mobius microbatch parallelism: {self.n_multi_fwd_stable} time: {run_time} {self.n_multi_fwd}")
            else:
                batches = pipeline.run()
                

        # Merge the micro-batches into one mini-batch.
        output = microbatch.gather(batches)

        for p in self.partitions:
            for m in p:
                m.mobius_module_attr.sort_tensor_by_micro_batch()
                m.mobius_module_attr.clean_mircrobatch_count()

        torchmobius.hook.FORWARD_FLAG = False

        return output
