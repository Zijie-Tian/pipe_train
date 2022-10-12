"""Attributes of Mobius layers"""
import torch
from torch import Tensor

import copy
from enum import Enum
from typing import (TYPE_CHECKING, Callable, Dict, Generator, List, Optional, Tuple, Type, Union,
                    cast)

import threading

from torchmobius.stream import (AbstractStream, current_stream, get_device, record_stream,
                               use_stream, wait_stream)
from torchmobius.copy import Copy, Wait
from torchmobius.memory import ContiguousMemoryAllocator

class MobiusPosistion(Enum):
    CPU = 1
    GPU = 2
                               
Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

FORWARD_MICROBTAH_NUM = 0
BACKWARD_MICROBATCH_NUM = 0

class MobiusModuleAttribute(object):
    def __init__(self,
                 microbatch_num: int,
                 upload_stream,
                 offload_stream,
                 device_mem: ContiguousMemoryAllocator,
                 cpu_mem: ContiguousMemoryAllocator,
                 device_mems
                 ):

        self.microbatch_num = microbatch_num
        self.microbatch_count = 0

        self.upload_stream = upload_stream
        self.offload_stream = offload_stream
        self.device_mem = device_mem
        self.cpu_mem = cpu_mem
        self.device_mems = device_mems

        self.transfer_fwd_tensor: List[Tuple[TensorOrTensors, int]] = []
        self.transfer_bwd_tensor: List[Tuple[TensorOrTensors, int]] = []
        self.transfer_activation: List[Tuple[TensorOrTensors, int]] = []

        self.transfer_fwd_micro_batch_num_front = 0
        self.transfer_bwd_micro_batch_num_front = 0
        self.transfer_micro_batch_num_for_activation_front = 0
        
        self.mutex = threading.Lock()

    def debug(self):
        for _, i in self.transfer_activation:
            print(i) 

    def get_device(self):
        return self.device_mem.device

    def get_microbatch_count(self):
        return self.microbatch_count

    def inc_microbatch_count(self):
         with self.mutex:
             self.microbatch_count += 1
             if self.microbatch_num == self.microbatch_count:
                 self.microbatch_count = 0
                 return True
             return False
    
    def inc_microbatch_count_forward(self):
        with self.mutex:
            self.microbatch_count += 1
            if FORWARD_MICROBTAH_NUM == self.microbatch_count:
                self.microbatch_count = 0
                return True
            return False
        
    def inc_microbatch_count_backward(self):
        with self.mutex:
            self.microbatch_count += 1
            if BACKWARD_MICROBATCH_NUM == self.microbatch_count:
                self.microbatch_count = 0
                return True
            return False

    def clean_mircrobatch_count(self):
        self.microbatch_count = 0
        self.transfer_fwd_micro_batch_num_front = 0
        self.transfer_bwd_micro_batch_num_front = 0
        self.transfer_micro_batch_num_for_activation_front = 0

    def bind_fwd_tensor(self, tensor, micro_batch):
        self.transfer_fwd_tensor.append(tuple([tensor, micro_batch]))

    def bind_bwd_tensor(self, tensor, micro_batch):
        self.transfer_bwd_tensor.append(tuple([tensor, micro_batch]))
    
    def bind_and_free_activation(self, 
                                 activation, transfer_time,
                                 upload_stream, offload_stream):
        self.transfer_activation.append(tuple([activation, transfer_time]))
        
        register_mobius_param_in_gpu(activation, 
                                    upload_stream, 
                                    offload_stream, 
                                    self.device_mems[activation.device], 
                                    self.cpu_mem,
                                    activation.device,
                                )
        
    def sort_tensor_by_micro_batch(self):
        def _sort_by_microbatch(elem: Tuple[TensorOrTensors, int]):
            return elem[1]

        self.transfer_fwd_tensor.sort(key=_sort_by_microbatch)
        self.transfer_bwd_tensor.sort(key=_sort_by_microbatch)
        self.transfer_activation.sort(key=_sort_by_microbatch)


    def _is_fwd_transfer_time(self):
        if len(self.transfer_fwd_tensor) <= self.transfer_fwd_micro_batch_num_front:
            return False

        if self.microbatch_count == self.transfer_fwd_tensor[self.transfer_fwd_micro_batch_num_front][1]:
            return True

        return False   

    def _is_bwd_transfer_time(self):
        if len(self.transfer_bwd_tensor) <= self.transfer_bwd_micro_batch_num_front:
            return False

        if self.microbatch_count == self.transfer_bwd_tensor[self.transfer_bwd_micro_batch_num_front][1]:
            return True

        return False   

    def _is_act_transfer_time(self):
        if len(self.transfer_activation) <= self.transfer_micro_batch_num_for_activation_front:
            return False

        return True

        if self.microbatch_count == self.transfer_activation[self.transfer_micro_batch_num_for_activation_front][1]:
            return True

        return False 

    def _transfer_future_param(self, transfer_param):
        #TODO(fyy) if no space, delay transfer
        if type(transfer_param) is tuple:
            for tensor in self.transfer_param:
                tensor.mobius_tensor_attr.upload_param()
        else:
            transfer_param.mobius_tensor_attr.upload_param()

    def transfer_fwd_future_param(self):
        with self.mutex:
            while self._is_fwd_transfer_time():
                transfer_tensor = self.transfer_fwd_tensor[self.transfer_fwd_micro_batch_num_front][0]
                self._transfer_future_param(transfer_tensor)
                self.transfer_fwd_micro_batch_num_front += 1

    def transfer_bwd_future_param(self):
        while self._is_bwd_transfer_time():
            transfer_tensor = self.transfer_bwd_tensor[self.transfer_bwd_micro_batch_num_front][0]
            self._transfer_future_param(transfer_tensor)
            self.transfer_bwd_micro_batch_num_front += 1

    def transfer_future_activation(self):
        while self._is_act_transfer_time():
            transfer_activation = self.transfer_activation[self.transfer_micro_batch_num_for_activation_front][0]
            transfer_activation.mobius_tensor_attr.upload_param()
            self.transfer_micro_batch_num_for_activation_front += 1

    def unbind_activation(self):
        self.transfer_activation.clear()


# NOTE: 
# the gpu initial tensor shape must be the same as the original one.
#  tensor.data = data will release the space of the tensor.data
class MobiusTensorAttribute(object):
    def __init__(self,
                 param: torch.nn.Parameter,
                 cpu_param_tensor: torch.Tensor,
                 device_param_tensor: torch.Tensor,
                 upload_stream: AbstractStream,
                 offload_stream: AbstractStream,
                 device_mem: ContiguousMemoryAllocator,
                 device: torch.device,

                 forward_offload=True,
                 backward_offload=True,
                 
                 half_mode=False,
                 
                ):
        self.param = param
        self.numel = param.numel()
        self.shape = param.shape
        self.device = device
        
        self.cpu_param_tensor = cpu_param_tensor
        self.device_param_tensor = device_param_tensor
        self.activation_tensor_id = 0

        self.device_tmp_param_tensor = None
        self.device_tmp_grad_tensor = None

        self.position =  MobiusPosistion.CPU
        self.device_mem = device_mem

        # layer transfer stream
        self.upload_stream = upload_stream
        self.offload_stream = offload_stream

        self.forward_offload = forward_offload
        self.backward_offload = backward_offload
        
        
        self.half_mode = half_mode

        # activation info
        self.is_activation = False
        self.cpu_mem: ContiguousMemoryAllocator = None

        # schedule
        # self.start_uploadwait_upload = torch.cuda.Event(enable_timing=False)
        self.end_upload_event = torch.cuda.Event(enable_timing=False)
        # self.start_compute_event = torch.cuda.Event(enable_timing=False)


    def get_position(self):
        return self.position

    def wait_upload(self):
        self.end_upload_event.wait()
        pass

    def wait_compute(self):
        # self.start_compute_event.wait()
        pass

    def start_compute(self):
        # self.start_compute_event.record()
        pass

    # def record_profile_info(self, phase):
    #     self.prof_info[phase + '_transfer_time'] = self.start_upload_event.elapsed_time(self.end_upload_event)
    #     self.prof_info[phase + '_compute_wait_time'] = self.end_upload_event.elapsed_time(self.start_compute_event)
    
    # called by other mobius tensor 
    # NOTE: only one data transfer task in the stream
    @torch.no_grad()
    def upload_param(self):
        if self.position == MobiusPosistion.GPU:
            return 
        torch.cuda.nvtx.range_push(f'upload param')
        with use_stream(self.upload_stream):
            # get a tensor from allocator
            if  self.device_mem.dtype == self.cpu_param_tensor.dtype:

                self.device_tmp_param_tensor = self.device_mem.allocate_tensor(self.numel)
                self.activation_tensor_id = id(self.device_tmp_param_tensor)

                # self.start_upload_event.record()

                # transfer the data from the cpu tensor
                self.device_tmp_param_tensor.copy_(self.cpu_param_tensor.view([self.numel]), non_blocking=True)

                # assign the buffer to the working tensor
                self.device_mem.assign_to_param(
                    self.device_tmp_param_tensor, 
                    self.device_param_tensor, 
                    self.numel, self.shape
                )

                # record the event
                # self.end_upload_event.record()

            else:
                # TODO add prof flag
                # self.start_upload_event.record()
                self.device_param_tensor.data = self.cpu_param_tensor.to(self.device, non_blocking=True).view(self.shape)
                # self.end_upload_event.record()


            # if activation, cpu buffer should be freed
            if self.is_activation:
                if self.cpu_mem != None:
                    self.cpu_mem.release_tensor(self.cpu_param_tensor)
                    self.cpu_param_tensor.data = torch.empty(0, 
                        dtype=self.cpu_param_tensor.data.dtype, 
                        device=self.cpu_param_tensor.data.device) 

            # flag the position of the tensor.data
            self.position = MobiusPosistion.GPU
        torch.cuda.nvtx.range_pop()

    @torch.no_grad()
    def free_fwd_param(self):
        # if self.forward_offload:
        # TODO(fyy) the parameter should update
        #       or it update in the cpu (need offload)
        #       or it update in the gpu (do not need offload)
        self.free_param()

    @torch.no_grad()   
    def free_param(self):
        if self.position == MobiusPosistion.CPU:
            return 

        self.position = MobiusPosistion.CPU
        if  self.device_mem.dtype == self.cpu_param_tensor.dtype:
            self.device_mem.release_tensor(self.device_tmp_param_tensor)
        self.device_param_tensor.data = torch.empty(0, 
                        dtype=self.device_param_tensor.data.dtype, 
                        device=self.device_param_tensor.data.device) 

    @torch.no_grad()
    def free_param_and_offload_grad(self):
        if self.position == MobiusPosistion.CPU:
            return 

        with use_stream(self.offload_stream):
            # transfer grad     
            self.cpu_param_tensor.grad.copy_(self.device_param_tensor.grad, non_blocking=True)

            # set the grad to None
            # if self.device_param_tensor.grad.shape == torch.Size([50400, 4096]):
            #     tmp = self.device_param_tensor.grad.to("cuda:2")
            #     print("GPU", torch.norm(self.device_param_tensor.grad, 2))
            #     print("GPU 2", torch.norm(tmp, 2))
            #     print("GPU sum", self.device_param_tensor.grad.sum())
            #     tmp = self.cpu_param_tensor.grad.to("cuda:2")
            #     print("CPU", torch.norm(self.cpu_param_tensor.grad, 2, dtype=torch.float32))
            #     print("CPU 2", torch.norm(tmp, 2))
            #     print("CPU sum", tmp.sum())
                
                
            # self.offload_stream.synchronize()
            self.device_param_tensor.grad = None
            
            # if self.backward_offload:
            self.free_param()
        
        
    def sync_upload(self):
        self.upload_stream.synchronize()

# step1. copy the module's param in cpu
# step2. send the param in cpu to gpu
# step3. clear the gpu's tensor
def register_mobius_param_in_cpu(param, 
                                 upload_stream, 
                                 offload_stream, 
                                 device_mem, 
                                 device,
                                 is_head=False,
                                 is_tail=False,
                                 half_mode=False,):
    cpu_tmp_tensor = param.clone().detach().pin_memory()
    cpu_tmp_tensor.grad = torch.empty(cpu_tmp_tensor.shape, dtype=param.dtype).pin_memory()
    cpu_tmp_tensor.grad.zero_()

    param.data = torch.empty(0, dtype=param.dtype, device=param.device)

    param.mobius_tensor_attr = MobiusTensorAttribute(
                                                cpu_tmp_tensor, 
                                                cpu_tmp_tensor, 
                                                param, 
                                                upload_stream, 
                                                offload_stream, 
                                                device_mem, 
                                                device,
                                                not is_tail,
                                                not is_head,
                                                half_mode,)

    if is_head:
        param.mobius_tensor_attr.upload_param()
    


def register_mobius_param_in_gpu(param,
                                 upload_stream,
                                 offload_stream,
                                 device_mem,
                                 cpu_mem,
                                 device):
    with use_stream(offload_stream):
        if cpu_mem.dtype == param.dtype:
            cpu_tmp_tensor = cpu_mem.allocate_tensor(param.numel()).pin_memory()
            cpu_tmp_tensor.copy_(param.view([param.numel()]), non_blocking=True)
        else:
            cpu_tmp_tensor = torch.empty(param.shape, dtype=param.dtype).pin_memory()
            cpu_tmp_tensor.copy_(param, non_blocking=True)

        param.mobius_tensor_attr = MobiusTensorAttribute(
                                            param, 
                                            cpu_tmp_tensor, 
                                            param, 
                                            upload_stream, 
                                            offload_stream, 
                                            device_mem, 
                                            device)

        if cpu_mem.dtype == param.dtype:
            param.mobius_tensor_attr.cpu_mem = cpu_mem 

        param.mobius_tensor_attr.is_activation = True

        param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        offload_stream.synchronize()


def register_mobius_module(module,
                           microbatch_num,
                           upload_stream,
                           offload_stream,
                           device_mem,
                           cpu_mem,
                           device_mems,
                           ):
   module.mobius_module_attr = MobiusModuleAttribute(
                                        microbatch_num, 
                                        upload_stream, 
                                        offload_stream, 
                                        device_mem,
                                        cpu_mem,
                                        device_mems
                                        ) 