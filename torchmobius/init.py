import torch
from typing import (TYPE_CHECKING, Callable, Dict, Generator, List, Optional, Tuple, Type, Union,
                    cast)

from torchmobius.hook import register_hooks_on_module #, generate_activation_dic, used_activation_dic
from torchmobius.attribute import (MobiusModuleAttribute, MobiusTensorAttribute,
                                    register_mobius_module, register_mobius_param_in_cpu, register_mobius_param_in_gpu)
from torchmobius.schedule import MobiusSchedulerMeta
from torchmobius.stream import AbstractStream, new_stream
from torchmobius.memory import ContiguousMemoryAllocator

def mobius_init(model: torch.nn.Sequential,
                device_mems: Dict[torch.device, ContiguousMemoryAllocator],
                cpu_mem: ContiguousMemoryAllocator,
                half_mode=False
                ):
    
    microbatch_num = schedule.batch_number
    fwd_schedule    = schedule.transfer_fwd_future_param
    bwd_schedule    = schedule.transfer_bwd_future_param
    act_schedule    = schedule.transfer_future_activation
    device_mapping  = schedule.device_mapping

    heads    = schedule.heads
    tails    = schedule.tails

    for name, child in model.named_children():
        device = device_mapping[child]
        upload_stream = torch.cuda.Stream(device)
        offload_stream = torch.cuda.Stream(device)
        device_mem = device_mems[
            device_mapping[child]
        ]

        register_mobius_module(module=child, 
                               microbatch_num=microbatch_num, 
                               upload_stream=upload_stream, 
                               offload_stream=offload_stream, 
                               device_mem=device_mem, 
                               cpu_mem=cpu_mem,
                               device_mems=device_mems)
        
        
        for _, tensor in child.named_parameters(recurse=True):
            register_mobius_param_in_cpu(tensor, 
                                         upload_stream, 
                                         offload_stream, 
                                         device_mem, 
                                         device,
                                         child in heads,
                                         child in tails,
                                         half_mode
                                    )
        child = child.to(device)


    # bind the tensor A to the tensor responsible for transferinf A
    for _, child in model.named_children():
        if child in fwd_schedule.keys():
            fwd_transfer_module, fwd_microbatch_num = fwd_schedule[child]
            for _, tensor in child.named_parameters(recurse=True):
                fwd_transfer_module.mobius_module_attr.bind_fwd_tensor(tensor, fwd_microbatch_num)

        if child in bwd_schedule.keys(): 
            bwd_transfer_module, bwd_microbatch_num = bwd_schedule[child]
            for _, tensor in child.named_parameters(recurse=True):
                bwd_transfer_module.mobius_module_attr.bind_bwd_tensor(tensor, bwd_microbatch_num)
        

    for _, child in model.named_children():
        child.mobius_module_attr.sort_tensor_by_micro_batch()

    register_hooks_on_module(model, act_schedule)
