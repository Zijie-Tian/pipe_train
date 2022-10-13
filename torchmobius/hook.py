import torch

from torchmobius.attribute import MobiusPosistion
from torchmobius.utils import print_memory_usage

FORWARD_FLAG = False
NO_PREFTECH = False

def _apply_to_tensors(module, grad_function, backward_function, tensor):
    if type(tensor) is tuple:
        touched_outputs = []
        for t in tensor:
            touched_output = _apply_to_tensors(
                module, grad_function, backward_function, t
            )
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(tensor) is torch.Tensor:
        return grad_function.apply(module, backward_function, tensor)
    else:
        return tensor


class PreBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        # print(f"Pre Backward: {ctx.module.__class__.__name__}")
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args


class PostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, post_backward_function, outputs):
        ctx.module = module
        ctx.post_backward_function = post_backward_function
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        # print(f"Post Backward: {ctx.module.__class__.__name__}")
        ctx.post_backward_function(ctx.module)
        return (None, None) + args

@torch.no_grad()
def _pre_module_forward_function(module, bind_dic):
    
    """
    use the code like this to free activation:

     - for k in generate_activation:
     -   generate_activation[k].data = torch.empty(0, dtype=generate_activation[k].data.dtype, device=generate_activation[k].data.device)
    """
     # step1. start transferring the future param
    if FORWARD_FLAG:
        module.mobius_module_attr.transfer_fwd_future_param()
    else:
        if NO_PREFTECH:
            module.mobius_module_attr.transfer_bwd_future_param()
            module.mobius_module_attr.transfer_future_activation()
            module.mobius_module_attr.unbind_activation()
            
    if NO_PREFTECH:        
        for _, param in module.named_parameters(recurse=True):
                param.mobius_tensor_attr.sync_upload()  
            
    # step0. wait for the param
    if module.mobius_module_attr.get_microbatch_count() == 0:
        # for _, param in module.named_parameters(recurse=True):
        #     param.mobius_tensor_attr.start_compute()

        for _, param in module.named_parameters(recurse=True):
            while True:
                if param.mobius_tensor_attr.get_position() == MobiusPosistion.GPU:
                    break
                # FIXME inference
                else:
                    param.mobius_tensor_attr.upload_param()
                    
                    # NOTE(fyy) dirty sync
                    # assert param.mobius_tensor_attr.device_param_tensor != None
                    # if torch.isnan(param.mobius_tensor_attr.device_param_tensor).any():
                    #     print("device_param_tensor : ", param.mobius_tensor_attr.device_param_tensor)
                    #     exit(-1)

        # for _, param in module.named_parameters(recurse=True):
        #     param.mobius_tensor_attr.wait_upload()

    # # step1. start transferring the future param
    # if FORWARD_FLAG:
    #     module.mobius_module_attr.transfer_fwd_future_param()
        
    
@torch.no_grad()
def _post_module_forward_function(module, inputs, outputs, bind_dic):
    if not FORWARD_FLAG:
        return 

    # step0. release the param
    if module.mobius_module_attr.inc_microbatch_count_forward():
        for _, param in module.named_parameters(recurse=True):
            param.mobius_tensor_attr.free_fwd_param()
        # clean the mirco batch counter
        module.mobius_module_attr.clean_mircrobatch_count()
    
    
@torch.no_grad()
def _pre_module_backward_function(module, output):       
    # step1. start transfering the future param and activation
    if not NO_PREFTECH:
        module.mobius_module_attr.transfer_bwd_future_param()
        module.mobius_module_attr.transfer_future_activation()

        # step2. unbind the activations to the module
        module.mobius_module_attr.unbind_activation()
    
    # step0. wait for the param
    if module.mobius_module_attr.get_microbatch_count() == 0:
        # for _, param in module.named_parameters(recurse=True):
        #     param.mobius_tensor_attr.start_compute()

        for _, param in module.named_parameters(recurse=True):
            while True:
                if param.mobius_tensor_attr.get_position() == MobiusPosistion.GPU:
                    break
                else:
                    param.mobius_tensor_attr.upload_param()
                    # if torch.isnan(param.mobius_tensor_attr.device_param_tensor).any():
                    #     print("device_param_tensor : ", param.mobius_tensor_attr.device_param_tensor)
                    #     exit(-1)



@torch.no_grad()
def _post_module_backward_function(module):
    if module.mobius_module_attr.inc_microbatch_count_backward():
        for _, param in module.named_parameters(recurse=True):
            param.mobius_tensor_attr.free_param_and_offload_grad()
        # clean the mirco batch counter
        module.mobius_module_attr.clean_mircrobatch_count() 
    

def register_hooks_on_module(module: torch.nn.Sequential,
                              bind_dic):
    for name, child in module.named_children():
        def _pre_forward_module_hook(module, inputs):
            _pre_module_forward_function(module, bind_dic)

        def _post_forward_module_hook(module, inputs, outputs):            
            _post_module_forward_function(module, inputs, outputs, bind_dic)

        def _pre_backward_module_hook(module, inputs, outputs):
            def _pre_module_backward_function_warp(module):
                _pre_module_backward_function(module, outputs) 

            return _apply_to_tensors(module, 
                              PreBackwardFunction, 
                              _pre_module_backward_function_warp, 
                              outputs
                            )

        def _post_backward_moduel_hook(module, inputs):
            return _apply_to_tensors(module, 
                              PostBackwardFunction, 
                              _post_module_backward_function, 
                              inputs
                            )

        child.register_forward_pre_hook(_pre_forward_module_hook)
        child.register_forward_hook(_post_forward_module_hook)

        child.register_forward_hook(_pre_backward_module_hook)
        child.register_forward_pre_hook(_post_backward_moduel_hook)
