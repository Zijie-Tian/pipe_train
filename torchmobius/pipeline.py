"""The pipeline parallelism of GPipe."""
from queue import Queue
from types import TracebackType
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Type, Union, cast

import torch
from torch import Tensor, nn
from torchmobius import microbatch

from torchmobius.checkpoint import Checkpointing
from torchmobius.copy import Copy, Wait
from torchmobius.dependency import fork, join
from torchmobius.microbatch import Batch
from torchmobius.skip.layout import SkipLayout, inspect_skip_layout
from torchmobius.skip.tracker import SkipTrackerThroughPotals, use_skip_tracker
from torchmobius.stream import AbstractStream, current_stream, use_device, new_stream
from torchmobius.worker import Task, spawn_workers
import torchmobius.attribute

__all__: List[str] = []


Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]

# Queue is generic only in stubs.
# https://mypy.readthedocs.io/en/latest/common_issues.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
if TYPE_CHECKING:
    InQueue = Queue[Optional['Task']]
    OutQueue = Queue[Tuple[bool, Union[Tuple['Task', Batch], ExcInfo, None]]]
else:
    InQueue = Queue
    OutQueue = Queue


def depend(fork_from: Batch, join_to: Batch) -> None:
    fork_from[0], phony = fork(fork_from[0])
    join_to[0] = join(join_to[0], phony)


def copy(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Copy.apply(prev_stream, next_stream, *batch)


def wait(batch: Batch, prev_stream: AbstractStream, next_stream: AbstractStream) -> None:
    batch[:] = Wait.apply(prev_stream, next_stream, *batch)


def multi_fwd_clock_cycles(m: int, n: int, step: int=1) -> Iterable[List[Tuple[int, int]]]:
    """Generates schedules for each clock cycle."""
    # m: number of micro-batches
    # n: number of partitions
    # i: index of micro-batch
    # j: index of partition
    # k: clock number
    #
    # k (i,j) (i,j) (i,j)
    # - ----- ----- -----
    # 0 (0,0)
    # 1 (1,0) (0,1)
    # 2 (2,0) (1,1) (0,2)
    # 3       (2,1) (1,2)
    # 4             (2,2)
    
    import math
    n_m = m
    m = math.ceil(m / step)
    
    
    
    for k in range(m+n-1):
        rc = []
        for j in range(max(1+k-m, 0), min(1+k, n)):
            temp = []
            for i in range(step):
                if (k-j) * step + i < n_m:
                    temp.append((k-j) * step + i)
            rc.append((temp, j))
        yield rc

class MobiusPipeline:
    """The pipeline parallelism for GPipe."""

    def __init__(self,
                 input,
                 partitions: List[nn.Sequential],
                 compute_stream: Dict[torch.device, List],
                 physic_devices: Optional[List[torch.device]] = None,
                 virtual_devices: Optional[List[torch.device]] = None,
                 copy_streams: Optional[List[List[AbstractStream]]] = None,
                 skip_layout: Optional[SkipLayout] = None,
                 checkpoint_stop: int = 0,
                 transfer_future_activation = None,
                 upload_streams_dic = None,
                 offload_streams_dic = None,
                 n_multi_fwd = 1):

        self.n_multi_fwd = n_multi_fwd
        torchmobius.attribute.FORWARD_MICROBTAH_NUM = len(physic_devices) // self.n_multi_fwd
        torchmobius.attribute.BACKWARD_MICROBATCH_NUM = torchmobius.attribute.FORWARD_MICROBTAH_NUM
        # torchgpipe.attribute.BACKWARD_MICROBATCH_NUM = len(physic_devices)
        
        # microbatch compact
        # Divide a mini-batch into micro-batches.
        self.batches = microbatch.scatter(input, int(len(physic_devices) //  self.n_multi_fwd))

        self.partitions = partitions

        self.virtual_devices = virtual_devices
        self.physic_devices = physic_devices

        # get the virtual device index mapping to the physic indexing
        self.virtual_to_physic_map = []
        tmp_index = {}
        for idx, physic_device in enumerate(self.physic_devices):
            tmp_index[physic_device] = idx
        for virtual_device in self.virtual_devices:
            self.virtual_to_physic_map.append(tmp_index[virtual_device])

        self.copy_streams = copy_streams

        if skip_layout is None:
            skip_layout = inspect_skip_layout(partitions)

        self.skip_layout = skip_layout
        self.checkpoint_stop = checkpoint_stop

        self.transfer_future_activation = transfer_future_activation
        self.upload_streams_dic = upload_streams_dic
        self.offload_streams_dic = offload_streams_dic
        
        # computation 
        self.compute_stream = compute_stream                

    def run(self) -> None:
        """Runs pipeline parallelism.

        It modifies the given batches in place.

        """
        batches = self.batches
        partitions = self.partitions
        physic_devices = self.physic_devices
        skip_layout = self.skip_layout

        m = len(batches)
        n = len(partitions)

        skip_trackers = [SkipTrackerThroughPotals(skip_layout) for _ in batches]
        
        torchmobius.attribute.GRADIENT_OVERFLOW = False

        with spawn_workers(physic_devices, 1) as (in_queues, out_queues):
            for schedule in multi_fwd_clock_cycles(m, n, 1):
                self.fence(schedule, skip_trackers)
                self.compute(schedule, skip_trackers, in_queues, out_queues)
                
        return self.batches

    def fence(self,
              schedule: List[Tuple[int, int]],
              skip_trackers: List[SkipTrackerThroughPotals],
              ) -> None:
        """Copies micro-batches after computation for the previous
        micro-batches.
        """
        batches = self.batches
        copy_streams = self.copy_streams
        skip_layout = self.skip_layout

        for i_list, j in schedule:
            # Ensure that batches[i-1] is executed after batches[i] in
            # backpropagation by an explicit dependency.
            physic_device_index = self.virtual_to_physic_map[j]
            for i in i_list:
                
                # NOTE(mobius) anything happen?
                if i != 0:
                    depend(batches[i-1], batches[i])

                next_stream = copy_streams[physic_device_index][i]

                # for prev_j, ns, name in skip_layout.copy_policy(j):
                #     prev_stream = copy_streams[prev_j][i]
                #     skip_trackers[i].copy(batches[i], prev_stream, next_stream, ns, name)

                if j != 0:
                    prev_physic_device_index = self.virtual_to_physic_map[j-1]
                    prev_stream = copy_streams[prev_physic_device_index][i]
                    copy(batches[i], prev_stream, next_stream)

    def compute(self,
                schedule: List[Tuple[int, int]],
                skip_trackers: List[SkipTrackerThroughPotals],
                in_queues: List[InQueue],
                out_queues: List[OutQueue],
                ) -> None:
        """Runs tasks with synchronization to copy streams."""
        batches = self.batches
        partitions = self.partitions
        devices = self.virtual_devices
        copy_streams = self.copy_streams
        checkpoint_stop = self.checkpoint_stop

        n = len(partitions)
        streams = self.compute_stream
        exc_info: Optional[ExcInfo] = None

        # With checkpointing, the autograd graph looks like this diagram:
        # ┌─────┸──────┐
        # │    Copy    │
        # └─────┰──────┘   (fence)
        # ─ ─ ─ ╂ ─ ─ ─ ─ ─ ─ ─ ─ ─
        #       ┃          (compute)
        # ┌─────┸──────┐
        # │    Wait    │ [1] Synchronize the current stream with the copy stream.
        # └─────┰──────┘
        # ┌─────┸──────┐
        # │ Checkpoint │ [2] Compute a partition within checkpointing.
        # └─────┰──────┘
        # ┌─────┸──────┐
        # │    Wait    │ [3] Synchronize the copy stream with the current stream.
        # └─────┰──────┘
        #       ┠ ─ ─ ─ ┐
        #       ┃ ┌─────┴─────┐
        #       ┃ │ Recompute │ [4] Schedule the recomputation at backpropagation.
        #       ┃ └─────┬─────┘
        #       ┠ ─ ─ ─ ┘
        #       ┃
        # ─ ─ ─ ╂ ─ ─ ─ ─ ─ ─ ─ ─ ─
        # ┌─────┸──────┐   (fence)
        # │    Copy    │
        # └─────┰──────┘
        for i_list, j in schedule:
            partition = partitions[j]
            physic_device_index = self.virtual_to_physic_map[j]
            physic_device = self.physic_devices[physic_device_index]
            
            for step, i in enumerate(i_list):
                batch = batches[i]

                # Synchronize with the copied input. ([1] in the diagram)
                # NOTE(fyy) the computation stream waits the copy stream(prev)
                if j != 0:
                    wait(batch, copy_streams[physic_device_index][i], streams[physic_device][step])

            # Determine whether checkpointing or not.
            # NOTE(fyy) each batch will be checkpointed once
            checkpoint = (i < checkpoint_stop)
            if checkpoint:
                def function(input: TensorOrTensors,
                             partition: nn.Sequential = partition,
                             skip_tracker: SkipTrackerThroughPotals = skip_trackers[i],
                             ) -> TensorOrTensors:                    
                    with use_skip_tracker(skip_tracker):
                        # print("PIN_1: ", input)
                        # print("PIN_1_meta: ", input.shape)
                        rc = partition(input)
                        return rc
                
                for step, i in enumerate(i_list):
                    
                    bind_module = None 
                    transfer_time =  None
                    if tuple([j, i]) in self.transfer_future_activation.keys():
                        bind_module, transfer_time = self.transfer_future_activation[tuple([j, i])]

                    chk = Checkpointing(function, batches[i], 
                                        bind_module  =bind_module,
                                        transfer_time= transfer_time,
                                        upload_streams_dic= self.upload_streams_dic,
                                        offload_streams_dic= self.offload_streams_dic,
                                        microbatch_parallelism=1
                                        )
                    
                    task = Task(streams[physic_device][step], compute=chk.checkpoint, finalize=chk.recompute)
                    
                    # Compute tasks in parallel. ([2] in the diagram)
                    in_queues[physic_device_index][step].put(task)
                    del chk
                    
                del function

            else:
                assert(False, "Not finished")

            
            # in_queues[physic_device_index][0].put(task)

        for i_list, j in schedule:
            physic_device_index = self.virtual_to_physic_map[j]
            
            for step, i in enumerate(i_list):
                
                ok, payload = out_queues[physic_device_index][step].get()

                # Hold the first exception.
                if exc_info is not None:
                    continue
                elif not ok:
                    exc_info = cast(ExcInfo, payload)
                    continue

                task, batch = cast(Tuple[Task, Batch], payload)

                
                # The copy stream synchronizes to copy the output. ([3] in the
                # diagram)
                if j != n-1:
                    wait(batch, streams[physic_device][step], copy_streams[physic_device_index][i])

                # Finalize tasks. If checkpointing is enabled, here the
                # recomputation is scheduled at backpropagation. ([4] in the
                # diagram)
                with use_device(devices[physic_device_index]):
                    task.finalize(batch)
                
                batches[i] = batch

        # Fail at the first exception.
        if exc_info is not None:
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])
        


class Pipeline:
    def __init__(self) -> None:
        pass