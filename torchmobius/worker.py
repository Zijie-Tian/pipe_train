"""Multithreading in pipeline parallelism."""
from contextlib import contextmanager
from queue import Queue
import sys
from threading import Thread
from types import TracebackType
from typing import (TYPE_CHECKING, Callable, Dict, Generator, List, Optional, Tuple, Type, Union,
                    cast)

import torch

from torchmobius.microbatch import Batch
from torchmobius.stream import AbstractStream, use_device, use_stream

__all__: List[str] = []


ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]

# Queue is generic only in stubs.
# https://mypy.readthedocs.io/en/latest/common_issues.html#using-classes-that-are-generic-in-stubs-but-not-at-runtime
if TYPE_CHECKING:
    InQueue = Queue[Optional['Task']]
    OutQueue = Queue[Tuple[bool, Union[Tuple['Task', Batch], ExcInfo, None]]]
else:
    InQueue = Queue
    OutQueue = Queue


class Task:
    """A task represents how to compute a micro-batch on a partition.

    It consists of two parts: :meth:`compute` and :meth:`finalize`.
    :meth:`compute` should be executed in worker threads concurrently.
    :meth:`finalize` should be executed after when worker threads complete to
    execute :meth:`compute`.

    :meth:`compute` might be boosted by worker threads. Because it produces
    several CUDA API calls by user code. In PyTorch, parallel CUDA API calls
    are not serialized through GIL. So more than one CUDA API call can be
    produced at the same time.

    """

    def __init__(self,
                 stream: AbstractStream,
                 *,
                 compute: Callable[[], Batch],
                 finalize: Optional[Callable[[Batch], None]],
                 ) -> None:
        self.stream = stream
        self._compute = compute
        self._finalize = finalize

    def compute(self) -> Batch:
        torch.cuda.nvtx.range_push('compute')
        with use_stream(self.stream):
            rc = self._compute()
        torch.cuda.nvtx.range_pop()
        return rc

    def finalize(self, batch: Batch) -> None:
        if self._finalize is None:
            return
        with use_stream(self.stream):
            self._finalize(batch)


def worker(in_queue: InQueue,
           out_queue: OutQueue,
           device: torch.device,
           grad_mode: bool,
           ) -> None:
    """The main loop of a worker thread."""
    torch.set_grad_enabled(grad_mode)

    with use_device(device):
        while True:
            task = in_queue.get()

            if task is None:
                break

            try:
                batch = task.compute()
            except Exception:
                exc_info = cast(ExcInfo, sys.exc_info())
                out_queue.put((False, exc_info))
                continue

            out_queue.put((True, (task, batch)))

    done = (False, None)
    out_queue.put(done)

@contextmanager
def spawn_workers(devices: List[torch.device], step: int=1) -> Generator[Tuple[List[List[InQueue]], List[List[OutQueue]]], None, None]:
    """Spawns worker threads. A worker thread is bound to a device."""
    in_queues: List[InQueue] = []
    out_queues: List[OutQueue] = []

    # Spawn workers.
    workers: Dict[torch.device, List[Tuple[InQueue, OutQueue]]] = {}

    def normalize_device(device: torch.device) -> torch.device:
        if device.type == 'cuda' and device.index is None:
            return torch.device('cuda', index=torch.cuda.current_device())

        if device.type == 'cpu' and device.index is not None:
            return torch.device('cpu')

        return device

    for device in devices:
        device = normalize_device(device)

        try:
            in_queue, out_queue = workers[device]
        except KeyError:
            in_queue = []
            out_queue = []
            for i in range(step):
                in_q = Queue()
                out_q = Queue()
                
                in_queue.append(in_q)
                out_queue.append(out_q)
                if device not in workers.keys():
                    workers[device] = []
                workers[device].append((in_q, out_q))

                t = Thread(
                    target=worker,
                    args=(in_q, out_q, device, torch.is_grad_enabled()),
                    daemon=True,
                )
                t.start()

        in_queues.append(in_queue)
        out_queues.append(out_queue)

    try:
        yield (in_queues, out_queues)
    finally:
        # Close workers.
        for in_queue in in_queues:
            for in_q in set(in_queue):
                in_q.put(None)

        # Join running workers.
        for out_queue in out_queues:
            running = set(out_queue)

            while running:
                out_q = running.pop()
                ok, payload = out_q.get()

                done = (False, None)
                if (ok, payload) == done:
                    continue

                running.add(out_q)
        