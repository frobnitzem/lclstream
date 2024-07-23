from subprocess import Popen
from typing import Optional, List,Dict, Any, Awaitable, Tuple
import time
import asyncio

from pynng import Push0 # type: ignore[import-untyped]
import zfpy # type: ignore[import-untyped]

from .models import DataRequest, AccessMode, ImageRetrievalMode

def serialize(data) -> bytes: return zfpy.compress_numpy(data,
        write_header=True)
    # inverse = zfpy.decompress_numpy(buf)

TransferStats = Tuple[int,float,float,float]

def send_experiment(req : DataRequest) -> TransferStats:

    mpi_pool_size = 3  # Hardcoding the mpi pool size for now

    assert req.access_mode in [AccessMode.idx, AccessMode.smd], \
            "Access mode should be one of: idx, smd"
     
    cmd = ["psana_push", "-e", req.exp,
                         "-r", req.run,
                         "-d", req.detector_name,
                         "-m", req.mode.value,
                         "-a", req.addr,
                         "-c", access_mode.value]
    if req.access_mode != AccessMode.idx:
        cmd = ["mpirun", "-np", str(mpi_pool_size)] + cmd
    proc = Popen(cmd)
    proc.wait()
    return 1, 1.0, 1.0, 1.0

async def send_experiment_async(req : DataRequest) -> TransferStats:
    loop = asyncio.get_running_loop()
    # TODO: separate the executor into its own
    # module or use threads...
    return await loop.run_in_executor(None, send_experiment, req)

class Transfer:
    req   : DataRequest
    state : str
    coro  : Optional[Awaitable[TransferStats]]

    def __init__(self, request : DataRequest):
        self.request = request
        self.state = "initial"
        self.coro = None

    async def run(self):
        self.state = "active"
        req = self.request
        self.value = await send_experiment_async(req)
        self.state = "completed"

    def start(self) -> bool:
        self.coro = asyncio.create_task(self.run(), name="transfer")
        return True

    def cancel(self) -> bool:
        if self.coro is None or self.state == "completed":
            return False
        self.coro.cancel() # type: ignore[attr-defined]
        self.coro = None
        return True
    
    async def __call__(self) -> str:
        if self.coro is None:
            raise ValueError("Transfer.start() was not called")
        # The transfer can be awaited
        try:
            await self.coro
        except asyncio.CancelledError as e:
            return "canceled"
        except Exception as e:
            return "failed"
        return self.state

