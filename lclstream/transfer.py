from subprocess import Popen
from typing import Optional, List,Dict, Any, Awaitable, Tuple
import time
import asyncio

from pynng import Push0 # type: ignore[import-untyped]
import zfpy # type: ignore[import-untyped]

from .models import DataRequest
from .psana_img_src import PsanaImgSrc


def serialize(data) -> bytes: return zfpy.compress_numpy(data,
        write_header=True)
    # inverse = zfpy.decompress_numpy(buf)

TransferStats = Tuple[int,float,float,float]

def send_experiment(exp : str, run : int, access_mode : str, detector_name :
        str, mode : str, addr : str) -> TransferStats:

    assert access_mode in ['idx', 'smd'], "Access mode should be one of: idx, smd"
     
    if access_mode == "idx":
        mpi_pool_size = 3  # Hardcoding the mpi pool size for now
        cmd=(f"psana_push.py -e {exp} -r {run} -d {detector_name} "
             f"-m {mode} -a {addr} -c {access_mode}")
    else:
        mpi_pool_size = 3  # Hardcoding the mpi pool size for now
        cmd=(f"mpirun -np {mpi_pool_size} psana_push.py -e {exp} "
             f"-r {run} -d {detector_name} -m {mode} -a {addr} "
             f"-c {access_mode}")
    proc = Popen(cmd, shell=True)
    proc.wait()


async def send_experiment_async(exp : str,
                                run : int,
                                access_mode : str,
                                detector_name : str,
                                mode : str,
                                addr : str) -> TransferStats:
    loop = asyncio.get_running_loop()
    # TODO: separate the executor into its own
    # module or use threads...
    return await loop.run_in_executor(
        None,
        send_experiment,
        exp, run, access_mode, detector_name, mode, addr
    )

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
        self.value = await send_experiment_async(
            req.exp,
            req.run,
            req.access_mode,
            req.detector_name,
            req.mode,
            req.addr )
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

