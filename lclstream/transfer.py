from subprocess import Popen, PIPE, STDOUT
from typing import Optional, List, Dict, Any, Awaitable, Tuple
import time
import asyncio

from pynng import Push0 # type: ignore[import-untyped]
import zfpy # type: ignore[import-untyped]

from .models import DataRequest
from .psana_img_src import PsanaImgSrc

def serialize(data) -> bytes:
    return zfpy.compress_numpy(data, write_header=True)
    # inverse = zfpy.decompress_numpy(buf)

TransferStats = Tuple[int,float,float,float]

def send_experiment(exp : str,
                    run : int,
                    access_mode : str,
                    detector_name : str,
                    mode : str,
                    addr : str) -> TransferStats:

    assert access_mode in ['idx', 'smd'], "Access mode should be one of: idx, smd"

    if mode == "idx":
        ps = PsanaImgSrc(exp, run, access_mode, detector_name)
        start = time.time()
        n = 0
        mbyte = 0 # uncompressed
        nbyte = 0 # sent
        send_opts = {
            "send_buffer_size": 32 # send blocks if 32 messages queue up
        }
        with Push0(dial=addr, **send_opts) as push:
            for img in ps(mode):
                buf = serialize(img)
                n += 1
                mbyte += img.nbytes
                nbyte += len(buf)
                push.send(buf)
        t = time.time() - start
    else:
        mpi_pool_size = 3  # Hardcoding the mpi pool size for now
        proc = Popen(["mpirun",
                     f"-n{mpi_pool_size}",
                      "python",
                      "-u",
                      "mpi_psana.py",
                     f"-e={exp}",
                     f"-r={run}",
                     f"-d={detector}",
                     f"-m={mode}",
                     f"-a={addr}"],
                     stdin=PIPE,
                     stdout=PIPE,
                     stderr=STDOUT)
        print(p.stdout.read())
        print(p.stderr.read())

    # print(f"Sent {n} messages in {t} seconds: {nbyte/t/1024**2} MB/sec ({nbyte*100/mbyte}% compression).")
    # return (n, mbyte, nbyte, t)

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

