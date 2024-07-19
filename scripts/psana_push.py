#!/usr/bin/env python
from enum import Enum
from typing import Annotated

from pynng import Push0 # type: ignore[import-untyped]
import typer
import zfpy # type: ignore[import-untyped]

from lclstream.psana_img_src import PsanaImgSrc

def serialize(data) -> bytes:
    return zfpy.compress_numpy(data, write_header=True)


class ImageRetrievalMode(str, Enum):
    raw = "raw"
    calib = "calib"
    image = "image"
    mask = "mask"

class AccessMode(str, Enum):
    idx = "idx"
    smd = "smd"



def psana_push(
        experiment: Annotated[
            str,
            typer.Option("--experiment", "-e", help="Experiment identifier"),
        ],
        run: Annotated[
            int,
            typer.Option("--run", "-r", help="Run number"),
        ],
        detector: Annotated[
            str,
            typer.Option("--detector", "-d", help="Detector name"),
        ],
        mode: Annotated[
            ImageRetrievalMode,
            typer.Option("--mode", "-m", help="Image retrieval mode"),
        ],    
        addr: Annotated[
            str,
            typer.Option("--addr", "-a", help="Push socket base address"),
        ],
        access_mode: Annotated[
            AccessMode,
            typer.Option("--access_mode", "-c", help="Data access mode"),
        ],

):
    ps = PsanaImgSrc(experiment, run, "smd", detector)

    if access_mode == "smd":
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        addr_parts = addr.split(":")
        base_curr_addr=":".join(addr_parts[0:-1])
        baseport = int(addr_parts[-1])
        curr_port = baseport + rank
        curr_addr = f"{base_curr_addr}:{curr_port}"
        ps = PsanaImgSrc(experiment, run, "smd", detector)
    else:
        curr_addr = addr
        ps = PsanaImgSrc(experiment, run, "idx", detector)

    send_opts = {
        "send_buffer_size": 32 # send blocks if 32 messages queue up
    }
    with Push0(dial=curr_addr, **send_opts) as push:
        print(f"Starting push socket at {curr_addr}")
        for img in ps(mode):
            buf = serialize(img)
            push.send(buf)
 
if __name__ == "__main__":
    typer.run(psana_push)
