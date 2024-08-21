#!/usr/bin/env python3

from io import BytesIO
from typing import Annotated, Iterable

from pynng import Push0 # type: ignore[import-untyped]
from pynng.exceptions import ConnectionRefused # type: ignore[import-untyped]

import numpy as np
import h5py # type: ignore[import-untyped]
import hdf5plugin # type: ignore[import-untyped]
import typer

from models import ImageRetrievalMode, AccessMode
from psana_img_src import PsanaImgSrc
import time
import json

import os

# Get the rank (process ID) from the environment variable
rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
# print("rank", rank)


class Hdf5FileWriter:
    """ This class sets up a writer for in-memory
        hdf5-format files.

        FIXME: this class may discard images
        at the end of an experiment run (due
        to fixed img_per_file rounding).
    """

    def __init__(self, img_per_file : int) -> None:
        self.img_per_file = img_per_file
        self.times = []
        self.gb_s = []

    def __call__(self, src : Iterable[np.ndarray]) -> Iterable[bytes]:
        """
        Returns an iterator over serialized hdf5 bytes.
        Each value yielded contains img_per_file images.

        Args:
            src: iterator over image arrays
        """
        while True:
            img = next(src)
            start = time.time()
            img_bytes = img.nbytes
            with BytesIO() as buffer:
                with h5py.File(buffer, 'w') as fh:
                    dataset = fh.create_dataset(
                        'data',
                        shape = (self.img_per_file,) + img.shape,
                        **hdf5plugin.Zfp()
                    )
                    dataset[0] = img
                    for idx in range(1, self.img_per_file):
                        img_bytes+=next(src).nbytes
                        dataset[idx] = next(src)
                end = time.time()
                execution_time = end-start
                self.times.append(execution_time)

                # breakpoint()
                
                byte_len = len(buffer.getvalue())
                bytes_per_sec = byte_len/execution_time
                self.gb_s.append(bytes_per_sec/1000000000)
                yield buffer.read()

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
            typer.Option("--addr", "-a", help="Destination address (URL format)."),
        ],
        access_mode: Annotated[
            AccessMode,
            typer.Option("--access_mode", "-c", help="Data access mode"),
        ],
        img_per_file: Annotated[
            int,
            typer.Option("--img_per_file", "-n", help="Number of images per file"),
        ] = 10,
    ):
    ps = PsanaImgSrc(experiment, run, access_mode, detector)
    send_opts : dict[str,int] = {
        #"send_buffer_size": 32 # send blocks if 32 messages queue up
    }
    try:
        #with Push0(dial=addr, block=True, **send_opts) as push:
        # with Push0(**send_opts) as push:
        #     push.dial(addr, block=True)
        #     print(f"Connected to {addr} - starting stream.")

        file_writer = Hdf5FileWriter(img_per_file)
        i = 0
        for msg in file_writer( ps(mode) ):
            # push.send(msg)
            i+=1
            if rank==0:
                print(i)
            if i==50:
                break
            else:
                pass
        with open('/sdf/home/m/mavaylon/lclstream_test/output_'+str(rank)+'.json', 'w') as f:
            json.dump(file_writer.gb_s, f)
            # json.dump(file_writer.times, f)
    except ConnectionRefused as e:
        print(f"Unable to connect to {addr} - {str(e)}.")
        return 1

    return 0

def run():
    typer.run(psana_push)

if __name__ == "__main__":
    run()
