#!/usr/bin/env python3

from io import BytesIO
from typing import Annotated

from pynng import Push0 # type: ignore[import-untyped]
import h5py
import hdf5plugin
import typer

from lclstream.models import ImageRetrievalMode, AccessMode
from lclstream.psana_img_src import PsanaImgSrc


class Hdf5FileWriter:

    def __init__(self, num_img_in_file):
        self._num_img_in_file = num_img_in_file
        self._new_file_reqrd = True
        self._dataset = None

    def add_img_to_file(self, img):
        if self._new_file_reqrd:
            self._bytes_buffer = BytesIO()
            self._fh = h5py.File(self._bytes_buffer, 'w')
            self._dataset = self._fh.create_dataset(
                'data',
                shape=(self._num_img_in_file,) + img.shape,
                **hdf5plugin.Zfp()
            )
            self._new_file_reqrd = False
            self._idx_img_to_write = 0

        self._dataset[self._idx_img_to_write] = img
        self._idx_img_to_write += 1
        
        if self._idx_img_to_write == self._num_img_in_file:
            # Returns the bytes buffer if the file is full, otherwise None
            self._new_file_reqrd = True
            return self._bytes_buffer
        else:
            return None
         

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

):
    ps = PsanaImgSrc(experiment, run, access_mode, detector)

    #send_opts = {
    #    "send_buffer_size": 32 # send blocks if 32 messages queue up
    #}
    with Push0(dial=addr) as push:
        print(f"Connected to {addr} - starting stream.")
         
        num_img_per_file = 20 # Hardcoded for now

        file_writer = Hdf5FileWriter(num_img_per_file)

        for img in ps(mode):
            bytes_to_send = file_writer.add_img_to_file(img)
            if bytes_to_send:
                push.send(bytes_to_send.read())

def run():
    typer.run(psana_push)

if __name__ == "__main__":
    run()
