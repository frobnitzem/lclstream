#!/usr/bin/env python
# -*- coding: utf-8 -*-

# server_async.py
# uvicorn --workers 10 --host localhost --port 5001 server_async:app

import msgpack
import os
import io
import h5py
import hdf5plugin
import numpy as np
import asyncio

from maxie.datasets.psana_utils import PsanaImg

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Response
from concurrent.futures import ProcessPoolExecutor

app = FastAPI()

executor = ProcessPoolExecutor(max_workers = 10)

# Buffer for each process (if using multiple processes with something like Gunicorn)
psana_img_buffer = {}

# Get the current process ID
pid = os.getpid()

def get_psana_img(exp: str, run: int, access_mode: str, detector_name: str):
    key = (exp, run)
    if key not in psana_img_buffer:
        psana_img_buffer[key] = PsanaImg(exp, run, access_mode, detector_name)
    return psana_img_buffer[key]

def get_psana_data(exp, run, access_mode, detector_name, event, mode):
    psana_img = get_psana_img(exp, run, access_mode, detector_name)
    data = psana_img.get(event, None, mode)
    return data

class DataRequest(BaseModel):
    exp: str
    run: int
    access_mode: str
    detector_name: str
    event: int
    mode: str = 'calib'

@app.post('/fetch-data')
async def fetch_data(request: DataRequest):
    exp = request.exp
    run = request.run
    access_mode = request.access_mode
    detector_name = request.detector_name
    event = request.event
    mode = request.mode

    ## psana_img = get_psana_img(exp, run, access_mode, detector_name)
    ## data = await psana_img.get(event, None, mode)
    data = await asyncio.get_event_loop().run_in_executor(
        executor,
        get_psana_data,
        exp, run, access_mode, detector_name, event, mode
    )

    response_dict = {
        'data': data.tolist(),  # Convert NumPy array to list
        'pid': pid
    }
    response_data = msgpack.packb(response_dict, use_bin_type=True)
    return Response(content=response_data, media_type='application/octet-stream')

@app.post('/fetch-hdf5')
async def fetch_hdf5(request: DataRequest):
    exp = request.exp
    run = request.run
    access_mode = request.access_mode
    detector_name = request.detector_name
    event = request.event
    mode = request.mode

    ## psana_img = get_psana_img(exp, run, access_mode, detector_name)
    ## data = await psana_img.get(event, None, mode)
    data = await asyncio.get_event_loop().run_in_executor(
        executor,
        get_psana_data,
        exp, run, access_mode, detector_name, event, mode
    )

    with io.BytesIO() as hdf5_bytes:
        with h5py.File(hdf5_bytes, 'w') as hdf5_file_handle:
            hdf5_file_handle.create_dataset('data', data=data, **hdf5plugin.Bitshuffle(nelems=0, lz4=True),)
            hdf5_file_handle.create_dataset('pid', data=np.array([pid]))
        response_data = hdf5_bytes.getvalue()

    return Response(content=response_data, media_type='application/octet-stream')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
