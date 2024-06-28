#!/usr/bin/env python
# -*- coding: utf-8 -*-

# server_async.py
# uvicorn --workers 4 --host localhost --port 5001 server_async:app

import os
import io
import asyncio
import signal

from .psana_img_src import PsanaImgSrc

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Response
from concurrent.futures import ProcessPoolExecutor

# ___/ ASYNC CONFIG \___
app = FastAPI()

# Initialize the executor with a specific number of workers
executor = ProcessPoolExecutor(max_workers=4)

# Cleanup function to ensure executor shutdown
def cleanup():
    executor.shutdown(wait=True)
    print("Executor has been shut down gracefully")

# Register cleanup with FastAPI shutdown event
@app.on_event("shutdown")
def shutdown_event():
    cleanup()

# Additional signal handling for manual interruption
def handle_exit(sig, frame):
    cleanup()
    asyncio.get_event_loop().stop()
    print("Signal received, shutting down.")

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

import pynng
import zfpy
def serialize(data) -> bytes:
    return zfpy.compress_numpy(orig_array, write_header=True)
    # inverse = zfpy.decompress_numpy(buf)

def send_experiment(exp, run, access_mode, detector_name, mode, addr) -> None:
    ps = PsanaImgSrc(exp, run, access_mode, detector_name)
    send_opts = {
       "send_buffer_size": 32 # send blocks if 32 messages queue up
    }

    start = time.time()
    n = 0
    nbyte = 0
    with Push0(dial=addr, **send_opts) as push:
        for img in ps:
            buf = serialize(img)
            n += 1
            #nbyte += img.nbytes
            nbyte += len(buf)
            push.send(buf)
    t = time.time() - start
    print(f"Sent {n} messages in {t} seconds: {nbyte/t/1024**2} MB/sec.")

async def send_experiment_async(exp, run, access_mode, detector_name, mode, addr) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        executor,
        send_experiment,
        exp, run, access_mode, detector_name, mode, addr
    )

class DataRequest(BaseModel):
    exp          : str
    run          : int
    access_mode  : str
    detector_name: str
    mode         : str = 'calib'
    addr         : str

@app.get('/')
async def list_experiments() -> List[str]:
    return []

@app.post('/fetch-data')
async def fetch_data(request: DataRequest) -> str:
    exp           = request.exp
    run           = request.run
    access_mode   = request.access_mode
    detector_name = request.detector_name
    mode          = request.mode
    addr          = request.addr

    await send_experiment_async(
        exp, run, access_mode, detector_name, mode, addr
    )

    return "ok"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
