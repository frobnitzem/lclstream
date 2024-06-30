#!/usr/bin/env python
# -*- coding: utf-8 -*-

# run as:
#
#     ./server.py
#
# or
#
#     uvicorn --host localhost --port 5001 server:app
#

from typing import Dict, List
import asyncio
import signal

from .psana_img_src import PsanaImgSrc
from .transfer import Transfer
from .models import DataRequest

from fastapi import FastAPI, HTTPException #, Response
from concurrent.futures import ProcessPoolExecutor

# ___/ ASYNC CONFIG \___
app = FastAPI()

# Initialize the executor with no specific number of workers
executor = ProcessPoolExecutor() #max_workers=4)

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

@app.get('/')
async def list_experiments() -> List[str]:
    return []

transfer_id = 0
transfers : Dict[int, Transfer] = {}

@app.get('/transfers')
async def list_transfers() -> List[DataRequest]:
    return [trs.request for trs in transfers.values()]

@app.post('/transfers/delete/{n}')
async def cancel_transfer(n : int) -> bool:
    try:
        trs = transfers.pop(n)
    except KeyError:
        return False
    return trs.cancel()

@app.get('/transfers/{n}')
async def get_transfer(n : int) -> str:
    try:
        trs = transfers[n]
    except KeyError:
        return "not found"
    # TODO: periodically await and clear these transfers out
    return trs.state

@app.post('/transfers/new')
async def new_transfer(request: DataRequest) -> int:
    trs = Transfer(request)
    ok = trs.start() # TODO catch some errors immediately
    #if not ok:

    n = transfer_id
    transfer_id += 1
    # stash in the list
    transfers[n] = trs
    return n

if __name__ == "__main__":
    import uvicorn
    print("Access this server at http://127.0.0.1:5001/docs")
    uvicorn.run(app, host="127.0.0.1", port=5001)
