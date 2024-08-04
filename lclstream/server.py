#!/usr/bin/env python
# -*- coding: utf-8 -*-

# run as:
#
#     uvicorn --host localhost --port 5001 lclstream.server:app
#

from typing import Dict, List
import asyncio
import signal

from fastapi import FastAPI, HTTPException

from .transfer import Transfer
from .models import DataRequest

# Cleanup function
def cleanup():
    pass

# Additional signal handling for manual interruption
def handle_exit(sig, frame):
    cleanup()

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

app = FastAPI()

# Register cleanup with FastAPI shutdown event
@app.on_event("shutdown")
def shutdown_event():
    cleanup()

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
    global transfer_id
    trs = Transfer(request)
    ok = trs.start() # TODO catch some errors immediately
    #if not ok:

    n = transfer_id
    transfer_id += 1
    # stash in the list
    transfers[n] = trs
    return n

