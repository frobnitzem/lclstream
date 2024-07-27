import asyncio

from typing import List
import pytest
import os
os.environ["RAND_PSANA"] = "1"

import pytest

from fastapi.testclient import TestClient

from lclstream.server import app
from lclstream.transfer import Transfer
from lclstream.models import DataRequest, ImageRetrievalMode, AccessMode

from pynng import Pull0, Timeout

ADDR = "tcp://127.0.0.1:28451"

client = TestClient(app)

async def puller(addr):
    done = False

    def show_open(pipe):
        print("pullz: pipe opened")

    def show_close(pipe):
        nonlocal done
        print("pullz: pipe closed")
        done = True

    n = 0
    with Pull0(listen=addr, recv_timeout=100) as pull:
        while not done:
            try:
                b = await pull.arecv()
                n += 1
            except Timeout:
                if not done:
                    print("pullz: waiting for data")
                continue
    print(f"pullz: received {n} messages")

@pytest.fixture()
def pull_server(event_loop):
    task = asyncio.ensure_future(puller(ADDR), loop=event_loop)

    # Sleeps to allow the server boot-up.
    event_loop.run_until_complete(asyncio.sleep(0.1))

    try:
        yield
    finally:
        task.cancel()

def test_get_list():
    response = client.get("/transfers")
    assert response.status_code == 200
    resp = response.json()
    assert isinstance(resp, list)

@pytest.mark.asyncio
async def test_mk_transfer(pull_server):
    response = client.post("/transfers/new", json={"abc": 2})
    assert response.status_code == 422

    trs = DataRequest(exp = "grail",
                      run = 42,
                      access_mode = AccessMode.idx,
                      detector_name = "excalibur",
                      mode = ImageRetrievalMode.image,
                      addr = ADDR)
    response = client.post("/transfers/new", json=trs.model_dump())
    assert response.status_code == 200
    tid = response.json()
    assert isinstance(tid, int)

    response = client.get(f"/transfers/{tid}")
    assert response.status_code == 200
    state = response.json()
    assert isinstance(state, str)
    print(f"Transfer state = {state}")
    
    response = client.post(f"/transfers/delete/{tid}")
    assert response.status_code == 200
    ok = response.json()
    assert isinstance(ok, bool)
    print(f"Delete transfer result = {ok}")
