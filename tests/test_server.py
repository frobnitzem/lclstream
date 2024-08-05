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
from lclstream.psana_pull import puller

from pynng import Pull0, Timeout

ADDR = "tcp://127.0.0.1:28451"

client = TestClient(app)

@pytest.fixture()
def pull_server(event_loop):
    async def run_pull(pull):
        nmsg = 0
        async for data in pull:
            nmsg += 1
        print(f"pull_server: received {nmsg} messages")

    P = puller(ADDR)
    task = asyncio.ensure_future(run_pull(P), loop=event_loop)

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
