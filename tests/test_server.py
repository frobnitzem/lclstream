from typing import List
import os
os.environ["RAND_PSANA"] = "1"

from fastapi.testclient import TestClient

from lclstream.server import app
from lclstream.transfer import Transfer
from lclstream.models import DataRequest

# docs: python-httpx.org/advanced/
client = TestClient(app)

def test_get_list():
    response = client.get("/")
    assert response.status_code == 200
    resp = response.json()
    assert isinstance(resp, list)

def test_mk_transfer():
    response = client.post("/transfers/new",
                           json='{"exp": "abc", "run" 42}')
    assert response.status_code == 404 or response.status_code == 422

    trs = DataRequest(exp = "grail",
                      run = 42,
                      access_mode = "swift",
                      detector_name = "excalibur",
                      mode = "image",
                      addr = "tcp://127.0.0.1:41739")
    response = client.post("/transfers/new",
                           json=trs.model_dump())
    assert response.status_code == 200
    tid = response.json()
    assert isinstance(tid, int)

    response2 = client.get(f"/transfers/{tid}")
    state = response2.json()
    assert isinstance(state, str)
    print(f"Transfer state = {state}")
    
    response3 = client.post(f"/transfers/delete/{tid}")
    ok = response3.json()
    assert isinstance(ok, bool)
    print(f"Delete transfer result = {ok}")
