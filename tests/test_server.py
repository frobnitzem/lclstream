from typing import List
import pytest
import os
os.environ["RAND_PSANA"] = "1"

from aiowire import EventLoop, Call
from actor_api import ActorTest

from lclstream.server import app
from lclstream.transfer import Transfer
from lclstream.models import DataRequest

from pynng import Pull0, Timeout

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

@pytest.mark.asyncio
async def test_get_list():
    atest = ActorTest( { "lclstream": app } )

    async def msg(*args):
        return await atest.message("lclstream", *args)

    async def cb(eve):
        response = await msg("list_transfers")
        assert isinstance(response, list)

    async with EventLoop(1.0) as eve:
        eve.start( atest )
        eve.start( cb )

@pytest.mark.asyncio
async def test_mk_transfer():
    atest = ActorTest( { "lclstream": app } )
    async def msg(*args):
        return await atest.message("lclstream", *args)

    addr = "inproc://pull_test"
    async def cb(eve):
        with pytest.raises(ValueError):
            response = await msg("new_transfer", "abc", 42)

        trs = DataRequest(exp = "grail",
                          run = 42,
                          access_mode = "swift",
                          detector_name = "excalibur",
                          mode = "image",
                          addr = addr)
        tid = await msg("new_transfer", trs)
        assert isinstance(tid, int)

        state = await msg("get_transfer", tid)
        assert isinstance(state, str)
        print(f"Transfer state = {state}")
    
        ok = await msg("del_transfer", tid)
        assert isinstance(ok, bool)
        print(f"Delete transfer result = {ok}")

    async with EventLoop(10.0) as eve:
        eve.start( atest )
        eve.start( Call(puller, addr) )
        eve.start( cb )
