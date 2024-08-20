#!/usr/bin/env python3

import time
from typing import Annotated, Iterable, Optional
from asyncio import run as aiorun

from pynng import Pull0, Timeout # type: ignore[import-untyped]

import typer

async def puller(listen : Optional[str] = None,
                 dial : Optional[str] = None,
                 verb : bool = False) -> Iterable[bytes]:
    """ An async iterator over data pulled from addr.

    Args:
        listen: URL to listen at
        dial: URL to dial
        verb: print status updates
    """
    done = 0
    started = 0
    t0 = time.time()
    def show_open(pipe):
        nonlocal started
        if started == 0:
            t0 = time.time()
        if verb:
            print("Pull: pipe opened")
        started += 1

    def show_close(pipe):
        nonlocal done
        if verb:
            print("Pull: pipe closed")
        done += 1

    nmsg = 0
    nbytes = 0
    with Pull0(listen=listen, dial=dial,
               recv_timeout=5000) as pull:
        pull.add_post_pipe_connect_cb(show_open)
        pull.add_post_pipe_remove_cb(show_close)
        if verb:
            print("Pull: waiting for connection.")

        while started == 0 or (done != started):
            try:
                b = await pull.arecv()
                nmsg += 1
                nbytes += len(b)
                yield b
            except Timeout:
                if started and verb:
                    assert t0 is not None
                    elapsed = time.time() - t0
                    mbyte = nbytes/1024**2
                    print(f"Pull: {nmsg} messages, {mbyte} MB @ {mbyte/elapsed} Mbps")
                continue

        if verb:
            elapsed = time.time() - t0
            mbyte = nbytes/1024**2
            print(f"Pull complete: {nmsg} messages, {mbyte} MB @ {mbyte/elapsed} Mbps")

def psana_pull(
        listen: Annotated[
            Optional[str],
            typer.Option("--listen", "-l", help="Address to listen at (URL format)."),
        ] = None,
        dial: Annotated[
            Optional[str],
            typer.Option("--dial", "-d", help="Address to dial (URL format)."),
        ] = None,
    ):
    async def main():
        nmsg = 0
        nbyte = 0
        t0 = None
        async for msg in puller(listen, dial, True):
            nmsg += 1
            nbyte += len(msg)
            if t0 is None:
                t0 = time.time()
        if t0 is None:
            print("No messages received.")
            return
        dt = time.time() - t0
        print(f"Received {nmsg} messages in {dt} seconds @ {8*nbyte/dt/1024**2} Mbit/sec.")

    assert (dial is not None) or (listen is not None), "Need an address."
    aiorun( main() )

def run():
    typer.run(psana_pull)

if __name__ == "__main__":
    run()
