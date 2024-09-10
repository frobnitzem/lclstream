from collections.abc import Iterator
import time
import logging
_logger = logging.getLogger(__name__)

import stream
from pynng import Push0, Pull0, Timeout, ConnectionRefused # type: ignore[import-untyped]

clock0 = lambda: {'count': 0, 'size': 0, 'wait': 0, 'time': time.time()}
def rate_clock(state, sz):
    t = time.time()
    return {
        'count': state['count'] + 1,
        'size': state['size'] + sz,
        'wait': state['wait'] + t - state['time'],
        'time': t
    }

send_opts : dict[str,int] = {
     #"send_buffer_size": 32 # send blocks if 32 messages queue up
}
recv_options = {"recv_timeout": 5000}

@stream.stream
def pusher(gen : Iterator[bytes], addr : str, ndial : int
          ) -> Iterator[int]:
    # transform messages sent into sizes sent
    assert ndial >= 0
    options = dict(send_opts)
    if ndial == 0:
        options["listen"] = addr
    try:
        with Push0(**options) as push:
            for dial in range(ndial):
                push.dial(addr, block=True)
            if ndial > 0:
                _logger.info("Connected to %s x %d - starting stream.",
                             addr, ndial)
            else:
                _logger.info("Listening on %s.", addr)

            for msg in gen:
                push.send(msg)
                yield len(msg)
    except ConnectionRefused as e:
        _logger.error("Unable to connect to %s - %s", addr, e)

@stream.source
def puller(addr : str, ndial : int) -> Iterator[bytes]:
    assert ndial >= 0

    done = 0
    started = 0
    def show_open(pipe):
        nonlocal started
        _logger.info("Pull: pipe opened")
        started += 1
    def show_close(pipe):
        nonlocal done
        _logger.info("Pull: pipe closed")
        done += 1

    options = dict(recv_options)
    if ndial == 0:
        options["listen"] = addr
    try:
        with Pull0(**options) as pull:
            pull.add_post_pipe_connect_cb(show_open)
            pull.add_post_pipe_remove_cb(show_close)
            for dial in range(ndial):
                pull.dial(addr, block=True)
            if ndial == 0:
                _logger.info("Pull: waiting for connection")
            else:
                _logger.info("Connected to %s x %d - starting recv.",
                              addr, ndial)

            while started == 0 or (done != started):
                try:
                    msg = pull.recv()
                    yield msg
                except Timeout:
                    if started:
                        _logger.debug("Pull: slow input")

    except ConnectionRefused as e:
        _logger.error("Unable to connect to %s - %s", addr, e)

@stream.source
def file_chunks(fname, chunksz=1024*1024) -> Iterator[bytes]:
    with open(fname, 'rb') as f:
        while True:
            data = f.read(chunksz)
            if len(data) == 0:
                break
            yield data

@stream.stream
def file_writer(gen : Iterator[bytes],
                fname : str,
                append : bool = False
               ) -> Iterator[int]:
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    with open(fname, mode) as f:
        for data in gen:
            sz = f.write(data)
            yield sz

