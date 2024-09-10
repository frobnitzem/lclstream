from collections.abc import Iterator
import time
import logging
_logger = logging.getLogger(__name__)

import stream
from pynng import Push0, Pull0

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
def file_chunks(fname, chunksz=1024*1024) -> Iterator[bytes]:
    with open(fname, 'rb') as f:
        while True:
            data = f.read(chunksz)
            if len(data) == 0:
                break
            yield data



