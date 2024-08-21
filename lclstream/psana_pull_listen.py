import sys

from pynng import Pull0 # type: ignore[import-untyped]
from pynng.exceptions import ConnectionRefused # type: ignore[import-untyped]


addr = sys.argv[1]

try:
    #with Push0(dial=addr, block=True, **send_opts) as push:
    with Pull0() as pull:
        pull.listen(addr)
        print(f"Connected to {addr} - starting stream.")
        while True:
           msg = pull.recv()
           print("Message received")
except ConnectionRefused as e:
    print(f"Unable to connect to {addr} - {str(e)}.")


