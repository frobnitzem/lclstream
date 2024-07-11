# LCLStream

Image streaming application for psana.

This is an actor-api server that pushes datasets
on command.  The steps run as follows,


1. Send a message to initiate a dataset push.
   - dataset
   - experiment number
   - receiving server

2. A separate process spawns to send all images
   to the requested server.


# Deployment Instructions

Install this package and its dependencies
using `pip install .` (for deployment)
or `poetry install` (for development).

If you do not yet have `$HOME/.config/actors.json`,
make one with `actors init 'user-id@domain'`.

Create server config (and keypair) using

    actors new 'lclstream@domain' tcp://1.2.3.4:5555 >lclstream.json

This server config file specifies exactly
who is allowed to interact with the service
by their public key.  Your identity from
`$HOME/.config/actors.json` is automatically
filled in.

Run the server (which will start a ZeroMQ listener
on the address-url above) with,

    actors run -v --config lclstream.json lclstream.server:app &

Connect to the server and run methods using `message`:

    message 'lclstream@domain' new_transfer '{"exp": "grail", "run": 42, "access_mode": "swift", "detector_name": "excalibur", "mode": "image", "addr": "tcp://1.2.3.5:6500"}

The lclstream service is known by name because it was
added by the `actors new` command.
