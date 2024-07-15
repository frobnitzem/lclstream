# LCLStream

Image streaming application for psana.

This is a FastAPI server that pushes datasets
on command.  The steps run as follows,

1. Send a REST-API request to initiate a dataset push.
   - dataset
   - experiment number
   - receiving server

2. A separate process spawns to send all images
   to the requested server.


# Development

Manually run the server code with:

    poetry run uvicorn lclstream.server:app --reload

or (mimicking deployment usage),

    poetry run gunicorn --config gunicorn_config.py lclstream.server:app


# Deployment Instructions

Install this package and its dependencies
using `pip install .` (for deployment)
or `poetry install` (for development).

If you do not yet have a server certificate,
create a server keypair using instructions from
[certified\_apis](https://code.ornl.gov/99R/certified_apis).

Run the server with the `uvicorn` launch command
above, but specifying the key and certificate files
as explained there.
