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

    poetry run uvicorn pstream.server:app --reload

or (mimicking deployment usage),

    poetry run gunicorn --config gunicorn_config.py pstream.server:app
