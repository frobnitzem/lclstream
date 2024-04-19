# LCLStream

Image streaming application for psana.

This is a simple FastAPI server that reads psana1 data from
a local directory and serves individual images.


# Development

Manually run the server code with:

    poetry run uvicorn pstream.server:app --reload

or (mimicking deployment usage),

    poetry run gunicorn --config gunicorn_config.py pstream.server:app
