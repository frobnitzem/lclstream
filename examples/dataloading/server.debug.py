#!/usr/bin/env python
# -*- coding: utf-8 -*-

# server.py
# gunicorn -w 10 -b 172.24.49.14:5000 server:app

from flask import Flask, request, Response
import os
import msgpack
import io
import h5py
import hdf5plugin
import numpy as np
from maxie.datasets.psana_utils import PsanaImg

import time

app = Flask(__name__)

# Buffer for each process (if using multiple processes with something like Gunicorn)
psana_img_buffer = {}

# Get the current process ID
pid = os.getpid()

def get_psana_img(exp, run, access_mode, detector_name):
    key = (exp, run)
    if key not in psana_img_buffer:
        psana_img_buffer[key] = PsanaImg(exp, run, access_mode, detector_name)
    return psana_img_buffer[key]

@app.route('/fetch-data', methods=['POST'])
def fetch_data():
    exp = request.json.get('exp')
    run = request.json.get('run')
    access_mode = request.json.get('access_mode')
    detector_name = request.json.get('detector_name')
    event = request.json.get('event')
    mode = request.json.get('mode', 'calib')

    psana_img = get_psana_img(exp, run, access_mode, detector_name)
    data = psana_img.get(event, None, mode)

    # Serialize data using msgpack
    response_dict = {
        'data': data.tolist(),  # Convert NumPy array to list
        'pid': pid
    }
    response_data = msgpack.packb(response_dict, use_bin_type=True)
    return Response(response_data, mimetype='application/octet-stream')


@app.route('/fetch-hdf5', methods=['POST'])
def fetch_hdf5():
    exp = request.json.get('exp')
    run = request.json.get('run')
    access_mode = request.json.get('access_mode')
    detector_name = request.json.get('detector_name')
    event = request.json.get('event')
    mode = request.json.get('mode', 'calib')

    psana_img = get_psana_img(exp, run, access_mode, detector_name)

    t_s  = time.monotonic()
    data = psana_img.get(event, None, mode)
    t_e  = time.monotonic()
    t_get_image = t_e - t_s

    # Serialize data using BytesIO and hdf5...
    t_s  = time.monotonic()
    with io.BytesIO() as hdf5_bytes:
        with h5py.File(hdf5_bytes, 'w') as hdf5_file_handle:
            hdf5_file_handle.create_dataset('data', data = data, **hdf5plugin.Bitshuffle(nelems=0, lz4=True),)
            hdf5_file_handle.create_dataset('pid' , data = np.array([pid]))

        response_data = hdf5_bytes.getvalue()
    t_e  = time.monotonic()
    t_pack_image = t_e - t_s

    print(f"Processed exp={exp}, run={run:04d}, event={event:06d}; psana={t_get_image} sec, packing={t_pack_image} sec.")

    return Response(response_data, mimetype='application/octet-stream')

if __name__ == "__main__":
    app.run(debug=True, port=5000)

