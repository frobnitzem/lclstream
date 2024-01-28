#!/usr/bin/env python
# -*- coding: utf-8 -*-

# server.py
# gunicorn -w 10 -b 172.24.49.14:5000 server:app

from flask import Flask, request, Response
import os
import msgpack
from maxie.datasets.psana_utils import PsanaImg

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

if __name__ == "__main__":
    app.run(debug=True, port=5000)

