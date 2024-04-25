#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hdf5plugin
import time
import requests
import io
import h5py
import msgpack
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class RemotePsanaDataset(Dataset):
    def __init__(self, url, requests_list):
        """ 
        requests_list: A list of tuples. Each tuple should contain:
                       (exp, run, access_mode, detector_name, event)
        """ 
        self.url           = url 
        self.requests_list = requests_list

    def __len__(self):
        return len(self.requests_list)

    def __getitem__(self, idx):
        exp, run, access_mode, detector_name, event = self.requests_list[idx]
        return self.fetch_event(exp, run, access_mode, detector_name, event)

    def fetch_event(self, exp, run, access_mode, detector_name, event):
        ## url = 'http://172.24.49.14:5000/fetch-data'
        url = self.url
        payload = {
            'exp'          : exp,
            'run'          : run,
            'access_mode'  : access_mode,
            'detector_name': detector_name,
            'event'        : event
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            with io.BytesIO(response.content) as hdf5_bytes:
                with h5py.File(hdf5_bytes, 'r') as hdf5_file:
                    data_array = np.array(hdf5_file['data'])
                    pid = hdf5_file['pid'][0]  # Assuming pid is stored as a dataset of size 1

            return data_array, pid, event
        else:
            print(f"Failed to fetch data for event {event}: {response.status_code}")
            return None, None, None

# Usage example
## url = 'http://172.24.49.14:5000/fetch-hdf5'
## url = 'http://172.24.49.14:5001/fetch-hdf5'
url = 'http://172.24.48.143:5001/fetch-hdf5'
## url = 'http://localhost:5000/fetch-hdf5'
requests_list = [ ('xpptut15'   , 630, 'idx', 'jungfrau1M', event) for event in range(1000) ] +\
                [ ('mfxp1002121',   7, 'idx',    'Rayonix', event) for event in range(1000) ]

dataset = RemotePsanaDataset(url = url, requests_list = requests_list)

dataloader = DataLoader(dataset, batch_size=20, num_workers=10, prefetch_factor = None)
dataloader_iter = iter(dataloader)
batch_idx       = 0
while True:
    try:
        t_s = time.monotonic()
        batch_data, batch_pid, batch_event = next(dataloader_iter)
        t_e = time.monotonic()
        loading_time_in_sec = (t_e - t_s)

        ## print(f"Batch idx: {batch_idx:d} (PID: {batch_pid.tolist()}), Total time: {loading_time_in_sec:.2f} s, Average time: {loading_time_in_sec / len(batch_data) * 1e3:.2f} ms/event, Batch shape: {batch_data.shape}")
        print(f"Batch idx: {batch_idx:d}, Total time: {loading_time_in_sec:.2f} s, Average time: {loading_time_in_sec / len(batch_data) * 1e3:.2f} ms/event, Batch shape: {batch_data.shape}")

        batch_idx += 1
    except StopIteration:
        break
