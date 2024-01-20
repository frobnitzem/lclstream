#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import time
import yaml
import msgpack
import ray
import signal
import argparse

from maxie.datasets.psana_dataset import PsanaDataset
from maxie.datasets.psana_utils   import PsanaImg
from maxie.datasets.utils         import split_list_into_chunk

# Define a workflow to test data loading performance...
@ray.remote
class BenchmarkDataloader:
    def __init__(self, config, device, events):
        self.config = config
        self.device = device
        self.events = events

        self.dataloader = None

    def _initialize_dataloader(self):
        config = self.config
        device = self.device
        events = self.events

        # Psana specific...
        exp           = config['exp'          ]
        run           = config['run'          ]
        img_load_mode = config['img_load_mode']
        access_mode   = config['access_mode'  ]
        detector_name = config['detector_name']
        photon_energy = config['photon_energy']
        encoder_value = config['encoder_value']

        # Data loader specific...
        dataloader_batch_size  = config["dataloader_batch_size"]
        dataloader_num_workers = config["dataloader_num_workers"]

        # Establish dataloader...
        psana_dataset = PsanaDataset(exp, run, access_mode, detector_name, img_load_mode, events)
        dataloader = torch.utils.data.DataLoader( psana_dataset,
                                                  shuffle     = False,
                                                  pin_memory  = True,
                                                  batch_size  = dataloader_batch_size,
                                                  num_workers = dataloader_num_workers, )

        self.dataloader = dataloader

    def measure(self):
        if self.dataloader is None:
            self._initialize_dataloader()

        dataloader_iter = iter(self.dataloader)
        batch_idx       = 0
        t_s = time.monotonic()
        while True:
            try:
                batch_data, batch_metadata = next(dataloader_iter)
                batch_idx += 1
            except StopIteration:
                break
        t_e = time.monotonic()
        loading_time_in_sec = (t_e - t_s)

        return loading_time_in_sec, len(self.events)

# Shutdown ray clients during a Ctrl+C event...
def signal_handler(sig, frame):
    print('SIGINT (Ctrl+C) caught, shutting down Ray...')
    ray.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# [[[ ARG PARSE ]]]
parser = argparse.ArgumentParser(description='Process a yaml file.')
parser.add_argument('yaml', help='The input yaml file.')
args = parser.parse_args()

# [[[ Configure ]]]
fl_yaml = args.yaml
basename_yaml = fl_yaml[:fl_yaml.rfind('.yaml')]

# Load the YAML file
with open(fl_yaml, 'r') as fh:
    config = yaml.safe_load(fh)

# ___/ USER INPUT \___
# Psana...
exp                  = config['exp'          ]
run                  = config['run'          ]
img_load_mode        = config['img_load_mode']
access_mode          = config['access_mode'  ]
detector_name        = config['detector_name']
photon_energy        = config['photon_energy']
encoder_value        = config['encoder_value']

# Data range...
event_min            = config['event_min']
event_max            = config['event_max']

# Divide events...
ray_num_cpus = config["ray_num_cpus"]

# Output...
dir_results    = config["dir_results"]

# Properties derived from user inputs...
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Derive total number of events...
psana_img = PsanaImg(exp, run, access_mode, detector_name)
if event_min is None: event_min = 0
if event_max is None: event_max = len(psana_img.timestamps)

# Split all events into event groups...
SLURM_NNODES          = os.getenv('SLURM_NNODES')
SLURM_NTASKS_PER_NODE = os.getenv('SLURM_NTASKS_PER_NODE')
max_num_chunk = int(SLURM_NNODES) * int(SLURM_NTASKS_PER_NODE) if not None in (SLURM_NNODES, SLURM_NTASKS_PER_NODE) else ray_num_cpus
events = range(event_min, event_max)
event_groups = split_list_into_chunk(events, max_num_chunk = max_num_chunk)
print(f"Total events of this run: {len(events)}")
print(f"Max number of chunks: {max_num_chunk}")

# Init ray...
# Check if RAY_ADDRESS is set in the environment
USES_MULTI_NODES = os.getenv('USES_MULTI_NODES')
if USES_MULTI_NODES:
    ray.init(address = 'auto')
else:
    ray.init(num_cpus = ray_num_cpus)

# Create an actor for each batch
actors = [BenchmarkDataloader.remote(config, device, event_group) for event_group in event_groups]

# Call the measure method on each actor
futures = [actor.measure.remote() for actor in actors]

# Gather results
results = ray.get(futures)

# Display results
for loading_time, num_events in results:
    print(f"Total events: {num_events}, Total time: {loading_time:.2f} s, Average time: {loading_time / num_events * 1e3:.2f} ms/event")

# Shutdown ray...
ray.shutdown()
