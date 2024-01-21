#!/usr/bin/env python
# -*- coding: utf-8 -*-

## import requests
## import msgpack
## import numpy as np
## import ray
## 
## ray.init(num_cpus = 4)  # Initialize Ray
## 
## @ray.remote
## def fetch_event(exp, run, access_mode, detector_name, event):
##     url = 'http://127.0.0.1:5000/fetch-data'
##     payload = {
##         'exp': exp,
##         'run': run,
##         'access_mode': access_mode,
##         'detector_name': detector_name,
##         'event': event
##     }
##     response = requests.post(url, json=payload)
##     if response.status_code == 200:
##         response_dict = msgpack.unpackb(response.content, raw=False)
##         data_list = response_dict['data']
##         pid = response_dict['pid']
##         data_array = np.array(data_list)
##         return data_array, pid
##     else:
##         print(f"Failed to fetch data for event {event}: {response.status_code}")
##         return None, None
## 
## if __name__ == "__main__":
##     exp = 'xpptut15'
##     run = 630
##     access_mode = "idx"
##     detector_name = "jungfrau1M"
##     events = range(0, 500)
## 
##     # Submit tasks to Ray
##     futures = [fetch_event.remote(exp, run, access_mode, detector_name, event) for event in events]
## 
##     # Retrieve results as they complete
##     for future in ray.get(futures):
##         data_array, pid = future
##         if data_array is not None:
##             print(f"PID ({pid:02d}), Data for event: {data_array.shape}", flush = True)



import requests
import msgpack
import numpy as np
import ray

ray.init()

@ray.remote
def fetch_event(exp, run, access_mode, detector_name, event):
    ## url = 'http://127.0.0.1:5000/fetch-data'
    url = 'http://172.24.49.14:5000/fetch-data'
    payload = {
        'exp': exp,
        'run': run,
        'access_mode': access_mode,
        'detector_name': detector_name,
        'event': event
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        response_dict = msgpack.unpackb(response.content, raw=False)
        data_list = response_dict['data']
        pid = response_dict['pid']
        data_array = np.array(data_list)
        return data_array, pid, event
    else:
        print(f"Failed to fetch data for event {event}: {response.status_code}")
        return None, None

if __name__ == "__main__":
    exp = 'xpptut15'
    run = 630
    access_mode = "idx"
    detector_name = "jungfrau1M"
    events = range(0, 1000)

    # Submit tasks to Ray
    futures = [fetch_event.remote(exp, run, access_mode, detector_name, event) for event in events]

    # Use Ray's wait method to process tasks as they complete
    while len(futures) > 0:
        done_futures, futures = ray.wait(futures, num_returns=1)
        data_array, pid, event = ray.get(done_futures[0])
        if data_array is not None:
            print(f"PID ({pid:02d}), Event: {event:06d}, Data for event: {data_array.shape}")

