import requests
import msgpack
import numpy as np
from torch.utils.data import Dataset

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
            response_dict = msgpack.unpackb(response.content, raw=False)
            data_list     = response_dict['data']
            pid           = response_dict['pid']
            data_array    = np.array(data_list)

            return data_array, pid, event
        else:
            print(f"Failed to fetch data for event {event}: {response.status_code}")
            return None, None, None

# Usage example
requests_list = [
    ('xpptut15', 630, 'idx', 'jungfrau1M', event) for event in range(1000)
    # You can add more tuples for different experiments, runs, etc.
]

dataset = RemoteDataDataset(requests_list)

# DataLoader setup
from torch.utils.data import DataLoader

data_loader = DataLoader(dataset, batch_size=1, num_workers=10)

# Iterate over data
for data_array, pid, event in data_loader:
    if data_array is not None:
        print(f"PID ({pid[0]:02d}), Event: {event[0]:06d}, Data for event: {data_array.shape}")
