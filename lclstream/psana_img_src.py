from collections.abc import Iterable
from typing import Union

import numpy as np

from psana import DataSource, MPIDataSource, Detector
from models import AccessMode, ImageRetrievalMode

EventImage = np.ndarray

class PsanaImgSrc:
    """
    It serves as an image accessing layer based on the data
    management system psana in LCLS (idx access mode)
    """

    def __init__(self, exp, run, access_mode : AccessMode, detector_name) -> None:
        # Boilerplate code to access an image
        # Set up data source
        self.access_mode = access_mode
        self.datasource_id = f"exp={exp}:run={run}:{access_mode.value}"
        if self.access_mode == AccessMode.idx:
            self.datasource    = DataSource(self.datasource_id )
            self.run_current   = next(self.datasource.runs())
            self.events        = self.run_current.times()
        else:
            self.datasource    = MPIDataSource(self.datasource_id )
            self.events        = self.datasource.events()
            
        # Set up detector
        self.detector = Detector(detector_name)

        # Set image reading mode
        self.read = { "raw"   : self.detector.raw,
                      "calib" : self.detector.calib,
                      "image" : self.detector.image,
                      "mask"  : self.detector.mask, }

    def __len__(self) -> int:
        return len(self.events)

    def __call__(self, mode : ImageRetrievalMode, id_panel = None) -> Iterable[EventImage]:
        # Only these modes are supported...
        assert mode in (ImageRetrievalMode.raw,
                        ImageRetrievalMode.calib,
                        ImageRetrievalMode.image), \
                f"Mode {mode.value} is not allowed!!!  Only 'raw', 'calib' and 'image' are supported."

        for event in self.events:
            if self.access_mode == AccessMode.idx:
                event_data = self.run_current.event(event)
            else:
                event_data = event
            data = self.read[mode](event_data)
            # assembling a multi-panel image:
            #img = self.read[mode](event, multipanel)
            yield data[int(id_panel)] if id_panel is not None else data

    def create_bad_pixel_mask(self):
        return self.read["mask"](self.run_current, calib       = True,
                                                   status      = True,
                                                   edges       = True,
                                                   central     = True,
                                                   unbond      = True,
                                                   unbondnbrs  = True,
                                                   unbondnbrs8 = False).astype(np.uint16)

