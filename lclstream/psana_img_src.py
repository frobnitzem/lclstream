from collections.abc import Iterable
from typing import Union, Optional, Tuple

import numpy as np

from .psana_stub import DataSource, MPIDataSource, Detector
from .models import AccessMode, ImageRetrievalMode

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
        self._runs : Optional[list] = None
        # list of psana.Run
        self.run_times : dict[int,list] = {}
        # list of psana.EventTime

        if self.access_mode == AccessMode.idx:
            self.datasource = DataSource(self.datasource_id )
        else:
            self.datasource = MPIDataSource(self.datasource_id )

        # Set up detector
        #self.detector = Detector(detector_name)
        self.detector = Detector(detector_name,
                                 self.datasource.env())

        # Set image reading mode
        self.read = { ImageRetrievalMode.raw   : self.detector.raw,
                      ImageRetrievalMode.calib : self.detector.calib,
                      ImageRetrievalMode.image : self.detector.image,
                      ImageRetrievalMode.mask  : self.detector.mask, }

    def __call__(self,
                 mode : ImageRetrievalMode,
                 id_panel : Optional[int] = None) -> Iterable[EventImage]:
        # Only these modes are supported...
        assert mode in (ImageRetrievalMode.raw,
                        ImageRetrievalMode.calib,
                        ImageRetrievalMode.image), \
                f"Mode {mode.value} is not allowed!!!  Only 'raw', 'calib' and 'image' are supported."

        read = self.read[mode]
        if id_panel is not None:
            read = lambda evt: self.read[mode](evt)[id_panel]

        # MPIDataSource (but not DataSource) provides small_data
        #smldata = self.datasource.small_data('my.h5')
        if self.access_mode == AccessMode.idx:
            #g = self.runs[0].events()
            g = map(self.event, range(len(self)))
        else:
            g = self.datasource.events()

        for evt in g:
            # assembling a multi-panel image:
            #img = read(evt, multipanel)
            data = read(evt)
            #smldata.append( cspad_mean = f(data) )
            yield data

    @property
    def runs(self) -> list:
        assert self.access_mode == AccessMode.idx
        if self._runs is None:
            self._runs = list(self.datasource.runs())
        return self._runs

    def event(self, idx : Union[int,Tuple[int,int]]):
        # TODO: allow slicing when, e.g. idx = slice(start,stop)
        if isinstance(idx, tuple):
            r, i = idx
        else:
            r = 0
            i = idx
        if r not in self.run_times:
            self.run_times[r] = self.runs[r].times()

        t = self.run_times[r][i]
        return self.runs[r].event(t)
    
    def __len__(self):
        assert self.access_mode == AccessMode.idx
        if 0 not in self.run_times:
            self.run_times[0] = self.runs[0].times()
        return len(self.run_times[0])

    def create_bad_pixel_mask(self):
        return self.read["mask"](self.run_current, calib       = True,
                                                   status      = True,
                                                   edges       = True,
                                                   central     = True,
                                                   unbond      = True,
                                                   unbondnbrs  = True,
                                                   unbondnbrs8 = False).astype(np.uint16)

