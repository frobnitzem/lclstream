import numpy as np
import psana

class PsanaImgSrc:
    """
    It serves as an image accessing layer based on the data
    management system psana in LCLS.  
    """

    def __init__(self, exp, run, mode, detector_name):
        # Boilerplate code to access an image
        # Set up data source
        self.datasource_id = f"exp={exp}:run={run}:{mode}"
        self.datasource    = psana.DataSource( self.datasource_id )
        self.run_current   = next(self.datasource.runs())
        self.timestamps    = self.run_current.times()

        # Set up detector
        self.detector = psana.Detector(detector_name)

        # Set image reading mode
        self.read = { "raw"   : self.detector.raw,
                      "calib" : self.detector.calib,
                      "image" : self.detector.image,
                      "mask"  : self.detector.mask, }

    def __len__(self):
        return len(self.timestamps)

    def __iter__(self, addr, mode, id_panel = None):
        # Only two modes are supported...
        assert mode in ("raw", "calib", "image"), \
                f"Mode {mode} is not allowed!!!  Only 'raw', 'calib' and 'image' are supported."
        
        for timestamp in self.timestamps:
            event = self.run_current.event(timestamp)
            data = self.read[mode](event)
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


