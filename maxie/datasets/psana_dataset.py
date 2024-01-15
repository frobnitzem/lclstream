import numpy as np
import logging

from .psana_utils import PsanaImg
from .utils       import apply_mask

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class PsanaDataset(Dataset):
    """
    The PsanaDataset class enables image batching directly from XTC files using
    the Psana interface, tailored for the Linac Coherent Light Source (LCLS).
    This class, as part of a PyTorch Dataset, implements lazy initialization of
    the Psana interface. This approach ensures seamless operation within
    Python's multiprocessing environment, crucial for parallel data processing
    tasks.

    Lazy initialization is a key feature, where the Psana interface and data
    are not fully initialized until required by a specific operation. This
    method is particularly effective when dealing with large datasets and is
    essential for compatibility with multiprocessing in Python, as it allows
    the dataset to be forked across multiple processes without encountering
    issues common with pre-initialized resources.

    Parameters:
        exp (str)                  : Experiment identifier.
        run (int)                  : Run number for the experiment.
        mode (str)                 : Operational mode for data access and processing.
        detector_name (str)        : Name of the detector used in the experiment.
        img_mode (str)             : Image data processing mode (e.g., raw, calibrated).
        event_list (list, optional): Specific events to be included in the dataset.
                                     Defaults to None, which includes all events.

    In addition to handling Psana interface initialization and data loading,
    this class also applies necessary preprocessing, such as masking bad
    pixels. It supports standard PyTorch dataset functionalities like length
    querying and item access, while ensuring the data and interfaces are
    initialized on-demand for efficient multiprocessing.
    """

    def __init__(self, exp, run, mode, detector_name, img_mode, event_list = None):
        super().__init__()

        self.exp           = exp
        self.run           = run
        self.mode          = mode
        self.detector_name = detector_name
        self.img_mode      = img_mode
        self.event_list    = event_list

        self._psana_img      = None
        self._bad_pixel_mask = None


    def _initialize_psana(self):
        exp           = self.exp
        run           = self.run
        mode          = self.mode
        detector_name = self.detector_name

        self._psana_img      = PsanaImg(exp, run, mode, detector_name)
        self._bad_pixel_mask = self._psana_img.create_bad_pixel_mask()


    def __len__(self):
        if self._psana_img is None:
            self._initialize_psana()

        return len(self._psana_img) if self.event_list is None else len(self.event_list)


    def __getitem__(self, idx):
        if self._psana_img is None:
            self._initialize_psana()

        # Fetch the event based on idx...
        event = idx if self.event_list is None else self.event_list[idx]

        # Fetch pixel data using psana...
        data = self._psana_img.get(event, None, self.img_mode)    # (B, H, W) or (H, W)

        if data is None:
            data = np.zeros_like(self._bad_pixel_mask, dtype = np.float32)

        # Mask out bad pixels...
        data = apply_mask(data, self._bad_pixel_mask, mask_value = 0)

        # Unify the data dimension...
        if data.ndim == 2: data = data[None,]    # (H, W) -> (1, H, W)

        # Build metadata...
        metadata = np.array([ (idx, event, panel_idx_in_batch) for panel_idx_in_batch, _ in enumerate(data) ], dtype = np.int32)

        return data, metadata
