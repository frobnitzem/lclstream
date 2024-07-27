from enum import Enum
from pydantic import BaseModel

class DataRequest(BaseModel):
    exp          : str
    run          : int
    access_mode  : str
    detector_name: str
    mode         : str = 'calib'
    addr         : str

class ImageRetrievalMode(str, Enum):
    raw = "raw"
    calib = "calib"
    image = "image"
    mask = "mask"

class AccessMode(str, Enum):
    idx = "idx"
    smd = "smd"
