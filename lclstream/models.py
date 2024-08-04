from enum import Enum
from pydantic import BaseModel

class ImageRetrievalMode(str, Enum):
    raw = "raw"
    calib = "calib"
    image = "image"
    mask = "mask"

class AccessMode(str, Enum):
    idx = "idx"
    smd = "smd"

class DataRequest(BaseModel):
    exp          : str
    run          : int
    access_mode  : AccessMode
    detector_name: str
    mode         : ImageRetrievalMode #= ImageRetrievalMode.calib
    addr         : str
