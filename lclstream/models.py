from pydantic import BaseModel

class DataRequest(BaseModel):
    exp          : str
    run          : int
    access_mode  : str
    detector_name: str
    mode         : str = 'calib'
    addr         : str
