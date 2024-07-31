from typing import List, Iterator
import numpy as np

class StubEvent:
    def __init__(self, m : int, n : int,
                 dtype : str ='float32',
                 sigma : float = 7.0) -> None:
        assert sigma > 0
        assert m > 0
        assert n > 0

        # Sigma is the width of features in k-space
        # smaller = smoother image
        self.m = m
        self.n = n
        self.nf = n//2 + 1
        self.dtype = dtype

        x = np.arange(m)
        y = np.arange(self.nf)
        scale = -0.5/sigma**2
        self.scale_x = np.exp(scale*x*x)
        self.scale_y = np.exp(scale*y*y)

    def read(self):
        Z = np.random.standard_normal((self.m, self.nf)) \
          + 1j*np.random.standard_normal((self.m, self.nf))
        symm(Z[:,0])
        if self.n % 2 == 0:
            symm(Z[:,-1])
        
        Z *= self.scale_x[:,None] * self.scale_y[None,:]
        return np.fft.irfft2(Z, s=(self.m, self.n)).astype(self.dtype)

def test_event():
    for m in [32, 35]:
        for n in [32, 127]:
            e = StubEvent(m, n)
            img = e.read()
            assert img.shape == (m, n)

# symmetrize [0, 1, ..., n//2| n//2+1, ..., n-1]
# even n: [0, 1, 2, *3| 4, 5]
# odd n:  [0, 1, *2| 3, 4]
# so its iFFT is real.
def symm(x):
    x[0:1] = x[0:1].real
    n = len(x)
    mid = n//2
    if n % 2 == 0:
        x[mid:mid+1] = x[mid:mid+1].real
        x[mid+1:] = x[mid-1:0:-1].conj()
        mid -= 1
    else:
        x[mid+1:] = x[mid:0:-1].conj()

class StubRun:
    def times(self) -> List[float]:
        return np.arange(20).tolist()
    def event(self, time : int) -> StubEvent:
        return StubEvent(1024, 1024, 'float32')

class StubDataSource:
    def __init__(self, source_id : str) -> None:
        self.source_id = source_id

    def runs(self) -> Iterator[StubRun]:
        yield StubRun()

class StubDetector:
    def __init__(self, name : str) -> None:
        self.name = name

    def raw(self, event):
        return event.read()
    def calib(self, event):
        return event.read()
    def image(self, event):
        return event.read()
    def mask(self, run, **kws):
        raise NotImplementedError()

try:
    from psana import DataSource, Detector, MPIDataSource
except ImportError:
    import os
    if os.environ.get("RAND_PSANA", "0") == "1":
        DataSource = StubDataSource
        Detector = StubDetector
        MPIDataSource = StubDataSource
    else:
        raise
