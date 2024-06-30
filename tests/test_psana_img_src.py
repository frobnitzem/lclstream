import os
os.environ["RAND_PSANA"] = "1"

import pytest

from lclstream.psana_img_src import PsanaImgSrc

def test_img_src():
    ps = PsanaImgSrc('xpptut15', 630, 'idx', 'jungfrau1M')
    assert len(ps) > 0
    for img in ps("image"):
        assert len(img.shape) == 2

    with pytest.raises(NotImplementedError):
        ps.create_bad_pixel_mask()
