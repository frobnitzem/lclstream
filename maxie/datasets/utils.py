import numpy as np

def apply_mask(data, mask, mask_value = np.nan):
    """
    Return masked data.

    Args:
        data: numpy.ndarray with the shape of (B, H, W).·
              - B: batch of images.
              - H: height of an image.
              - W: width of an image.

        mask: numpy.ndarray with the shape of (B, H, W).·

    Returns:
        data_masked: numpy.ndarray.
    """
    # Mask unwanted pixels with np.nan...
    data_masked = np.where(mask, data, mask_value)

    return data_masked
