import numpy as np

# That very smart code was taken from http://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python/30110497
def im2col(array, size, stride=1, padding=0):
    # Add padding to our array
    padded_array = np.pad(
        array,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode='constant'
    )
    # Get the shape of our newly made array
    H,W = np.shape(padded_array)
    # Get the extent
    extent = H - size + 1

    # Start index
    start_idx = np.arange(size)[:, None] * H + np.arange(size)
    offset_idx = np.arange(extent)[:, None] * H + np.arange(extent)

    return np.take(
        array, 
        np.ravel(start_idx)[:, None] + np.ravel(offset_idx)[::stride]
    )