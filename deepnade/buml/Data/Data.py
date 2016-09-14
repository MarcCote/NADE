import numpy as np
import ctypes

def n_blocks(n_frames, block_length):
    """Calculates how many blocks of _block_size_ frames with frame_shift separation can be taken from n_frames"""
    return n_frames - block_length + 1

def expand_array_in_blocks(m, block_length, offset):
    r, c = m.shape
    r = n_blocks(r, block_length)
    if offset > 0:
        block_length = 1
    c = c * block_length
    output = np.ndarray((r, c), dtype=m.dtype)
    dst = output.ctypes.data
    frame_size = m.strides[0]
    for i in xrange(r):
        src = m.ctypes.data + int((i + offset) * frame_size)
        for j in xrange(block_length):
            ctypes.memmove(dst, src, frame_size)
            dst += int(frame_size)
            src += int(frame_size)
    return output
