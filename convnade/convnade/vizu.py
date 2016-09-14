import numpy as np


def concatenate_images(images, shape=None, dim=None, border_size=0, clim=(-1, 1)):
    """
    Parameters
    ----------
    images : list of 1D, 2D, 3D arrays
    shape : (height, width)
        Shape of individual image
    dim : tuple (nrows, ncols)
    border_size : int
    """
    if dim is None:
        if type(images) is np.ndarray and images.ndim == 3:
            dim = images.shape[:2]
            images = images.reshape(-1, images.shape[2])
        else:
            dim = (int(np.ceil(np.sqrt(len(images)))), ) * 2

    if shape is None and images[0].ndim == 2:
        shape = images[0].shape

    img_shape = (dim[0] * (shape[0] + 2*border_size)), (dim[1] * (shape[1] + 2*border_size))
    img = np.ones(img_shape, dtype=float) * clim[0]

    for i, image in enumerate(images):
        row = i // dim[1]
        col = i % dim[1]
        starty, endy = row * (shape[0] + 2*border_size), (row+1) * (shape[0] + 2*border_size)
        startx, endx = col * (shape[1] + 2*border_size), (col+1) * (shape[1] + 2*border_size)

        img[starty+border_size:endy-border_size, startx+border_size:endx-border_size] = image.reshape(shape)
        pixels = img[starty+border_size:endy-border_size, startx+border_size:endx-border_size]
        pixels[:, :] = np.maximum(pixels, clim[0])
        pixels[:, :] = np.minimum(pixels, clim[1])

    return img
