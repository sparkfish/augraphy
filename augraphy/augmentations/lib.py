"""This module contains functions generally useful for building augmentations."""

import numpy as np
from sklearn.datasets import make_blobs
import random
import cv2

def addNoise(image, intensity_range=(0.1, 0.2), color_range=(0, 224)):
    """Applies random noise to the input image.

    :param image: The image to noise.
    :type image: numpy.array
    :param intensity_range: Pair of bounds for intensity sample.
    :type intensity_range: tuple, optional
    :param color_range: Pair of bounds for 8-bit colors.
    :type color_range: tuple, optional
    """

    intensity = random.uniform(intensity_range[0], intensity_range[1])
    noise = (
        lambda x: random.randint(color_range[0], color_range[1])
        if (x == 0 and random.random() < intensity)
        else x
    )
    add_noise = np.vectorize(noise)

    return add_noise(image)


def _create_blob(
    size_range=(10, 20),
    points_range=(5, 25),
    std_range=(10, 75),
    features_range=(15, 25),
    value_range=(180, 250),
):
    """Generates a Gaussian noise blob for placement in an image.
    To be used with _apply_blob()

    :param size_range: Pair of ints determining the range from which the
           diameter of a blob is sampled.
    :type size_range: tuple, optional
    :param points_range: Pair of ints determining the range from which the
           number of points in a blob is sampled.
    :type points_range: tuple, optional
    :param std_range: Pair of ints determining the range from which the
           standard deviation of the blob distribution is sampled.
    :type std_range: tuple, optional
    :param features_range: Pair of ints determining the range from which the
           number of features in the blob is sampled.
    :type features_range: tuple, optional
    :param value_range: Pair of ints determining the range from which the
           value of a point in the blob is sampled.
    :type value_range: tuple, optional
    """
    size = random.randint(size_range[0], size_range[1])
    std = random.randint(std_range[0], std_range[1]) / 100
    points = random.randint(points_range[0], points_range[1])
    features = random.randint(features_range[0], features_range[1])

    X, y = make_blobs(
        n_samples=points, cluster_std=[std], centers=[(0, 0)], n_features=features
    )  # , random_state=1)
    X *= size // 4
    X += size // 2
    X = [[int(item) for item in items] for items in X]
    blob = np.full((size, size, 1), 0, dtype="uint8")

    for point in X:
        if (
            point[0] < blob.shape[0]
            and point[1] < blob.shape[1]
            and point[0] > 0
            and point[1] > 0
        ):
            value = random.randint(value_range[0], value_range[1])
            blob[point[0], point[1]] = value

    return blob


def applyBlob(
    mask,
    size_range=(10, 20),
    points_range=(5, 25),
    std_range=(10, 75),
    features_range=(15, 25),
    value_range=(180, 250),
):
    """Places a Gaussian blob at a random location in the image.

    :param mask: The image to place the blob in.
    :type mask: numpy.array
    :param size_range: Pair of ints determining the range from which the
           diameter of a blob is sampled.
    :type size_range: tuple, optional
    :param points_range: Pair of ints determining the range from which the
           number of points in a blob is sampled.
    :type points_range: tuple, optional
    :param std_range: Pair of ints determining the range from which the
           standard deviation of the blob distribution is sampled.
    :type std_range: tuple, optional
    :param features_range: Pair of ints determining the range from which the
           number of features in the blob is sampled.
    :type features_range: tuple, optional
    :param value_range: Pair of ints determining the range from which the
           value of a point in the blob is sampled.
    :type value_range: tuple, optional
    """
    dim = min(mask.shape[0], mask.shape[1]) # we don't want to generate blobs larger than the mask

    # temporary local variables, in case
    size = size_range
    std = std_range

    # make sure we don't generate a blob larger than the mask
    if (2 * (size_range[1] + std_range[1]) > dim):
        # don't make a radius that won't fit in our mask
        size = (1,dim//2 - 1)
        # don't make a std.deviation that when added to radius, is larger than mask
        std = (0,dim//2 - size[1])

    blob = _create_blob(
        size, points_range, std, features_range, value_range
    )

    x_start = random.randint(0, mask.shape[1] - blob.shape[1])
    y_start = random.randint(0, mask.shape[0] - blob.shape[0])
    x_stop = x_start + blob.shape[1]
    y_stop = y_start + blob.shape[0]

    mask_chunk = mask[y_start:y_stop, x_start:x_stop]

    apply_chunk = np.vectorize(lambda x, y: max(x, y))

    mask_dim = len(mask.shape) # mask channels
    if mask_dim>2: # colour image or > 3 channels
        for i in range(mask_dim):
            mask[y_start:y_stop, x_start:x_stop,i] = apply_chunk(mask_chunk[:,:,i], blob[:, :, 0])
    else: # single channel grayscale or binary image
        mask[y_start:y_stop, x_start:x_stop] = apply_chunk(mask_chunk[:,:], blob[:, :,0])

    return mask


# create horizontal or vertical oriented patch of blobs
def create_blob_with_shape(
                size_range, 
                points_range,
                std_range,
                features_range,
                value_range, 
                new_size, 
                f_dir
                ):
    """Create horizonal or vertical shape image and filled with blobs. 

    :param size_range: Pair of ints determining the range from which the
           diameter of a blob is sampled.
    :type size_range: tuple, optional
    :param points_range: Pair of ints determining the range from which the
           number of points in a blob is sampled.
    :type points_range: tuple, optional
    :param std_range: Pair of ints determining the range from which the
           standard deviation of the blob distribution is sampled.
    :type std_range: tuple, optional
    :param features_range: Pair of ints determining the range from which the
           number of features in the blob is sampled.
    :type features_range: tuple, optional
    :param value_range: Pair of ints determining the range from which the
           value of a point in the blob is sampled.
    :type value_range: tuple, optional
    :new_size: Pair of ints determining the scaling factor of image shape.
    :type new_size: tuple, optional
    :f_dir: Flag to specify image shape, 0=vertical, 1=horizontal
    :type f_dir: int, optional
    """
    
    new_length = size_range[1] * new_size
    ori_length = size_range[1]
    interval = int(np.ceil(size_range[1] /2))
    
    if f_dir:
        img_blobs = np.zeros((ori_length, new_length))
    else:
        img_blobs = np.zeros((new_length, ori_length))
    
    ysize,xsize = img_blobs.shape
    
    apply_chunk = np.vectorize(lambda x, y: y if y != 0 else max(x, y))
    
    for i in range(0, new_length-ori_length+1, interval ):
        
        blob = _create_blob(size_range, points_range, std_range, features_range, value_range)
        blob = blob[:,:,0]
        
        ybsize, xbsize = blob.shape
        shape_max = size_range[1]
        
        # add small discontinuous lines
        img_lines = np.zeros_like(blob)
        ylsize, xlsize = img_lines.shape  
        
        if random.random()>0.5:
            h_dir = 1
        else:
            h_dir = 0
        
        for xl in range(xlsize): 
            if random.randint(1,10)>6:
                n_line = random.randint(1, np.ceil(ylsize/20)) 
                for j in range(n_line):
                    line_length = random.randint(1, np.ceil(ylsize/30))  
                    ylstart = random.randint(0, ylsize-line_length) 
                    ylend = ylstart + line_length 
                    
                    if h_dir:
                        img_lines[ylstart:ylend,xl] = random.randint(5,25)
                    else:
                        img_lines[xl, ylstart:ylend] = random.randint(5,25)
        
        if f_dir: 
            # get start coordinates
            ystart = random.randint(0,ysize-ybsize )
            xstart= random.randint(i,i+(shape_max-xbsize) )
            yend= ystart+ybsize
            xend = xstart+xbsize
            # apply blobs of noise and lines
            img_blobs[ystart:yend,xstart:xend] = apply_chunk(img_blobs[ystart:yend,xstart:xend] , blob)
            img_blobs[ystart:yend,xstart:xend] = apply_chunk(img_blobs[ystart:yend,xstart:xend] , img_lines)
        else:
            # get start coordinates
            ystart = random.randint(i,i+(shape_max-ybsize) )
            xstart= random.randint(0,xsize-xbsize )
            yend= ystart+ybsize
            xend = xstart+xbsize
            # apply blobs of noise and lines
            img_blobs[ystart:yend,xstart:xend] = apply_chunk(img_blobs[ystart:yend,xstart:xend], blob)
            img_blobs[ystart:yend,xstart:xend] = apply_chunk(img_blobs[ystart:yend,xstart:xend], img_lines)


    return img_blobs

# extend from applyBlob - create patch of blobs at image corner
def applyBlob_corners(
        mask,
        size_range=(10, 20),
        points_range=(5, 25),
        std_range=(10, 75),
        features_range=(15, 25),
        value_range=(180, 250),
        scale_blob = (1,3),
        f_topleft= 1,
        f_topright= 1,
        f_bottomleft= 1,
        f_bottomright= 1,
        inverse=1,
        ):
    
    """Create blobs at corners of provided image. 
    :param mask: The image to place the blob in.
    :type mask: numpy.array
    :param size_range: Pair of ints determining the range from which the
           diameter of a blob is sampled.
    :type size_range: tuple, optional
    :param points_range: Pair of ints determining the range from which the
           number of points in a blob is sampled.
    :type points_range: tuple, optional
    :param std_range: Pair of ints determining the range from which the
           standard deviation of the blob distribution is sampled.
    :type std_range: tuple, optional
    :param features_range: Pair of ints determining the range from which the
           number of features in the blob is sampled.
    :type features_range: tuple, optional
    :param value_range: Pair of ints determining the range from which the
           value of a point in the blob is sampled.
    :type value_range: tuple, optional
    :scale_blob: Pair of ints determining the scaling factor of blobs.
    :type scale_blob: tuple, optional
    :f_topleft: Flag to enable blobs at image top left corner.
    :type f_topleft: int, optional
    :f_topright: Flag to enable blobs at image top right corner.
    :type f_topright: int, optional
    :f_bottomleft: Flag to enable blobs at image bottom left corner.
    :type f_bottomleft: int, optional
    :f_bottomright: Flag to enable blobs at image bottom right corner.
    :type f_bottomright: int, optional
    :inverse: Flag to select a lower non zero value from apply_chunk,
            instead of a higher value.
    :type inverse: int, optional
    """
    
    if inverse: apply_chunk = np.vectorize(lambda x, y: y if y != 0 else max(x, y))
    else:       apply_chunk = np.vectorize(lambda x, y: max(x, y))

    # create patch of blobs
    new_size_v = random.randint(1,2)
    img_blobs_v = create_blob_with_shape(size_range, points_range,std_range,features_range,value_range, new_size_v, 0)
    new_size_h = random.randint(1,2)
    img_blobs_h = create_blob_with_shape(size_range, points_range,std_range,features_range,value_range, new_size_h, 1)
     
    scale = random.randint(scale_blob[0],scale_blob[1])
    img_blobs_h = cv2.resize(img_blobs_h, (img_blobs_h.shape[1]*scale,img_blobs_h.shape[0]*scale), interpolation = cv2.INTER_AREA)
    scale = random.randint(scale_blob[0],scale_blob[1]) 
    img_blobs_v = cv2.resize(img_blobs_v, (img_blobs_v.shape[1]*scale,img_blobs_v.shape[0]*scale), interpolation = cv2.INTER_AREA)
    
    ysize, xsize = mask.shape
    
    # resolve size issue, to not exceeding input mask size
    if (img_blobs_h.shape[0]>ysize) and (img_blobs_h.shape[0]>ysize) :
        img_blobs_h = cv2.resize(img_blobs_h, (xsize,ysize), interpolation = cv2.INTER_AREA)
    elif (img_blobs_h.shape[0]>ysize):
        img_blobs_h = cv2.resize(img_blobs_h, (img_blobs_h.shape[1],ysize), interpolation = cv2.INTER_AREA)
    elif (img_blobs_h.shape[1]>xsize):
        img_blobs_h = cv2.resize(img_blobs_h, (xsize,img_blobs_h.shape[0]), interpolation = cv2.INTER_AREA)
    
    if (img_blobs_v.shape[0]>ysize) and (img_blobs_v.shape[0]>ysize) :
        img_blobs_v = cv2.resize(img_blobs_v, (xsize,ysize), interpolation = cv2.INTER_AREA)
    elif (img_blobs_v.shape[0]>ysize):
        img_blobs_v = cv2.resize(img_blobs_v, (img_blobs_v.shape[1],ysize), interpolation = cv2.INTER_AREA)
    elif (img_blobs_v.shape[1]>xsize):
        img_blobs_v = cv2.resize(img_blobs_v, (xsize,img_blobs_v.shape[0]), interpolation = cv2.INTER_AREA)
    
    # separate blobs into smaller patches
    img_blobs_h_half_top = img_blobs_h[:int(img_blobs_h.shape[0]/2), :]
    img_blobs_h_half_bottom = img_blobs_h[int(img_blobs_h.shape[0]/2):,:]
    img_blobs_h_half_topleft = img_blobs_h_half_top[:,:int(img_blobs_h_half_top.shape[1]/2)]
    img_blobs_h_half_topright = img_blobs_h_half_top[:,int(img_blobs_h_half_top.shape[1]/2):]
    img_blobs_h_half_bottomleft = img_blobs_h_half_bottom[:,:int(img_blobs_h_half_bottom.shape[1]/2)]
    img_blobs_h_half_bottomright = img_blobs_h_half_bottom[:,int(img_blobs_h_half_bottom.shape[1]/2):]
    img_blobs_v_half_left = img_blobs_v[:,:int(img_blobs_v.shape[1]/2)]
    img_blobs_v_half_right = img_blobs_v[:,int(img_blobs_v.shape[1]/2):]
    img_blobs_v_half_topleft = img_blobs_v_half_left[:int(img_blobs_v_half_left.shape[0]/2),:]
    img_blobs_v_half_bottomleft = img_blobs_v_half_left[int(img_blobs_v_half_left.shape[0]/2):,:]
    img_blobs_v_half_topright = img_blobs_v_half_right[:int(img_blobs_v_half_right.shape[0]/2),:]
    img_blobs_v_half_bottomright = img_blobs_v_half_right[int(img_blobs_v_half_right.shape[0]/2):,:]

    # get size of patches
    yhsize,  xhsize = img_blobs_h_half_top.shape
    yvsize,  xvsize = img_blobs_v_half_left.shape
    yhhsize,  xhhsize = img_blobs_h_half_topleft.shape
    yvvsize,  xvvsize = img_blobs_v_half_topleft.shape
   
    if f_topleft:
        # top line - left
        mask[:yhsize,:xhsize] = apply_chunk( mask[:yhsize,:xhsize] , img_blobs_h_half_bottom)
        # left line  - top
        mask[:yvsize,:xvsize] = apply_chunk(mask[:yvsize,:xvsize] , img_blobs_v_half_right)
        # top left corner - top
        mask[:yhhsize,:xhhsize] = apply_chunk(mask[:yhhsize,:xhhsize], img_blobs_h_half_bottomright)
        # top left corner - left
        mask[:yvvsize,:xvvsize] = apply_chunk(mask[:yvvsize,:xvvsize]  , img_blobs_v_half_bottomright)
    
    if f_topright:
        # top line - right
        mask[:yhsize,xsize-xhsize:xsize] = apply_chunk( mask[:yhsize,xsize-xhsize:xsize] , img_blobs_h_half_bottom)
        # right line  - top
        mask[:yvsize,xsize-xvsize:xsize] = apply_chunk(mask[:yvsize,xsize-xvsize:xsize] , img_blobs_v_half_left)
        # top right corner - top
        mask[:yhhsize,xsize-xhhsize:xsize] = apply_chunk(mask[:yhhsize,xsize-xhhsize:xsize], img_blobs_h_half_bottomleft)
        # top right corner - right
        mask[:yvvsize,xsize-xvvsize:xsize] = apply_chunk(mask[:yvvsize,xsize-xvvsize:xsize], img_blobs_v_half_bottomleft)
    
    if f_bottomleft:
        # bottom line - left
        mask[ysize-yhsize:ysize,:xhsize] = apply_chunk(mask[ysize-yhsize:ysize,:xhsize] , img_blobs_h_half_bottom)
        # left line  - bottom
        mask[ysize-yvsize:ysize,:xvsize] = apply_chunk(mask[ysize-yvsize:ysize,:xvsize], img_blobs_v_half_right)
        # bottom left corner - bottom
        mask[ysize-yhhsize:ysize,:xhhsize] = apply_chunk(mask[ysize-yhhsize:ysize,:xhhsize], img_blobs_h_half_topright)
        # bottom left corner - left
        mask[ysize-yvvsize:ysize,:xvvsize] = apply_chunk(mask[ysize-yvvsize:ysize,:xvvsize] , img_blobs_v_half_topright)
    
    if f_bottomright:
    
        # bottom line - right
        mask[ysize-yhsize:ysize,xsize-xhsize:xsize] = apply_chunk(mask[ysize-yhsize:ysize,xsize-xhsize:xsize] , img_blobs_h_half_top) 
        # right line  - bottom
        mask[ysize-yvsize:ysize,xsize-xvsize:xsize] = apply_chunk(mask[ysize-yvsize:ysize,xsize-xvsize:xsize] , img_blobs_v_half_left)
        # bottom right corner - bottom
        mask[ysize-yhhsize:ysize,xsize-xhhsize:xsize] = apply_chunk(mask[ysize-yhhsize:ysize,xsize-xhhsize:xsize] , img_blobs_h_half_topleft)
        # bottom right corner - right
        mask[ysize-yvvsize:ysize,xsize-xvvsize:xsize] = apply_chunk(mask[ysize-yvvsize:ysize,xsize-xvvsize:xsize] , img_blobs_v_half_topleft)
    
    return mask

# extend from applyBlob -create patch of blobs randomly in the image
def applyBlob_random(
    mask,
    size_range=(30, 60),
    points_range=(5, 25),
    std_range=(10, 75),
    features_range=(15, 25),
    value_range=(180, 250),
    scale_blob=(3,3),
    inverse=1,
    ):

    """Create patches of blobs randomly in the provided image. 
    :param mask: The image to place the blob in.
    :type mask: numpy.array
    :param size_range: Pair of ints determining the range from which the
           diameter of a blob is sampled.
    :type size_range: tuple, optional
    :param points_range: Pair of ints determining the range from which the
           number of points in a blob is sampled.
    :type points_range: tuple, optional
    :param std_range: Pair of ints determining the range from which the
           standard deviation of the blob distribution is sampled.
    :type std_range: tuple, optional
    :param features_range: Pair of ints determining the range from which the
           number of features in the blob is sampled.
    :type features_range: tuple, optional
    :param value_range: Pair of ints determining the range from which the
           value of a point in the blob is sampled.
    :type value_range: tuple, optional
    :scale_blob: Pair of ints determining the scaling factor of blobs.
    :type scale_blob: tuple, optional
    :inverse: Flag to select a lower non zero value from apply_chunk,
            instead of a higher value.
    :type inverse: int, optional
    """
    
    
    # initialization
    img_blobs_v_array = []
    img_blobs_h_array = []
    yv_size_all = []
    xv_size_all = []
    yh_size_all = []
    xh_size_all = []
    
    for i in range(2): # stack multiple times to create multiple layers of blobs 

        size_range= [int(size + (i*0.5)) for size in size_range]
        points_range=[max(int(points - (i)),1) for points in points_range]
        
        if inverse: apply_chunk = np.vectorize(lambda x, y: y if y != 0 else max(x, y))
        else:       apply_chunk = np.vectorize(lambda x, y: max(x, y))
    
        ysize, xsize = mask.shape
        scale_length = random.randint(1,1)
        
        # create patch of blobs
        new_size_v = random.randint(1,scale_length)
        img_blobs_v = create_blob_with_shape(size_range, points_range,std_range,features_range,value_range, new_size_v, 0)
        new_size_h = random.randint(1,scale_length)
        img_blobs_h = create_blob_with_shape(size_range, points_range,std_range,features_range,value_range, new_size_h, 1)
     
        # resize images based on input scale
        scale = random.randint(scale_blob[0],scale_blob[1])
        img_blobs_h = cv2.resize(img_blobs_h, (img_blobs_h.shape[1]*scale,img_blobs_h.shape[0]*scale), interpolation = cv2.INTER_AREA)
        scale = random.randint(scale_blob[0],scale_blob[1])
        img_blobs_v = cv2.resize(img_blobs_v, (img_blobs_v.shape[1]*scale,img_blobs_v.shape[0]*scale), interpolation = cv2.INTER_AREA)
    
        # resolve size issue, to not exceeding input mask size
        if (img_blobs_h.shape[0]>ysize) and (img_blobs_h.shape[0]>ysize) :
            img_blobs_h = cv2.resize(img_blobs_h, (xsize,ysize), interpolation = cv2.INTER_AREA)
        elif (img_blobs_h.shape[0]>ysize):
            img_blobs_h = cv2.resize(img_blobs_h, (img_blobs_h.shape[1],ysize), interpolation = cv2.INTER_AREA)
        elif (img_blobs_h.shape[1]>xsize):
            img_blobs_h = cv2.resize(img_blobs_h, (xsize,img_blobs_h.shape[0]), interpolation = cv2.INTER_AREA)
        
        if (img_blobs_v.shape[0]>ysize) and (img_blobs_v.shape[0]>ysize) :
            img_blobs_v = cv2.resize(img_blobs_v, (xsize,ysize), interpolation = cv2.INTER_AREA)
        elif (img_blobs_v.shape[0]>ysize):
            img_blobs_v = cv2.resize(img_blobs_v, (img_blobs_v.shape[1],ysize), interpolation = cv2.INTER_AREA)
        elif (img_blobs_v.shape[1]>xsize):
            img_blobs_v = cv2.resize(img_blobs_v, (xsize,img_blobs_v.shape[0]), interpolation = cv2.INTER_AREA)
    
        # pack patch of image with blobs
        img_blobs_v_array.append(img_blobs_v)
        img_blobs_h_array.append(img_blobs_h)

        # pack size
        yv_size_all.append(img_blobs_v.shape[0])
        xv_size_all.append(img_blobs_v.shape[1])
        yh_size_all.append(img_blobs_h.shape[0])
        xh_size_all.append(img_blobs_h.shape[1])

    max_yv = np.max(yv_size_all)
    max_xv = np.max(xv_size_all)
    max_yh = np.max(yh_size_all)
    max_xh = np.max(xh_size_all)
    
    xh_start_fixed = random.randint(0,xsize-max_xh)
    yh_start_fixed = random.randint(0,ysize-max_yh)
    
    xv_start_fixed = random.randint(0,xsize-max_xv)
    yv_start_fixed = random.randint(0,ysize-max_yv)

    for (img_blobs_v, img_blobs_h) in zip(img_blobs_v_array,img_blobs_h_array):

        yhsize,  xhsize = img_blobs_h.shape
        yvsize,  xvsize = img_blobs_v.shape

        # to let all patches of image having same origin
        xh_start = xh_start_fixed + int(np.floor((max_xh - xhsize)/2))
        yh_start = yh_start_fixed + int(np.floor((max_yh - yhsize)/2))
        xh_end = xh_start + xhsize
        yh_end = yh_start + yhsize
        
        # to let all patches of image having same origin
        xv_start = xv_start_fixed + int(np.floor((max_xv - xvsize)/2))
        yv_start = yv_start_fixed + int(np.floor((max_yv - yvsize)/2))
        xv_end = xv_start + xvsize
        yv_end = yv_start + yvsize
        
        # apply blobs to image
        mask[yh_start:yh_end,xh_start:xh_end] = apply_chunk( mask[yh_start:yh_end,xh_start:xh_end] , img_blobs_h)
        mask[yv_start:yv_end,xv_start:xv_end] = apply_chunk( mask[yv_start:yv_end,xv_start:xv_end] , img_blobs_v)

    return mask


# create small tiny spot of blobs everywhere in the image
def applyBlob_full(mask, blob_density):
    """Create tiny blobs randomly in the provided image. 
    :param mask: The image to place the blob in.
    :type mask: numpy.array
    """
    
    # add small discontinuous lines
    img_lines = mask.copy()
    ylsize, xlsize = img_lines.shape  
    
    for xl in range(xlsize): 
        if random.randint(1,100)>95:
            n_line = random.randint(1, int(np.ceil(ylsize/100)*blob_density)) 
            for j in range(n_line):
                line_length = random.randint(1, np.ceil(ylsize/200))  
                ylstart = random.randint(0, ylsize-line_length) 
                ylend = ylstart + line_length 
                
                if random.random()>0.5:
                    xle = xl+random.randint(1,3)
                    xle = min(xlsize,xle)          
                    img_lines[ylstart:ylend,xl:xle] = random.randint(5,25)
                else:
                    xle = xl+random.randint(1,3)
                    xle = min(ylsize,xle)  
                    img_lines[xl:xle, ylstart:ylend] = random.randint(5,25)

    return img_lines


