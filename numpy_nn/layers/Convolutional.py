import numpy as np

class Convolutional:
    '''
    2D Convolutional layer.

    Parameters:
    -----------
    num_filters (int) : Number of photo maps
    shape (width, height) : Width, Height of each filters

    '''

    def __init__(self, num_filters, shape, strides, padding):
        self.num_filters = num_filters
        self.shape = shape
        self.strides = strides
        self.padding = padding

        self.filters = np.random.randn(num_filters, shape[0], shape[1]) / 9

    def iterate_regions(self, image):
        # image needs to be in shape of (batch, height, width, channel)
        if len(image.shape) == 4:
            h, w, c = image.shape[1:]
        else:
            h,w = image.shape[1],image.shape[2]
            image = image.reshape((-1,h,w,1))
            h, w, c = image.shape[1:]
        

        
        
class Conv(Convolutional):

    def __init__(self, num_filters, shape):
        super(num_filters, shape)