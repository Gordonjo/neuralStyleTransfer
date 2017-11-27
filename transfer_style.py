import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
from loss_functions import *
from style import *
from image_helpers import *
import pdb, sys, os

""" main script to use support to transfer style across images """

# argv[1] - path to source image 
# argv[2] - path to style image
# argv[3] - path for saved image

content_image = load_image(sys.argv[1],max_size=None)
style_image = load_image(sys.argv[2],max_size=None)

## identify the layers in VGG16 that we want to use for style transfer
content_layer_ids = [4,5,6,7]
# The VGG16-model has 13 convolutional layers.
# This selects all those layers as the style-layers.
# This is somewhat slow to optimize.
style_layer_ids = list(range(13))


## Now optimize the style transfer and time it
img = style_transfer(content_image=content_image,
                     style_image=style_image,
                     content_layer_ids=content_layer_ids,
                     style_layer_ids=style_layer_ids,
                     weight_content=1.5,
                     weight_style=10.0,
                     weight_denoise=0.3,
                     num_iterations=250,
                     step_size=10.0,
		     img_file=sys.argv[3])




