import numpy as np
import tensorflow as tf
import download
import os

########################################################################
# Various directories and file-names.

# The pre-trained VGG16 model is taken from this tutorial:
# https://github.com/pkmital/CADL/blob/master/session-4/libs/vgg16.py

# The class-names are available in the following URL:
# https://s3.amazonaws.com/cadl/models/synset.txt

# Internet URL for the file with the VGG16 model.
# Note that this might change in the future and will need to be updated.
data_url = "https://s3.amazonaws.com/cadl/models/vgg16.tfmodel"

# Directory to store the downloaded data.
data_dir = "vgg16/"

# File containing the TensorFlow graph definition. (Downloaded)
path_graph_def = "vgg16.tfmodel"

#########i###############################################################

if not os.path.exists(data_dir):
    os.makedirs(data_dir)


"""
Download the VGG16 model from the internet if it does not already
exist in the data_dir. WARNING! The file is about 550 MB.
"""

print("Downloading VGG16 Model ...")

# The file on the internet is not stored in a compressed format.
# This function should not extract the file when it does not have
# a relevant filename-extensions such as .zip or .tar.gz
download.maybe_download_and_extract(url=data_url, download_dir=data_dir)

