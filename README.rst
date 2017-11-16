Neural Style Transfer
=======
This repository is the implementation of the article on `Neural Style Transfer <https://arxiv.org/abs/1508.06576>`_.


The implementation is in `TensorFlow <>`_, and uses `VGG16 <https://arxiv.org/abs/1409.1556>`_ which needs to be downloaded from the internet (the vgg16.py enables downloading the pretrained model).


Installation
------------
Please make sure you have installed the requirements before executing the python scripts.


**Install**


.. code-block:: bash

  pip install tensorflow-gpu
  pip install numpy
  pip install Pillow 
  pip install matplotlib

Usage
-------------
The repository primarily includes a script to transfer style from a target (style) to a source image - *python transfer_style.py path_to_source path_to_style path_to_save_image*.


