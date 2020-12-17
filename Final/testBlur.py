import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.io as io
import glob
import blur


IMG_SIZE = [ 224, 224 ]

dataDir = 'images'
fileName = 'ILSVRC2012_val_00000004.JPEG'
I = io.imread( '%s/%s'%(dataDir,fileName) )

image = tf.image.resize( I, IMG_SIZE )
image /= 255

image = blur.applyBlur( image, 50, 170, 60 )
print( np.min( image ) )
print( np.max( image ) )

plt.imshow( image )
plt.show()