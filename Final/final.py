import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io as io
import os


# From:
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


# Constants
BATCH_SIZE = 16
IMG_SIZE = [ 224, 224 ]

# IMG_DIMS is [ None, IMG_SIZE, 3 ]
IMG_DIMS = [ None ]
IMG_DIMS.extend( IMG_SIZE )
IMG_DIMS.extend( [3] )

# Location of TFRecords
recPath = 'records'
recName = 'ImageNet'


# Display some plots
def testLoad( data, info ):

    plt.figure( figsize=(10,10) )
    i=0
    for image, label in data:

        if i == 25:
            break
        plt.subplot( 5, 5, i+1 )
        plt.xticks([])
        plt.yticks([])
        plt.grid( False )
        plt.imshow( image )
        label = info.features["label"].int2str(label)
        plt.xlabel( label )
        i += 1

    plt.show()


# Read TFRecord file from:
# https://stackoverflow.com/questions/47861084/how-to-store-numpy-arrays-as-tfrecord
def _parse_tfr_element(element):

    parse_dic = {
            'image': tf.io.FixedLenFeature([], tf.string), # Note that it is tf.string, not tf.float32
            'label': tf.io.FixedLenFeature([], tf.string),
            'bbox': tf.io.FixedLenFeature([], tf.string),
    }
    example_message = tf.io.parse_single_example(element, parse_dic)

    b_image = example_message['image'] # get byte string
    b_bbox = example_message['bbox']
    b_label = example_message['label']
    
    img = tf.io.parse_tensor(b_image, out_type=tf.uint8) # restore 2D array from byte string
    bbox = tf.io.parse_tensor(b_bbox, out_type=tf.int32)
    label = tf.io.parse_tensor(b_label, out_type=tf.string)
    label = int(label)

    return img, label, bbox


def normalize_img(image, label, bbox):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255, label, bbox


# Python function to manipulate dataset
def map_func(image, label, bbox ):
    """ Scales images to IMG_SIZE.
        Removes bounding box element of dataset."""
    image = tf.image.resize( image, [ 224, 224 ] )
    return image, label


if __name__ == "__main__":
    
    # Load data
    filename = [ os.path.join(recPath, recName + '-' + '1' + '.tfrecords'), 
            os.path.join(recPath, recName + '-' + '2' + '.tfrecords') ]
    tfr_dataset = tf.data.TFRecordDataset(filename) 
    dataset = tfr_dataset.map(_parse_tfr_element)

    print("\n\n\n\n")
    print( dataset.element_spec )

    # Map dataset
    ds = dataset.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print( ds.element_spec )

    # Map using tf.py_function
    ds = ds.map( lambda image, label, bbox: tf.py_function(func=map_func,
          inp=[image, label, bbox], Tout=[tf.float32,tf.int32]) )

    for img, label in ds.take(3):
        
        fig, ax = plt.subplots()
        print( tf.shape( img ) )
        print( label )
        ax.imshow( img )
        plt.show()

    # Load pre-trained model
    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v1/classification/4")
    ])
    model.build( IMG_DIMS )  # Batch input shape
    model.summary()

    model.compile()

    # Test the model
    ds = ds.batch(128).cache().prefetch(tf.data.experimental.AUTOTUNE)
    result = model.evaluate(ds)
    #model.load_weights( ckptName )

    # dataDir = 'images'
    # fileName = 'ILSVRC2012_val_00000002.JPEG'
    # I = io.imread( '%s/%s'%(dataDir,fileName) )
    # I = tf.convert_to_tensor(I)
    # I /= 255
    # print(I)
    # I = tf.image.resize( I, [ 224, 224 ] )
    # plt.imshow(I)
    # plt.show()
    # I = np.expand_dims( I, axis=0 )
    # print(tf.shape(I))
    # x = model.predict( I )
    # print(x)
    # print(tf.math.top_k(x, k=5))