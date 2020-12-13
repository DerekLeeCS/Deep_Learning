import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io as io
import pickle
import os
import glob
import blur
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from collections import defaultdict
from collections import Counter
import linecache
import tracemalloc


# From:
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


# Constants
IMG_SIZE = [ 224, 224 ]
kVal = 5    # Top 5

# IMG_DIMS is [ None, IMG_SIZE, 3 ]
IMG_DIMS = [ None ]
IMG_DIMS.extend( IMG_SIZE )
IMG_DIMS.extend( [3] )

# Location of TFRecords
recPath = 'records'
recName = 'ImageNet'


# Measure memory usage
def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


tracemalloc.start()



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


def normalize_img( image, label, bbox ):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255, label, bbox


# Python function to manipulate dataset
def map_func( image, label, bbox ):
    """ Scales images to IMG_SIZE.
        Removes bounding box element of dataset."""

    # Deal with grayscale images
    if len( tf.shape(image) ) == 2:
        image = np.expand_dims( image, axis=-1 )
        image = tf.concat( [image, image, image], axis=-1 )

    image = tf.image.resize( image, IMG_SIZE )

    # image = blur.applyBlur( image, IMG_SIZE[0]//2, IMG_SIZE[1]//2, 70 )

    return image, label


# Function to define shape of tfds
def ensureShape( image, label ):

    # dims -> [ IMG_SIZE, 3 ]
    dims = []
    dims.extend( IMG_SIZE )
    dims.extend( [3] )

    image = tf.ensure_shape( image, dims )

    return image, label


def calcAcc( probs, truth, k ):

    numEx = tf.shape( probs )[0]

    correctBools = tf.math.in_top_k( truth[ np.arange( 1500,1500+numEx ) ], probs, kVal )
    numCorrect = tf.math.reduce_sum( tf.cast( correctBools, tf.float32 ) )
    print( numCorrect )
    print( numCorrect / tf.cast( numEx, tf.float32 ) )

    return
    

def sortRecs( rec ):

    fileName, _ = rec.split( '.' )
    _, num = fileName.split( '-' )
    return int(num)

    
if __name__ == "__main__":
    
    # Load data
    # fileName = [ os.path.join(recPath, recName + '-' + '1' + '.tfrecords'), 
    #         os.path.join(recPath, recName + '-' + '2' + '.tfrecords') ]
    # Iterate through all images of a specific extension in the specified directory
    # fileName = []
    # imgPath = os.path.join( recPath, '*.tfrecords' )

    # for filepath in glob.iglob( imgPath ):
    #     #print(filepath)
    #     fileName.append( filepath )

    # # Sort list of tfrecords in numerical ascending order b/c ground truth labels are in that order
    # fileName.sort( key=sortRecs )  
    # print( fileName )

    fileName = os.path.join(recPath, recName + '-' + '2' + '.tfrecords')

    tfr_dataset = tf.data.TFRecordDataset(fileName) 
    dataset = tfr_dataset.map(_parse_tfr_element)
    
    print("\n\n")
    print( dataset.element_spec )

    # Map dataset
    ds = dataset.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print( ds.element_spec )

    # Map using tf.py_function
    ds = ds.map( lambda image, label, bbox: tf.py_function(func=map_func,
          inp=[image, label, bbox], Tout=[tf.float32, tf.int32]), 
          num_parallel_calls=tf.data.experimental.AUTOTUNE )

    # Set (previously known) shapes of images
    ds = ds.map( 
        ensureShape, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds = ds.take( 500 )
    print( ds.element_spec )

    # for img, label in ds.take(3):
        
    #     fig, ax = plt.subplots()
    #     print( tf.shape( img ) )
    #     print( label )
    #     img = blur.applyBlur( img, 60, 60, 40 )
    #     ax.imshow( img )
    #     plt.show()

    # Load mappings from a file
    # with open('INToTf.pkl', 'rb') as f: 
    #     data = f.read() 
    #     INToTf = pickle.loads(data) 

    # Load ground truth labels from a file
    with open('truth.pkl', 'rb') as f: 
        data = f.read() 
        truthDict = pickle.loads(data)

    # Load mapped ground truth labels from a file
    with open('truthMapped.pkl', 'rb') as f:
        data = f.read() 
        mappedTruthDict = pickle.loads(data)  
    
    # Use mappings to get the correct labels
    mappedTruthDict = { k:v[0] for (k,v) in mappedTruthDict.items() }
    truth = np.array( list( mappedTruthDict.values() ) )

    # Load pre-trained model
    model = tf.keras.Sequential([

        hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v1/classification/4"),

    ])
    model.build( IMG_DIMS )  # Batch input shape
    model.summary()

    # # Test the model
    # ds = ds.cache().batch(128).prefetch(tf.data.experimental.AUTOTUNE)
    # #ds = ds.cache().prefetch(tf.data.experimental.AUTOTUNE)
    # logits = model.predict(ds)
    # print( tf.shape(logits) )
    # print( tf.math.top_k(logits, k=kVal) )

    # # Calculate accuracy
    # calcAcc( logits, truth, kVal )

    # Store the labels
    topKOri = np.zeros( [1,kVal] )
    topKSq = np.zeros( [kVal,kVal] )
    confs = np.zeros( [kVal,kVal] )

    numTested = 500
    numCorr = 0
    count = 0

    # Loss function
    bce = tf.keras.losses.BinaryCrossentropy( from_logits=True )

    for I, tempLabel in ds.take( numTested ):
    
        img = I
        img = blur.applyBlur( img, 112, 112, 70 )
        img = tf.reshape( img, [1, 224, 224, 3] )
        imageVar = tf.Variable( img )

        trueLabel = truth[ count+1500 ]
        count += 1

        for topLabel in range(kVal):

            with tf.GradientTape() as tape:

                # Watch the input image to compute saliency map later
                tape.watch( imageVar )

                # Forward-pass to get initial predictions
                probs = model( imageVar )

                # Get top-k predictions
                logits, preds = tf.math.top_k(probs, k=kVal) # Throw out the probs for each top prediction (included in probs variable)
                topKOri = preds
                dictOri = dict( zip( preds[0].numpy(), logits[0].numpy() ) )

                true = tf.one_hot( preds[0], len( probs[0] )  )  # One-hot encode the predictions to the same size as probs

                loss = bce( probs[0], true[topLabel] )


            grads = abs( tape.gradient( loss, imageVar ) )
            grads = tf.reduce_max( grads[0], axis=-1 )

            thres = tfp.stats.percentile( grads, q=80 )
            grads = tf.keras.activations.relu( grads, threshold=thres )

            image = grads.numpy()

            # apply threshold
            thres = threshold_otsu(image)
            bw = closing(image > thres, square(3))

            # label image regions
            label_image = label(bw)
            # to make the background transparent, pass the value of `bg_label`,
            # and leave `bg_color` as `None` and `kind` as `overlay`
            image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

            curMaxArea = 0

            # Get max region
            for region in regionprops(label_image):
                
                if region.area >= curMaxArea:

                    curMaxArea = region.area
                    maxRegion = region


            minr, minc, maxr, maxc = maxRegion.bbox

            # Second pass
            height = maxc - minc
            width = maxr - minr
            f_new = np.floor( np.max( [height, width] ) / 2 )
            f_new = np.max( [30, f_new] )   # Minimum foveal size
            centerX = minc + height/2
            centerY = minr + width/2
  
            imgSec = I
            #imgSec = blur.applyBlur( imgSec, centerX, centerY, f_new )
            imgSec = blur.applyBlur( imgSec, centerX, centerY, 70 )

            # Show image
            # fig, ax = plt.subplots()
            # plt.imshow( imgSec )
            # ax.add_patch( patches.Circle( (centerX, centerY), f_new, fill=False, color='r' ) )
            # plt.show()
            imgSec = tf.reshape( imgSec, [1, 224, 224, 3] )

            logits = model.predict( imgSec )
            conf, preds = tf.math.top_k(logits, k=kVal)
            confs[ topLabel ] = conf
            topKSq[ topLabel ] = preds

        # Map top-k into dicts
        dicts = []
        for i in range( kVal ):
            dicts.append( dict( zip( topKSq[i], confs[i] ) ) )

        # Get the highest confidences for each unique label
        dictTopK = defaultdict(int)

        # Do not include the original top-k
        dictTopK.update( dicts[0] )
        for i in range(1,kVal):
            dictTopK.update( (k,v) for k,v in dicts[i].items() if dictTopK[k] < v )

        # # Include the original top-k
        # dictTopK.update( dictOri )
        # for i in range(kVal):
        #     dictTopK.update( (k,v) for k,v in dicts[i].items() if dictTopK[k] < v )

        # Sort the dict in descending order
        # Get topK labels
        tupleTopK = sorted(dictTopK.items(), key=lambda x: x[1], reverse=True)[:kVal]

        dictTopK = {}

        # Get labels into a list
        newTopK = [ int(x[0]) for x in tupleTopK ]

        if trueLabel in newTopK:
            numCorr += 1
        else:
            print( newTopK )
            print( trueLabel )
            print( "Completed", count )

        # # For single pass
        # if trueLabel in topKOri[0]:
        #     numCorr += 1
        # else:
        #     print( topKOri )
        #     print( trueLabel )
        #     print( "Completed", count )

        # if count % 10 == 0:
        #     snapshot = tracemalloc.take_snapshot()
        #     display_top(snapshot)

    print( numCorr, numTested )
    print( numCorr/numTested )