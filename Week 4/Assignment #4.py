import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import pickle

C10 = False     # If true, training on CIFAR10; Otherwise, training on CIFAR100
C10 = True     # If true, training on CIFAR10; Otherwise, training on CIFAR100

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
NUM_EPOCHS = 160
BATCH_SIZE = 64
VALID_SIZE = 0.2    # Size of validation set
FIG_HEIGHT = 5      # Height of figure for final plot

# Model optimizer
OPTIMIZER = tf.keras.optimizers.Adam()

# CIFAR10
if C10:

    NUM_CLASSES = 10
    REGULARIZER = tf.keras.regularizers.l2( 0.001 )
    METRIC = tf.keras.metrics.SparseCategoricalAccuracy()
    PLOT_METRIC = 'sparse_categorical_accuracy'
    PLOT_LABEL = 'Accuracy'
    FIG_WIDTH = 10

# CIFAR100
else:

    NUM_CLASSES = 100
    REGULARIZER = tf.keras.regularizers.l2( 0.001 )
    METRIC = tf.keras.metrics.SparseTopKCategoricalAccuracy( k = 5 )    # Top 5 Accuracy
    PLOT_METRIC = 'sparse_top_k_categorical_accuracy'
    PLOT_LABEL = 'Top 5 Accuracy'
    FIG_WIDTH = 15


# Data processing from:
# https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle( file ):

    with open( file, 'rb' ) as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict


# loadTrain - True for training, False for testing
# C10 - True for CIFAR-10, False for CIFAR-100
def loadCIFAR( loadTrain, C10 ):

    # CIFAR 10
    file10Train = [ 

        'cifar-10-batches-py/data_batch_1',
        'cifar-10-batches-py/data_batch_2',
        'cifar-10-batches-py/data_batch_3',
        'cifar-10-batches-py/data_batch_4',
        'cifar-10-batches-py/data_batch_5',

    ]
    file10Test = [ 'cifar-10-batches-py/test_batch' ]

    # CIFAR 100
    file100Train = [ 'cifar-100-python/train' ]
    file100Test = [ 'cifar-100-python/test' ]

    # Dictionaries
    tempDict = {}

    if C10:
        finalDict = { b'data': [], b'labels': [] }

        if loadTrain:
            fileList = file10Train
        else:
            fileList = file10Test
            
    else:
        finalDict = { b'data': [], b'fine_labels': [] }

        if loadTrain:
            fileList = file100Train
        else:
            fileList = file100Test

    # Get keys representing data and labels
    dictData = list( finalDict.keys() )[0]
    dictLabels = list( finalDict.keys() )[1]

    # Load to dictionary
    for fileName in fileList:

        tempDict.update( unpickle( fileName ) )
        finalDict[ dictData ].extend( tempDict[ dictData ] )
        finalDict[ dictLabels ].extend( tempDict[ dictLabels ] )
    
    return finalDict


# Testing if images were properly loaded from:
# https://www.tensorflow.org/tutorials/images/cnn
def testLoad( img, label ):

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure( figsize=(10,10) )
    for i in range(25):
        plt.subplot( 5, 5, i+1 )
        plt.xticks([])
        plt.yticks([])
        plt.grid( False )
        plt.imshow( img[i] )
        plt.xlabel( class_names[ label[i] ] )

    plt.show()


# Decreases learning rate at specific epochs from:
# https://towardsdatascience.com/understand-and-implement-resnet-50-with-tensorflow-2-0-1190b9b52691
def lrdecay(epoch):

    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1

    return lr


# Architecture based on Full Pre-Activation from:
# https://arxiv.org/pdf/1603.05027.pdf
# Implementation inspired by:
# https://www.tensorflow.org/tutorials/customization/custom_layers
class identityBlock( tf.keras.Model ):

    def __init__( self, filters ):

        super( identityBlock, self ).__init__( name = '' )
        f1, f2 = filters
        k = 3   # Kernel size

        self.conv2a = tf.keras.layers.Conv2D( f1, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', kernel_regularizer = REGULARIZER )
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D( f1, kernel_size = (k, k), strides = (1, 1), padding = 'same', kernel_regularizer = REGULARIZER )
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D( f2, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', kernel_regularizer = REGULARIZER )
        self.bn2c = tf.keras.layers.BatchNormalization()    


    def call( self, inputTensor, training = False ):        
        
        x = inputTensor

        # Block 1
        x = self.bn2a( x, training = training )
        x = tf.nn.leaky_relu( x )
        x = self.conv2a( x )   

        # Block 2
        x = self.bn2b( x, training = training )
        x = tf.nn.leaky_relu( x )
        x = self.conv2b( x )
        
        # Block 3
        x = self.bn2c( x, training = training )
        x = tf.nn.relu( x )
        x = self.conv2c( x )
        
        # Output
        x += inputTensor
        
        return x


class convBlock( tf.keras.Model ):

    def __init__( self, filters, s ):

        super( convBlock, self ).__init__( name = '' )
        f1, f2 = filters
        k = 3   # Kernel size

        self.conv2a = tf.keras.layers.Conv2D( f1, kernel_size = (1, 1), strides = (s, s), padding = 'valid', kernel_regularizer = REGULARIZER )
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D( f1, kernel_size = (k, k), strides = (1, 1), padding = 'same', kernel_regularizer = REGULARIZER )
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D( f2, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', kernel_regularizer = REGULARIZER )
        self.bn2c = tf.keras.layers.BatchNormalization()

        self.conv2Shortcut = tf.keras.layers.Conv2D( f2, kernel_size = (1, 1), strides = (s, s), padding = 'valid', kernel_regularizer = REGULARIZER )
        self.bn2Shortcut = tf.keras.layers.BatchNormalization()
    

    def call( self, inputTensor, training = False ):     

        x = inputTensor
        xShort = inputTensor

        # Block 1
        x = self.bn2a( x, training = training )
        x = tf.nn.leaky_relu( x )
        x = self.conv2a( x ) 

        # Block 2
        x = self.bn2b( x, training = training )
        x = tf.nn.leaky_relu( x )
        x = self.conv2b( x )

        # Block 3
        x = self.bn2c( x, training = training )
        x = tf.nn.relu( x )
        x = self.conv2c( x )

        # Shortcut
        xShort = self.bn2Shortcut( xShort, training = training )
        xShort = tf.nn.relu( xShort )
        xShort = self.conv2Shortcut( xShort ) 

        # Output
        x += xShort
        
        return x


# Module containing Image Classification model
# Architecture based on:
# https://towardsdatascience.com/understand-and-implement-resnet-50-with-tensorflow-2-0-1190b9b52691
class imgClassMod( tf.Module ):

    def __init__( self ):

        self.model = tf.keras.models.Sequential()

        # Stem Layer
        self.model.add( tf.keras.layers.ZeroPadding2D( (3, 3) ) )
        self.model.add( tf.keras.layers.Conv2D( 64, (7, 7), strides = (2, 2) ) )
        self.model.add( tf.keras.layers.BatchNormalization() )
        self.model.add( tf.keras.layers.ReLU() )
        self.model.add( tf.keras.layers.MaxPooling2D( (3, 3), strides = (2, 2) ) )

        # Hidden Layers
        # Stage 1
        self.model.add( convBlock( filters = [ 64, 256 ], s = 1 ) )
        for _ in range(2):
            self.model.add( identityBlock( filters = [ 64, 256 ] ) )

        # Stage 2
        self.model.add( convBlock( filters = [ 128, 512 ], s = 2 ) )
        for _ in range(3):
            self.model.add( identityBlock( filters = [ 128, 512 ] ) )

        if not C10:
            self.model.add( tf.keras.layers.Dropout( 0.15 ) )
        
        # Stage 3
        self.model.add( convBlock( filters = [ 256, 1024 ], s = 2 ) )
        for _ in range(5):
            self.model.add( identityBlock( filters = [ 256, 1024 ] ) )

        if not C10:
            self.model.add( tf.keras.layers.Dropout( 0.2 ) )

        # Stage 4
        self.model.add( convBlock( filters = [ 512, 2048 ], s = 2 ) )
        for _ in range(2):
            self.model.add( identityBlock( filters = [ 512, 2048 ] ) )         

        # Pooling
        self.model.add( tf.keras.layers.AveragePooling2D( (2, 2), padding = 'same' ) )

        # Output
        self.model.add( tf.keras.layers.Flatten() )

        if not C10:
            self.model.add( tf.keras.layers.Dropout( 0.2 ) )

        self.model.add( tf.keras.layers.Dense( NUM_CLASSES, activation = 'softmax', kernel_initializer = 'he_normal' ) )


    def train( self, train, valid, trainSteps, validSteps ):
    def train( self, dataGenTrain, trainImg, trainLabel, validImg, validLabel ):

        self.lrdecay = tf.keras.callbacks.LearningRateScheduler(lrdecay) # Learning rate decay  
        self.model.compile( loss = tf.keras.losses.SparseCategoricalCrossentropy(), 
                            optimizer = OPTIMIZER, metrics = METRIC )
        trainSteps = trainImg.shape[0] / BATCH_SIZE
        validSteps = validImg.shape[0] / BATCH_SIZE

        self.history = self.model.fit( dataGenTrain.flow( trainImg, trainLabel, batch_size = BATCH_SIZE ), epochs = NUM_EPOCHS, 
                    steps_per_epoch = trainSteps, validation_steps = validSteps,
                    validation_data = ( validImg, validLabel ), callbacks = [ self.lrdecay ] )


    def test( self, testImg, testLabel ):

        self.model.evaluate( testImg, testLabel )

    
    # Plots accuracy over time
    def plotAccuracy( self ):

        plt.figure( figsize = ( FIG_WIDTH, FIG_HEIGHT ) )
        plt.plot( self.history.history[ PLOT_METRIC ] )
        plt.plot( self.history.history[ 'val_' + PLOT_METRIC] )
        plt.title( 'Model ' + PLOT_LABEL )
        plt.xlabel( 'Epochs' )
        plt.ylabel( PLOT_LABEL, rotation = 'horizontal', ha = 'right' )
        plt.legend( [ 'Train', 'Valid' ], loc = 'upper left' )
        plt.show()


def main():

    # Load data
    dictCIFARTrain = loadCIFAR( True, C10 )
    dictCIFARTest = loadCIFAR( False, C10 )

    # Get keys
    dictData = list( dictCIFARTrain.keys() )[0]
    dictLabels = list( dictCIFARTrain.keys() )[1]

    img = dictCIFARTrain[ dictData ]
    label = dictCIFARTrain[ dictLabels ]
    testImg = dictCIFARTest[ dictData ]
    testLabel = dictCIFARTest[ dictLabels ]

    # Reshapes each image into 32x32 and 3 channels ( RGB )
    img = np.reshape( img, [ -1, 32, 32, 3 ], order = 'F' )
    testImg = np.reshape( testImg, [ -1, 32, 32, 3 ], order = 'F' )

    # Mean-STD normalization from:
    # https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks
    mean = np.mean( img, axis=(0,1,2,3) )
    std = np.std( img, axis=(0,1,2,3) )
    img = np.array( (img-mean) / (std+1e-7) )
    testImg = np.array( (testImg-mean) / (std+1e-7) )

    # Train / Valid Split
    trainImg, validImg, trainLabel, validLabel = train_test_split( img, label, test_size = VALID_SIZE )

    # Rotate, normalize, and convert to tensor
    trainImg = tf.convert_to_tensor( trainImg, dtype=tf.float32 )
    trainLabel = tf.convert_to_tensor( trainLabel )
    validImg = tf.convert_to_tensor( validImg, dtype=tf.float32 )
    validLabel = tf.convert_to_tensor( validLabel )
    testImg = tf.convert_to_tensor( testImg, dtype=tf.float32 )
    testLabel = tf.convert_to_tensor( testLabel )

    # Check if image was loaded properly
    #testLoad( trainImg, trainLabel )

    # Data Augmentation
    dataGenTrain = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range = 0.1, horizontal_flip=True)
    dataGenTrain.fit( trainImg )
    
    model = imgClassMod()
    model.train( dataGenTrain, trainImg, trainLabel, validImg, validLabel )
    model.test( testImg, testLabel )
    model.plotAccuracy()
    

main()
