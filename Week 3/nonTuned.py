import struct as st
import numpy as np
import tensorflow as tf
import kerastuner as kt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Constants
NUM_EPOCHS = 30
VALIDATION_PERCENTAGE = 0.2
DROP_RATE = 0.2

# Image specific parameters
IMG_DIMS = ( 28, 28, )
NUM_CLASSES = 10
HIDDEN_SIZE = [ 512, 512 ]  # Number of output neurons for each hidden layer

# Data processing inspired by:
# https://stackoverflow.com/questions/39969045/parsing-yann-lecuns-mnist-idx-file-format
def imgRead( fileName ):

    with open( fileName, 'rb') as f:
    
        # Go to beginning of file and start reading
        f.seek(0)
        _numMagic = st.unpack( '>I', f.read(4) )[0]                 # Read magic number
        numImg, numRows, numCols = st.unpack( '>III', f.read(12) )  # Read ints

        numTotalBytes = numImg * numRows * numCols * 1  # Each pixel is 1 byte

        # Read the remaining data in the file
        img = np.asarray( st.unpack( 
                    '>' + 'B'*numTotalBytes, f.read( numTotalBytes )  
                    ) ).reshape( ( numImg, numRows, numCols ) )

    return img


def labelRead( fileName ):

    with open( fileName, 'rb') as f:

        # Go to beginning of file and start reading
        f.seek(0)
        _numMagic = st.unpack( '>I', f.read(4) )[0]     # Read magic number
        numImg = st.unpack( '>I', f.read(4) )[0]        # Read number of examples

        # Read the remaining data in the file
        labels = st.unpack( '>' + 'B'*numImg, f.read( numImg ) )

    return labels


# Module containing Image Classification model
class imgClassMod( tf.Module ):

    def __init__( self ):

        self.model = tf.keras.models.Sequential()

        # Input Layer
        self.model.add( tf.keras.layers.Flatten( input_shape = IMG_DIMS ) )

        # Hidden Layers
        for numNeurons in HIDDEN_SIZE:
            self.model.add( tf.keras.layers.Dense( numNeurons, activation = 'relu', kernel_regularizer = 'l2' ) )
            self.model.add( tf.keras.layers.Dropout( rate = DROP_RATE, trainable = False ) )

        # Output Layer
        self.model.add( tf.keras.layers.Dense( NUM_CLASSES, activation = 'softmax' ) )
        

    def train( self, trainImg, trainLabel ):

        self.model.compile( loss = "sparse_categorical_crossentropy", 
                            optimizer = "sgd", metrics = "accuracy" )
        self.history = self.model.fit( trainImg, trainLabel, epochs = NUM_EPOCHS, 
                    validation_split = VALIDATION_PERCENTAGE )


    def test( self, testImg, testLabel ):

        self.model.evaluate( testImg, testLabel )

    
    def plotAccuracy( self ):

        # Plots accuracy over time
        plt.figure( figsize = (10,5) )
        plt.plot( self.history.history['accuracy'] )
        plt.plot( self.history.history['val_accuracy'] )
        plt.title( 'Model Accuracy' )
        plt.xlabel( 'Epochs' )
        plt.ylabel( 'Accuracy', rotation = 'horizontal', ha = 'right' )
        plt.legend( [ 'Train', 'Valid' ], loc = 'upper left' )
        plt.show()



def main():

    # Load data
    trainImg = imgRead( 'trainImages.idx3-ubyte' )
    trainLabel = labelRead( 'trainLabels.idx1-ubyte' )
    testImg = imgRead( 'testImages.idx3-ubyte' )
    testLabel = labelRead( 'testLabels.idx1-ubyte' )

    # Normalize and convert to tensor
    trainImg = tf.convert_to_tensor( trainImg / 255, dtype=tf.float32 )
    trainLabel = tf.convert_to_tensor( trainLabel )
    testImg = tf.convert_to_tensor( testImg / 255, dtype=tf.float32 )
    testLabel = tf.convert_to_tensor( testLabel )

    # Test if images were properly loaded
    plt.figure()
    plt.imshow( trainImg[0,:,:], cmap = 'gray' )
    plt.show()

    # Test if labels were properly loaded
    print( trainLabel[0] )

    classifier = imgClassMod()

    # Train and Test
    print( "\nStarted Training\n" )
    classifier.train( trainImg, trainLabel )
    print( "\nFinished Training\n" )
    classifier.test( testImg, testLabel )

    # Display accuracy
    classifier.plotAccuracy()


main()
