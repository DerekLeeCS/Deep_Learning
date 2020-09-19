import struct as st
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


NUM_EPOCHS = 10
LEARN_RATE = 0.1
MOMENTUM = 0
DROP_RATE = 0.2
IMG_DIMS = ( 28, 28 )
NUM_CLASSES = 10

HIDDEN_SIZE = [ 200, 100, 50 ]

# Data processing inspired by:
# https://stackoverflow.com/questions/39969045/parsing-yann-lecuns-mnist-idx-file-format
def imgRead( fileName ):

    with open( fileName, 'rb') as f:
    
        # Go to beginning of file and start reading
        f.seek(0)
        _numMagic = st.unpack( '>4B', f.read(4) )[0]                # Read 4 chars
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
        _numMagic = st.unpack( '>4B', f.read(4) )[0]    # Read 4 chars
        numImg = st.unpack( '>I', f.read(4) )[0]        # Read number of examples

        labels = st.unpack( '>' + 'B'*numImg, f.read( numImg ) )

    return labels


# Class containing Image Classification model
class imgModel( tf.keras.Model ):

    def __init__( self ):

        super().__init__()
        self.inputLayer = tf.keras.layers.Flatten( input_shape = IMG_DIMS )
        self.hiddenLayer0 = tf.keras.layers.Dense( HIDDEN_SIZE[0], activation = 'elu')
        self.hiddenLayer1 = tf.keras.layers.Dense( HIDDEN_SIZE[1], activation = 'elu' )
        self.hiddenLayer2 = tf.keras.layers.Dense( HIDDEN_SIZE[2], activation = 'elu' )
        self.outputLayer = tf.keras.layers.Dense( NUM_CLASSES, activation = 'softmax' )
        self.dropLayer = tf.keras.layers.Dropout( rate = DROP_RATE, input_shape = IMG_DIMS )

    def call( self, img, trainBool = None ):

        inLayer = self.inputLayer( img )
        inLayer = self.dropLayer( inLayer, training = trainBool )
        hidden0 = self.hiddenLayer0( inLayer )
        hidden0 = self.dropLayer( hidden0, training = trainBool )
        hidden1 = self.hiddenLayer1( hidden0 )
        hidden1 = self.dropLayer( hidden1, training = trainBool )
        hidden2 = self.hiddenLayer2( hidden1 )
        hidden2 = self.dropLayer( hidden2, training = trainBool )
        outLayer = self.outputLayer( hidden2 )
        
        return outLayer



# Module containing Logistic Classification model
class imgClassMod( tf.Module ):

    def __init__( self ):

        self.model = imgModel()
    
    def train( self, trainImg, trainLabel ):

        opt = tf.keras.optimizers.SGD( LEARN_RATE, MOMENTUM )
        self.model.compile( optimizer = opt, metrics = "CategoricalAccuracy" )
        self.model.fit( trainImg, trainLabel, epochs = NUM_EPOCHS, validation_split = 0.3 )

    def test( self, testImg, testLabel ):

        self.model.evaluate( testImg, testLabel )


def main():

    # Load data
    trainImg = imgRead( 'trainImages.idx3-ubyte' )
    trainLabel = labelRead( 'trainLabels.idx1-ubyte' )
    #trainImg, validImg, trainLabel, validLabel = train_test_split( img, label, test_size = 0.3 )
    testImg = imgRead( 'testImages.idx3-ubyte' )
    testLabel = labelRead( 'testLabels.idx1-ubyte' )

    trainImg = tf.convert_to_tensor( trainImg, dtype=tf.float32 )
    trainLabel = tf.convert_to_tensor( trainLabel )
    #validImg = tf.convert_to_tensor( validImg, dtype=tf.float32 )
    #validLabel = tf.convert_to_tensor( validLabel )
    testImg = tf.convert_to_tensor( testImg, dtype=tf.float32 )
    testLabel = tf.convert_to_tensor( testLabel )

    # One hot encoding
    trainOneHot = tf.one_hot( trainLabel, NUM_CLASSES )
    #validOneHot = tf.one_hot( validLabel, NUM_CLASSES )
    testOneHot = tf.one_hot( testLabel, NUM_CLASSES )

    print( "Train: ", len(trainImg) )
    print( "Train: ", len(trainLabel) )
    #print( "Valid: ", len(validImg) )
    #print( "Valid: ", len(validLabel) )
    print( "Test: ", len(testImg) )
    print( "Test: ", len(testLabel) )
    print(testOneHot[0])
    # Test if images were properly loaded
    #plt.imshow(trainImg[1,:,:], cmap='gray')
    #plt.show()
    # Test if labels were properly loaded
    #print( trainImg[1:].shape )
    classifier = imgClassMod()
    classifier.train( trainImg, trainOneHot )
    print( "Finished Training\n" )
    classifier.model.evaluate( testImg, testOneHot )


    



main()


