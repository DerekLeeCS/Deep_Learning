import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from cycler import cycler

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# Constants
N = 50
sigNoise = 0.1
M = 10   # Number of Gaussians 
numEpochs = 200

# Variables
eps = tf.random.normal( [N,1], 0, sigNoise )
x = tf.random.uniform( [N,1], 0, 1 )
y = tf.sin( 2 * np.pi * x ) + eps

# Used for graphing true sinewave without noise
trueX = np.linspace( 0, 1, 500, dtype = 'float32' )
trueY = np.sin( 2 * np.pi * trueX )

# Trainable Tensorflow variables
w = tf.Variable( tf.random.uniform( [M], -0.5, 0.5 ) )
mu =  tf.Variable( tf.linspace( -0.1, 1.1, [M] ) )
sig = tf.Variable( 0.25 * tf.ones( shape = [M] ) )
b = tf.Variable( tf.random.uniform( [1], -0.5, 0.5 ) )


# Loss function
@tf.function
def lossFunc( x, y ): 
    
    yHat = tf.map_fn( estY, x ) # Applies estY elementwise to x
    MSE = tf.reduce_sum( 0.5 * ( y - yHat )**2 )
        
    return MSE


# Calculates yHat_i given x_i
@tf.function
def estY( x ):

    return tf.reduce_sum( w * gaussian( x, mu, sig ) ) + b


def gaussian( x, mu, sig ):

    return tf.math.exp( -( x - mu )**2 / sig**2 )


def main():

    print( "Starting MSE:", lossFunc( x, y ).numpy() )

    # Stochastic Gradient Descent
    # Idea from https://stackoverflow.com/questions/57759563/minimize-multivariate-function-in-tensorflow
    opt = tf.keras.optimizers.SGD()
    varList = [ w, mu, sig, b ]

    # Iterate through epochs
    for _ in range( numEpochs ):

        with tf.GradientTape() as tape:

            loss = lossFunc( x, y )
            print(loss)

        grads = tape.gradient( loss, varList )
        opt.apply_gradients( zip( grads, varList ) )

    print( "Final MSE:", lossFunc( x, y ).numpy() )

    # Calculate y from training data
    yPred = tf.map_fn( estY, trueX )

    # First plot
    plt.figure()
    plt.scatter( x, y, color = 'g' )                        # Noisy data
    plt.plot( trueX, yPred, color = 'r', linestyle = '--' ) # Regression manifold
    plt.plot( trueX, trueY, color = 'b' )                   # Noiseless sinewave
    plt.xlabel( 'x' ) 
    plt.ylabel( 'y', rotation = 0 ) 
    plt.title( "Fit 1" ) 
    plt.show()

    # Second plot
    plt.figure()
    plt.rc( 'axes', prop_cycle = ( cycler ('color', ['r', 'g', 'b', 'm', 'y', 'c']) ) )

    for j in range( M ):
        plt.plot( trueX, gaussian( trueX, mu[j], sig[j] ) )

    plt.xlabel( 'x' ) 
    plt.ylabel( 'y', rotation = 0 ) 
    plt.title( "Bases for Fit 1" ) 
    plt.show()


main()