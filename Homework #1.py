import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from cycler import cycler

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


# Constants
N = 50
sigNoise = 0.1
M = 6   # Number of Gaussians 
numEpochs = 50

# Variables
eps = tf.random.normal( [N], 0, sigNoise )
x = tf.random.uniform( [N], 0, 1 )
y = np.sin( 2 * np.pi * x ) + eps

# Used for graphing true sinewave without noise
trueX = np.linspace( 0, 1, 500 )
trueY = np.sin( 2 * np.pi * trueX )

# Trainable Tensorflow variables
w = tf.Variable( tf.random.uniform( [M], 0, 1 ) )
mu =  tf.Variable( tf.random.uniform( [M], -0.5, 0.5 ) )
sig = tf.Variable( tf.random.uniform( [M], -0.5, 0.5 ) )
b = tf.Variable( tf.random.uniform( [1], -1, 1 ) )


# Loss function
@tf.function
def lossFunc( x, w, mu, sig, b, y ): 
    
    MSE = 0
    yHat = []

    for i in range( N ):
        yHat.append( estY( x[i], w, mu, sig, b ) )

    for i in range( N ):
        MSE += 0.5 * ( y[i] - yHat[i] )**2
        
    return MSE


# Calculates yHat
@tf.function
def estY( x, w, mu, sig, b ):

    yHat = b

    for j in range( M ):
        yHat = yHat + w[j] * gaussian( x, mu[j], sig[j] ) 
    
    return yHat


def gaussian( x, mu, sig ):

    return tf.math.exp( -( x - mu )**2 / sig**2 )




def main():

    print ( "Starting MSE:", lossFunc(x, w, mu, sig, b, y ).numpy() )

    # Stochastic Gradient Descent
    opt = tf.keras.optimizers.SGD( learning_rate=0.01 )
    loss = lambda: lossFunc(x, w, mu, sig, b, y )
    var_list = [ w, mu, sig, b ]

    for _ in range( numEpochs ):
        opt.minimize( loss, var_list )

    print ( "Final MSE:", lossFunc(x, w, mu, sig, b, y ).numpy() )

    '''
    #init = tf.global_variables_initializer()
    model = CustomModel()
    
    
    model.compile( optimizer=opt, metrics='MSE' )

    epochs = 100
    history = model.fit(

        x, myVars.get( 'w' ), myVars.get( 'mu' ), myVars.get( 'sig' ), myVars.get( 'b' ), y

        )

    predictions = model.predict(x).flatten()
    '''
    '''
    with tf.GradientTape() as tape:
        
        yHat = []
        for i in range( N ): 
            yHat.append( estY( x[i], myVars.get( 'w' ), myVars.get( 'mu' ), myVars.get( 'sig' ), myVars.get( 'b' ) ) )

        loss = tf.reduce_sum( lossFunc( y, yHat ) )
        grad = tape.gradient( loss, myVars )
        grad['b']
    
        
    yModel = np.zeros( N )
    for i in range( N ): 
        yModel[i] = estY( x[i], myVars.get( 'w' ), myVars.get( 'mu' ), myVars.get( 'sig' ), myVars.get( 'b' ) ) 
'''
    yPred = np.zeros( N )
    for i in range( N ):
        yPred[i] = estY( x[i], w, mu, sig, b ).numpy()

    plt.figure()
    plt.scatter( x, y )
    plt.scatter( x, yPred, color = 'r' )
    plt.plot( trueX, trueY )
    plt.xlabel('x') 
    plt.ylabel('y') 
    plt.title("Fit 1") 
    plt.show()

    
    plt.figure()
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'm', 'y', 'c']) ) )

    for j in range( M ):
        plt.plot( trueX, gaussian( trueX, mu[j], sig[j] ) )

    plt.show()


main()