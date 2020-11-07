import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
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
NUM_EPOCHS = 200
BATCH_SIZE = 16
LEARN_RATE = 0.001
IMG_SIZE = 100

# Training
NUM_CLASSES = 80
OPTIMIZER = tf.keras.optimizers.SGD( learning_rate = LEARN_RATE, momentum = 0.9 )
REGULARIZER = tf.keras.regularizers.l2( 0.01 )
LOSS = tf.keras.losses.CategoricalCrossentropy( from_logits = False )
METRIC = tf.keras.metrics.CategoricalAccuracy()

checkpoint_path = os.getcwd() + "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# Create a callback that saves the model's weights from:
# https://www.tensorflow.org/tutorials/keras/save_and_load
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=20)  # Saves every 10 epochs


# Decreases learning rate at specific epochs
def lrdecay(epoch):

    lr = LEARN_RATE
    if epoch > 150:
        lr *= 1e-3
    elif epoch > 100:
        lr *= 1e-2
    elif epoch > 50:
        lr *= 1e-1

    return lr


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


if __name__ == "__main__":
    
    # Load data
    a=1