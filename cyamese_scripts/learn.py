import argparse

### Setup argument parser ###

parser = argparse.ArgumentParser(description="""
Welcome to CYAMESE: training a model that learns an individual's cytometry
fingerprint. This is a program made as a rotation project by George
Hartoularos in Atul Butte's lab at University of California, San Francisco.

The input is a series of studies done at Stanford in Mark Davis' group in 
which CYTOF data was collected from the peripheral blood cells of patients 
before being vaccinated for influenza. Because it is pre-vaccination, this
represents a patient in their "healthy" state. The idea is that by treating
these flow cytometry data as "images" of a person, we can learn what that
individual looks like simply based on their cytometry profile.

Right now the program only takes in those studies as input, but will 
eventually generalize to any flow/gene expression data. The program is still
being developed and is not ready to make accurate predictions.

This is made for python2 and uses the Keras machine learning framework 
with TensorFlow as backend.

""", formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('f', metavar='pathtopkl', type=str,
                    help='folder containing pickled ML data')

parser.add_argument('-e', metavar='epochs', type=int, default=100,
                    help='number of epochs to train (default=100) ')

parser.add_argument('--lb', action='store_false', default=True,
                    help='option to print loading bar (default: on)')

parser.add_argument('--gpu', action='store_true', default=False,
                    help='option if running on GPU (default: off)')

args = parser.parse_args()  # Parse arguments

pathtopkl = args.f
epochs = args.e
loadbar = args.lb
gpu_switch = args.gpu

# from polish import stacker, normer
from mlutils import getmodel, trainCNN
import numpy as np
import os
'''
testpairs just runs a sanity check to make sure the images that are being 
generated are truly positive and negative pairs. The above functions can be 
commented out and the line below commented in to just run a sanity check.
'''
# alltrain, alltest = testpairs()


'''
normer Z-normalizes the data before input into the trainer.
'''
# print("Normalizing data.")
# train_x, test_x = normer(train_x, test_x)

'''
getmodel gets the base network for training.
'''
examplearray = np.load(pathtopkl + '/' + os.listdir(pathtopkl)[0]) # list npy files, take the first one
input_shape = examplearray.shape[2:] # shape should be the same for all
width = input_shape[1] # this is the number of markers we're training from
model = getmodel(input_shape, width)

'''
trainCNN trains the convolutional neural network.
'''
print("~~~~~~Begin Training~~~~~~")
trainCNN(pathtopkl, model, epochs, loadbar, gpu_switch)