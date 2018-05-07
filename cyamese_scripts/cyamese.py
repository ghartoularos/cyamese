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

parser.add_argument('f', metavar='pathtofcs', type=str,
                    help='location of fcs files with metadata pickle')

parser.add_argument('-s', metavar='subset', type=int, default=-1,
                help='random subset of files to use for training ' + 
                '(default: use all)')

parser.add_argument('-e', metavar='epochs', type=int, default=100,
                    help='number of epochs to train (default=100) ')

parser.add_argument('-nc', metavar='numcells', type=int, default=1500,
                help='number of cells per "image" of individual ' + 
                '(default=1500)')

parser.add_argument('-tr', metavar='trainloops', type=int, default=300,
                help='number of image pairs per subject for training ' + 
                '(defaults=300)')

parser.add_argument('-te', metavar='testloops', type=int, default=300,
                help='number of image pairs per subject for testing ' + 
                '(defaults=300)')

parser.add_argument('--lb', action='store_false', default=True,
                    help='option to print loading bar (default: on)')



args = parser.parse_args()  # Parse arguments

pathtofcs = args.f
rand_subset = args.s
epochs = args.e
numcells = args.nc
trainloops = args.tr
testloops = args.te
loadbar = args.lb

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TODO:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sprinkle exceptions throughout all scripts to catch inputs if they are
not of the right type.

Figure out why I'm getting the:
UserWarning: text in segment does not start and end with delimiter
warning from fcm when I use all of the data; occurs towards the beginning

Make a separate script for just acquiring the data that uses the API.

Eventually, make the testing be only from a new subject, see if it can be 
used to recognize new patients off the bat. There will be two testing sets.
One asks can we recognize the same subject from the training. The other, 
cooler more important question: can we recognize a new patient that 
wasn't even trained on at all?


'''

# Imports
import itertools as it
import os
print("Welcome to CYAMESE: training a model that learns an individual's\
 cytometry fingerprint. Beginning program.")

import tensorflow as tf

from metautils import makemeta
from markers import setmarkers
from createpairs import generatedicts, pairandsplit, testpairs
from polish import stacker, normer
from mlutils import getmodel, trainCNN

###########################################################################
'''
Quick input check
'''
if pathtofcs[-1] != '/': # add the forward slash if it's not there
    pathtofcs += '/'
if not os.path.exists(pathtofcs): # Confirm that meta path exists
    print('Supplied pathtofcs does not exist. Try again.')
    raise SystemExit
'''
The training regimen is only using data from Mark Davis' study
from Stanford. The chosen studies are shown below. Despite the 
numbering, they are in chronological order. The dictionary
"studdict" is used to index the studies according to chronology.
The "split" variable splits the studies into training and testing.
The training data is based *only* on the first two years, while the
testing data uses all three years.

Split is a tuple of exactly two tuples that dictates what data will 
be used for training and testing. The first tuple is training and the
second is testing. Training tuple will compare two different studies
to eachother to create positive pairs (the same subject in SDY311 and
SDY112) and negative pairs (different subjects in the same or different 
studies).
'''

studies = ['SDY311', 'SDY112', 'SDY315']
studdict = dict(zip(range(3),studies))
split = (('SDY311', 'SDY112'), ('SDY315',))

'''
The "TIME" channel in the flow data is an irrelevant parameter, not
so much a phenotype of the cells but only useful for troubleshooting
experimentation. It will be ignored.
'''

ignorechan = ['TIME'] # images will not contain this channel

'''
makemeta takes the metadata file (a pickled pandas dataframe) located
in the directory created from the apidownload script, and unpickles it.
If a random number of subjects has been specified, it only extracts the
data from those subjects.
'''
print("Making meta data file.")
meta, numsubs = makemeta(pathtofcs, studies, studdict, rand_subset)

files = list(meta['filename'])

'''
setmarkers makes a set of unique markers common to all the flow cytometry
data files. It will ignore any channels fed in by ignorechan and also 
check that there are at least 20 markers to learn from.
'''
print("Extracting out common markers from flow data.")
markerset = setmarkers(pathtofcs, files, ignorechan=ignorechan)

'''
generatedicts generates dictionaries that have subject/study pairs as
keys and a numpy array of the cells x markers data as the value. It ensures
that the flow data have the markers all in the same order. It also splits
the data into train and test such that, although data from the same
subject/study might be used in both training and testing, that the same data
from an individual cell is not reused.
'''
print("Generating dictionaries with the data.")
alltrain, alltest = generatedicts(studies, numsubs, files, pathtofcs, 
                                  markerset, split, loadbar)

'''
testpairs just runs a sanity check to make sure the images that are being 
generated are truly positive and negative pairs. The above functions can be 
commented out and the line below commented in to just run a sanity check.
'''
# alltrain, alltest = testpairs()

'''
pairandsplit is the main workhorse of the data formatting. It takes random 
images from a person's flow data and pairs it with either the same person's 
flow data from a different study (positive pair) or a different person's flow 
data from the same or different year (negative pair). Depending on how many
loops are used, it might need to reuse cells; it does this in such a way that
it never reuses the same cell before using all others first. The most 
important part is that it doesnt not bias the training towards any one 
individual or class: there is a roughly equal representation of positive and
negative pairs, of individuals, of studies, and of cells.
'''
print("Making pairs of images.")
trainx, trainy, testx, testy = pairandsplit(alltrain, alltest, numcells, 
                                            trainloops, testloops, split,
                                            loadbar)

'''
stacker simply takes the output and "stacks" the images along an axis to 
get them in the correct ndarray representation for learning.
'''
print("Stacking data to conform to multi-dimensional array input for \
training. This may take a minute.")
train_x, train_y, test_x, test_y = stacker(trainx, trainy, testx, testy)

'''
normer Z-normalizes the data before input into the trainer.
'''
print("Normalizing data.")
train_x, test_x = normer(train_x, test_x)

'''
getmodel gets the base network for training.
'''
input_shape = train_x.shape[2:]
width = len(markerset)
model = getmodel(input_shape, width)

'''
trainCNN trains the convolutional neural network.
'''
print("~~~~~~Begin Training~~~~~~")
trainCNN(train_x, train_y, model, epochs, test_x, test_y, loadbar)
