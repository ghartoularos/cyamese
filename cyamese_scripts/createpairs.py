import numpy as np
import pandas as pd
import fcm
import itertools as it
from tqdm import tqdm
from markers import cleanmarkers
import random

def loading(iterable, loadbar):
    if loadbar:
        def feediter():
            for i in tqdm(iterable):
                yield i
    else:
        def feediter():
            for i in iterable:
                yield i
    return feediter()

def testpairs():
    '''
    This can be used to test the "pair and split" code below
    to make sure that it's generating pairs correctly. This represents
    3 subject in  3 studies (years), with 1 marker and 20,000 cells each,
    in which half the data of the first two years are used for training
    and the other half of those years (and the third year) are used for
    testing. 
    '''
    alltrain = {(0,'SDY311'): np.full((10000,1),-1),
                (0,'SDY112'): np.full((10000,1),-2),
                (1,'SDY311'): np.full((10000,1),-4),
                (1,'SDY112'): np.full((10000,1),-5),
                (2,'SDY311'): np.full((10000,1),-7),
                (2,'SDY112'): np.full((10000,1),-8)}

    alltest = {(0,'SDY311'): np.full((10000,1),1),
                (0,'SDY112'): np.full((10000,1),2),
                (0,'SDY315'): np.full((20000,1),3),
                (1,'SDY311'): np.full((10000,1),4),
                (1,'SDY112'): np.full((10000,1),5),
                (1,'SDY315'): np.full((20000,1),6),
                (2,'SDY311'): np.full((10000,1),7),
                (2,'SDY112'): np.full((10000,1),8),
                (2,'SDY315'): np.full((20000,1),9)}
    return alltrain, alltest

def generatedicts(studies, numsubs, files, pathtofcs, 
                  markerset, split, loadbar):
    alldata = {}
    prod = it.product(range(numsubs),studies) # (study, subject): array
    print('Generating dictionaries "alldata", "alltrain", and "alltest".')
    feedfiles = loading(files, loadbar)
    for filename in feedfiles: # make a dictionary with a (study, subject) code as the keys and the fcs array as the value
        data = fcm.loadFCS(pathtofcs+filename)
        df = pd.DataFrame(np.array(data),columns=cleanmarkers(data.channels))
        colset = set(df.columns)
        if colset != markerset:
            df = df.drop(columns=list(colset - markerset)) # colset should never be smaller than markerset
        df = df[list(markerset)] # to put them all in the same order
        df = df.sample(frac=1).reset_index(drop=True) 
        alldata[next(prod)] = df.as_matrix()

    alltrain = {}
    alltest = {}
    feedkeys = loading(alldata.keys(), loadbar)
    for studsub in feedkeys: # split all the data into train and test arrays
        if studsub[1] in split[0]: # the 1 corresponds to order of multiplicands in cartesian product above (prod=it.product())
            length = len(alldata[studsub])
            splittrain = np.random.choice(range(length),size=length/2,
                                          replace=False)
            splittest = np.setdiff1d(np.arange(length),splittrain)
            alltrain[studsub] = alldata[studsub][splittrain,:]
            alltest[studsub] = alldata[studsub][splittest,:]
        else:
            alltest[studsub] = alldata[studsub]
    return alltrain, alltest

def pairandsplit(alltrain, alltest, numcells, 
                 trainloops, testloops, split, loadbar):
    # loops1 = 500 # the number of times you should loop through all *other* combinations of subjects and studies
    ######################
    def get_other(item,itype='stud',trte='train'):

        if trte == 'train':
            others = set(alltrain.keys())
        elif trte == 'test':
            others = set(alltest.keys())
        else:
            print('Input not interpretable.')
            raise SystemExit
        if itype == 'stud':
            others -= set([i for i in others if \
                (item[1] ==  i[1] or item[0] != i[0])])
        if itype == 'sub':
            others -= set([i for i in others if item[0] ==  i[0]])
    #     for _ in range(loops1):
        while True:
            others = list(others)
            random.shuffle(others)
            for j in others:
                yield j
    ######################
    # loops2 = 500 # the number of times every studsub's array will be cycled through, # of times each cell will be used
    def get_sample(array):
        while True:
    #     for i in range(loops2):
            a = array
            overflow = len(array)%numcells
            b = np.random.choice(range(len(a)),size=overflow,replace=False)
            a = np.delete(a,b,axis=0)
            length = len(a)
            for i in range(length/numcells):
                b = np.random.choice(range(len(a)),size=numcells,replace=False)
                image = a[b,:]
                yield image.reshape(image.shape[0],image.shape[1],1)
                a = np.delete(a,b,axis=0) 
    
    ######################
    '''
    At some point will want to keep track of if a file's cells are being used (looped through)
    more than once. Maybe make a dataframe (or two, train & test) with keys as rows/columns and 
    number of cells per file.
    '''
    trainx, trainy, testx, testy = list(), list(), list(), list() # initialize the lists
    
    gendict = {}
    for studsub in alltrain.keys(): # for each study*subject
        gendict[studsub] = get_sample(alltrain[studsub]) # make a dictionary of generators that generates n-cell images
    pairslistx = trainx # use the training data
    pairslisty = trainy
    print("Training set: making random pairs of images from individuals'" + 
        " flow cytometry data for training.")
    tot = len(alltrain)
    count = 0
    for i in sorted(alltrain.keys(), key=lambda tup: tup[0]): # go in subject order
        count += 1
        print("Subject, study: %s,  %s | Pair (%d/%d) " % (i[0], i[1], count, tot))
        image = gendict[i] # get the correct "image" generator
        otherstud = get_other(i,'stud','train') # get the *key* for the **same subject** but **different study**
        othersub = get_other(i,'sub','train') # get the *key*  for the **same study** but **different subject**
        feedtrloops = loading(range(trainloops), loadbar)
        for _ in feedtrloops:
            pospair = np.stack([next(image),next(gendict[next(otherstud)])]) # get two images from the same person, a positive
            negpair = np.stack([next(image),next(gendict[next(othersub)])]) # get two images from different people,  a negative
            pairslistx.append([pospair,negpair]) # add those to the list as a pair
            pairslisty.append([1,0])  # add positive (1) and negative (0)  

    gendict = {}         
    for studsub in alltest.keys(): # for each study*subject
        gendict[studsub] = get_sample(alltest[studsub]) # make a dictionary of generators that generates n-cell images
    pairslistx = testx # use the testing data
    pairslisty = testy
    onlytestkeys =  [i for i in alltest.keys() if i[1] in split[1]]
    print("Testing set: making random pairs of images from individuals'" + 
        " flow cytometry data for testing.")
    tot = len(onlytestkeys)
    count = 0
    for i in sorted(onlytestkeys, key=lambda tup: tup[0]): # go in subject order
        count += 1
        print("Subject, study: %s,  %s | Pair (%d/%d) " % (i[0], i[1], count, tot))
        image = gendict[i] # get the correct "image" generator
        otherstud = get_other(i,'stud','test') # get the *key* for the **same subject** but **different study**
        othersub = get_other(i,'sub','test') # get the *key*  for the **same study** but **different subject**
        feedteloops = loading(range(testloops), loadbar)
        for _ in feedteloops:
            pospair = np.stack([next(image),next(gendict[next(otherstud)])]) # get two images from the same person, a positive
            negpair = np.stack([next(image),next(gendict[next(othersub)])]) # get two images from different people,  a negative
            pairslistx.append([pospair,negpair]) # add those to the list as a pair
            pairslisty.append([1,0])  # add positive (1) and negative (0)  
    return trainx, trainy, testx, testy

