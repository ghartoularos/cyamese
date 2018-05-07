import argparse
import getpass
import os
from tqdm import tqdm
import pandas as pd
import subprocess
from time import localtime, strftime
import numpy as np
import getpass
from pprint import pprint
pd.options.mode.chained_assignment = None

def makemeta(pathtofcs, studies, studdict, rand_subset):
    '''
    Makes a metadata Pandas dataframe from a csv or pickle
    '''

    if pathtofcs[-1] != '/': # add the forward slash if it's not there
        pathtofcs += '/'
    if not os.path.exists(pathtofcs): # Confirm that meta path exists
        print('Supplied pathtofcs does not exist. Try again.')
        raise SystemExit

    # get the metadata file
    metafiles = [f for f in os.listdir(pathtofcs) if 'meta' in f]

    if len(metafiles) == 0: # if there are no metadata files
        print('No metadata files found in supplied path. Try again.')
        raise SystemExit
    if len(metafiles) > 1: # if there's too many  metadata files
        print('Too many metadata files found in supplied path. Try again.')
        raise SystemExit
    pathtometa = pathtofcs + metafiles[0]

    try:
        meta = pd.read_pickle(pathtometa) # read the metadata file
    except:
        print('Metadata pickle not interpretable. Should be pickle.')
        raise SystemExit

    first = True # initialize for iteratively adding to a pandas dataframe
    numsubs = 0 # initialize number of subjects
    for i in set(meta['subject']): # for each subject
        minimeta = meta.loc[meta['subject'] == i] # extract only that subject

        # subjects must be present in every year 
        # if subject is present in every year, add to the dataframe
        if set(minimeta['study']) == set(studies): 
            numsubs += 1
            metaadd = pd.DataFrame(minimeta.loc[minimeta['study'].isin(studies)])
            if first == True:
                newmeta = metaadd
                first = False
            else:
                newmeta = pd.concat((newmeta,metaadd),ignore_index=True)

    if rand_subset != -1 and rand_subset < numsubs: 
        sample = np.random.choice(numsubs,rand_subset,False)*len(studies)
        sample = sorted([i + j for j in range(len(studies)) for i in sample])
        newmeta = newmeta.iloc[sample,:]
    elif rand_subset > numsubs:
        print('Metadata file has less subjects than number specified.' + \
            ' Returning maximum of %d subjects.' % (numsubs))

    localstuddict = {v: k for k, v in studdict.iteritems()}  #invert the dict

    studcolumn = newmeta['study']
    for i in range(len(newmeta)): # replace the study names with integers
        studcolumn.iloc[i] = localstuddict[studcolumn.iloc[i]]

    # sort the values by subject, then by study
    newmeta = newmeta.sort_values(by=['subject','study']).reset_index(drop=True)

    studcolumn = newmeta['study']

    for i in range(len(newmeta)):
        studcolumn.iloc[i] = studdict[studcolumn.iloc[i]]

    meta = newmeta

    return meta.reset_index(drop=True), numsubs