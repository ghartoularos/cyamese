import numpy as np
from tqdm import tqdm

def normalized(a, axis=0):
    axisALL = range(0,len(a.shape))
    axisC = list()
    axisC = [i for i in axisALL if i!=axis]  #  single-liner without tqdm
    axisC = tuple(axisC)
    a_mean = a.mean(axis=axisC, keepdims=False)
    a_std = a.std(axis=axisC, keepdims=False)
    a_std[a_std==0]=1
    new_shape = np.repeat(1,len(a.shape))
    new_shape[axis]= a.shape[axis]
    a_std = a_std.reshape(new_shape)
    a_mean = a_std.reshape(new_shape)
    a = (a - a_mean)/a_std
    return a, a_mean, a_std

def stacker(trainx, trainy, testx, testy):
    train_x = np.stack([item for sublist in trainx for item in sublist]) # list comprehension flattens
    print('Training X Data: Stacked!')
    train_y = np.stack([item for sublist in trainy for item in sublist])
    print('Training Y Data: Stacked!')
    test_x = np.stack([item for sublist in testx for item in sublist])
    print('Testing X Data: Stacked!')
    test_y = np.stack([item for sublist in testy for item in sublist])
    print('Testing Y Data: Stacked!')
    return train_x, train_y, test_x, test_y

def normer(train_x, test_x):
    print("Normalizing training data. This may take a minute.")
    train_x, train_mean, train_std = normalized(train_x,axis = 3)
    print("Training data normalized!")
    print("Normalizing testing data. This may take a minute.")
    test_x = (test_x - train_mean)/train_std
    print("Testing data normalized!")
    return train_x, test_x
