from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.models import Model
from keras.layers import (Input, Flatten, Dense, Dropout, Lambda,
                          Conv2D, AveragePooling2D, MaxPooling2D,
                          Conv1D, Activation)

def create_base_network(input_shape, width):
    input = Input(shape=input_shape)
    x=Conv2D(filters=32,#32, 
             kernel_size=(1, width),
             activation='relu')(input)
    x = AveragePooling2D(pool_size=(input_shape[0], 1))(x)
    x = Flatten()(x)
    x = Dense(20, activation='relu')(x)
    return Model(input, x)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def getmodel(input_shape, width):
    # network definition
    base_network = create_base_network(input_shape, width)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    return model

def trainCNN(train_x, train_y, model, epochs, test_x, test_y, loadbar):
    rms = Adam()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    if loadbar:
      verbose = 1
    else:
      verbose = 2
    modelFit = model.fit([train_x[:, 0], train_x[:, 1]], train_y,
                     batch_size=128,
                     epochs=epochs,
                     validation_data=([test_x[:, 0], test_x[:, 1]], test_y),
                     verbose=verbose)
    return