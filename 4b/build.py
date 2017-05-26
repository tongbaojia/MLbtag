print "import numpy"
import numpy as np
#from numpy.lib.recfunctions import stack_arrays
#from root_numpy import root2array, root2rec
#import glob
print "import keras stuff"
from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers, losses
from keras.layers import Dropout, add, BatchNormalization
from matplotlib import pyplot as plt
import glob, time, argparse


filepath = "hist-MiniNtuple.h5"

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir", default="")
    return parser.parse_args()


def makeNetwork(inputwidth, nodes, regularizer):
    # we define the input shape (i.e., how many input features) **without** the batch size
    x = Input(shape=(inputwidth, ))

    # all Keras Ops look like z = f(z) (like functional programming)
    h = Dense(nodes,kernel_regularizer=regularizer)(x)
    h = Activation('relu')(h)
    h = BatchNormalization()(h)

    h = Dense(nodes,kernel_regularizer=regularizer)(h)
    #h = Dropout(0.1)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    h = Dense(nodes,kernel_regularizer=regularizer)(h)
    h = Activation('relu')(h)
    h = BatchNormalization()(h)

    # our output is a single number, the house price.
    y = Dense(1)(h)
    y = Activation('sigmoid')(y)

    net = Model(x, y)

    net.compile(optimizer='sgd', loss=losses.binary_crossentropy)
    return net

def getHyperParameters():
    nodes=30
    alpha=0.01
    regularizer=regularizers.l2(alpha)
    return (nodes, regularizer)

def main():
    '''here is where everything is setup, basic options of plots and direcotries, fits'''
    start_time = time.time()
    ops = options()

    ##or just load the matricies
    print "load the npy file directly"
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")

    ##setup the constants
    nodes, regularizer = getHyperParameters()

    ##setup the neutral net
    net = makeNetwork(X_train.shape[1], nodes, regularizer)

    ##setup the epoc
    callbacks = [
        # if we don't have a decrease of the loss for 10 epochs, terminate training.
        EarlyStopping(verbose=True, patience=10, monitor='val_loss'), 
        # Always make sure that we're saving the model weights with the best val loss.
        ModelCheckpoint('model.h5', monitor='val_loss', verbose=True, save_best_only=True)]


    ##train
    history = net.fit(X_train, y_train, validation_split=0.2, epochs=300, verbose=1, callbacks=callbacks, batch_size=128)
    # plt.plot(history.history['val_loss'], label='val_loss')
    # plt.plot(history.history['loss'], label='loss')
    # plt.legend()
    # plt.savefig("loss.png")
    # plt.clf()

    #plt.show()
    #raw_input()


    yhat_test = net.predict(X_test)
    yhat_test_round = np.array([1 if x>0.5 else 0 for x in yhat_test])
    correct_test = np.logical_not(np.logical_xor(y_test,yhat_test_round))

    yhat_train = net.predict(X_train)
    yhat_train_round = np.array([1 if x>0.5 else 0 for x in yhat_train])
    correct_train = np.logical_not(np.logical_xor(y_train,yhat_train_round))

    print "(train) Fraction Correct =",np.average(correct_train),"+/-",correct_train.size**-0.5
    print " (test) Fraction Correct =",np.average(correct_test),"+/-",correct_test.size**-0.5

    # _, bins, _ = plt.hist(y_test, histtype='step', label=r'$y_{\mathsf{true}}$')
    # plt.hist(yhat_test,   bins=bins,   histtype='step', label=r'$\hat{y}$')
    # plt.hist(correct_test,bins=bins, histtype='step', label=r'NXOR')
    # plt.legend()
    # plt.savefig("output.png")
    # plt.clf()

    ##make the roc curve
    #fpr, tpr, thresholds = roc_curve(y_test, yhat_test, pos_label=2)
    #roc_auc = auc(fpr, tpr)
    #print fpr, tpr
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.savefig("roc.png")
    # plt.clf()

    #plt.show()
    #raw_input()



    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
