print "import numpy"
import numpy as np
#from numpy.lib.recfunctions import stack_arrays
#from root_numpy import root2array, root2rec
#import glob
print "import keras stuff"
from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras.utils import plot_model
from keras import regularizers, losses
from keras.layers import Dropout, add, BatchNormalization
from matplotlib import pyplot as plt
plt.ioff()

import glob, time, argparse
from build import makeNetwork, getHyperParameters

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir", default="")
    return parser.parse_args()


def splitXy(X,y):
    idx = range(y.size)
    X0 = np.array([X[i] for i in idx if y[i] == 0])
    X1 = np.array([X[i] for i in idx if y[i] == 1])
    return (X0,X1)


def main():
    X = np.load("X.npy")
    y = np.load("y.npy")

    ##setup the constants
    nodes, regularizer = getHyperParameters()

    ##setup the neutral net
    net = makeNetwork(X.shape[1], nodes, regularizer)
    net.load_weights("model.h5")

    X0, X1 = splitXy(X,y)

    yhat0 = net.predict(X0)
    yhat1 = net.predict(X1)

    fig, ax = plt.subplots()
    bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.hist(yhat0, bins=bins, histtype='step', label=r'$\hat{y}_{0}$', normed=True)
    plt.hist(yhat1, bins=bins, histtype='step', label=r'$\hat{y}_{1}$', normed=True)
    plt.legend()
    ax.set_xlim([0,1])
    ax.set_xlabel("NN Score")
    ax.set_ylabel("Arb. Units")
    plt.savefig("separation.png")
    plt.clf()



if __name__ == "__main__":
    main()
