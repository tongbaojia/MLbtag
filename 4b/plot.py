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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

pyplot.ioff()

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


def interpolate(x,X,Y): #return interpolated y for desired x given the scatter plot values X,Y
    if x<X[ 0]: return Y[ 0]
    if x>X[-1]: return Y[-1]

    #find X on either side of x
    bin = 0
    for i in range(len(X)):
        if x<X[i]:
            bin = i-1
            break
    
    #return interpolated value
    m = (Y[bin+1]-Y[bin])/(X[bin+1]-X[bin])
    y = m*(x-X[bin])+Y[bin]
    return y

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

    y = [yhat1,yhat0]

    #fig, ax = pyplot.subplots()
    fig, (ax1, ax2) = pyplot.subplots(nrows=2)
    bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ns, _, patches = ax1.hist(y, bins=bins, histtype='step', label=[r'$\hat{y}_{1}$', r'$\hat{y}_{0}$'], normed=True)
    #ax1.hist(yhat1, bins=bins, histtype='step', label=r'$\hat{y}_{1}$', normed=True)
    #ax.set_xlim([0,1])
    ax2.set_xlabel("NN Score")
    ax1.set_ylabel("Arb. Units")

    binCenters = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
    ratio = [ns[0][i]/ns[1][i] if ns[1][i]>0 else 0 for i in range(len(bins)-1)]
    print ratio
    ax2.scatter(binCenters,ratio)
    #ax2.hist(ratio, bins=bins)

    weights = [interpolate(yhat0[i][0],binCenters,ratio) for i in range(yhat0.size)]
    #print weights
    ax1.hist(yhat0,bins=bins, histtype='step', label=r'$\hat{y}_{0} weighted$', normed=True, weights=weights)

    ax1.legend()

    fig.savefig("separation.png")
    fig.clf()




if __name__ == "__main__":
    main()
