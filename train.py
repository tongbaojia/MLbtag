import argparse, copy, glob, os, sys, time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.rcParams.update({'font.size': 16})
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Input, Activation, Dropout, add
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping



def flatten(column):
    '''
    Args:
    -----
    : a column of a pandas df whose entries are lists (or regular entries -- in which case nothing is done)
                e.g.: my_df['some_variable'] 

    Returns:
    --------    
        flattened out version of the column. 

        For example, it will turn:
        [1791, 2719, 1891]
        [1717, 1, 0, 171, 9181, 537, 12]
        [82, 11]
        ...
        into:
        1791, 2719, 1891, 1717, 1, 0, 171, 9181, 537, 12, 82, 11, ...
    '''
    try:
        return np.array([v for e in column for v in e])
    except (TypeError, ValueError):
        return column

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir", default="")
    return parser.parse_args()

def transform_data():
    print "Transforming and preparing training data!!"
    df = pd.read_hdf('testl_pd.h5', 'data')
    ##pick different variables

    jf_df = df[[key for key in df.keys() if (key.startswith('jet_jf') and '_vtx_' not in key)]]

    jf_df_flat = pd.DataFrame({k: flatten(c) for k, c in jf_df.iteritems()})
    #jf_df_flat = jf_df_flat[jf_df_flat['jet_sv1_efc'] > -99]

    flavor = flatten(df['jet_LabDr_HadF'])
    flavor_pids = np.unique(flavor)

    X = jf_df_flat.as_matrix() # I think this is the same as jf_df_flat.values
    le = LabelEncoder()
    y = le.fit_transform(flavor)
    #ix = range(X.shape[0]) # array of indices, just to keep track of them for safety reasons and future checks
    #X_train, X_test, y_train, y_test, ix_train, ix_test = train_test_split(X, y, ix, train_size=0.8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    np.save("X_train", X_train)
    np.save("X_test", X_test)
    np.save("y_train", y_train)
    np.save("y_test", y_test)

def main():
    '''here is where everything is setup, basic options of plots and direcotries, fits'''
    start_time = time.time()
    ops = options()

    ##load the data
    #transform_data()

    ##or just load the matricies
    print "load the npy file directly"
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print X_train.size

    ##now start building the neuro net
    x = Input(shape=(22, ))
    # all Keras Ops look like z = f(z) (like functional programming)
    skip = Dense(30)(x)
    h = Dropout(0.1)(skip)
    h = Activation('relu')(h)

    h = Dense(30)(h)
    skip = add([h, skip])
    h = Dropout(0.1)(skip)
    h = Activation('relu')(h)

    h = Dense(30)(h)
    skip = add([h, skip])
    h = Dropout(0.1)(skip)
    h = Activation('relu')(h)

    # our output is a single number, the house price.
    y = Dense(1)(h)

    # A model is a conta
    net = Model(x, y)

    net.compile(optimizer='adam', loss='mse')
    callbacks = [
        # if we don't have a decrease of the loss for 10 epochs, terminate training.
        EarlyStopping(verbose=True, patience=10, monitor='val_loss'), 
        # Always make sure that we're saving the model weights with the best val loss.
        ModelCheckpoint('model.h5', monitor='val_loss', verbose=True, save_best_only=True)]
    history = net.fit(X_train, y_train, validation_split=0.2, epochs=30, verbose=2, callbacks=callbacks)

    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['loss'], label='loss')
    plt.legend()
    plt.savefig("loss.png")
    plt.clf()


    y_score = net.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=2)
    roc_auc = auc(fpr, tpr)
    # Compute micro-average ROC curve and ROC area
    #print fpr, tpr
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("roc.png")
    plt.clf()


    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
