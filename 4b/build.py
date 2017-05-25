print "import numpy"
import numpy as np
#from numpy.lib.recfunctions import stack_arrays
#from root_numpy import root2array, root2rec
print "import pandas"
import pandas as pd
#import glob


filepath = "../../patrick_data_test16L.h5"

df = pd.read_hdf(filepath, 'data')

#skim data to get only the two categories being studied and get equal statistics in each
is0b = df['category'] == 0
is2b = df['category'] == 2

df_0b = df[is0b]
df_2b = df[is2b]

frames = [df_0b[:df_2b.shape[0]],df_2b]
df_skim = pd.concat(frames)

print "skim"
print df_skim[:2]

print "import sklearn stuff"
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#make X matrix
X=df_skim[
    ['j0_pt', 'j0_eta', 'j0_phi', 'j0_m','j0_nTrk','j0_trk0_pt','j0_trk0_eta','j0_trk0_phi','j0_trk0_m',
     'j1_pt', 'j1_eta', 'j1_phi', 'j1_m','j1_nTrk','j1_trk0_pt','j1_trk0_eta','j1_trk0_phi','j1_trk0_m']].as_matrix()
scaler = StandardScaler()
X = scaler.fit_transform(X)

#make y vector
le = LabelEncoder()
y = le.fit_transform(df_skim['category'])


#split into training and testing samples
ix = range(X.shape[0]) # array of indices, just to keep track of them for safety reasons and future checks
X_train, X_test, y_train, y_test, ix_train, ix_test = train_test_split(X, y, ix, train_size=0.6)

print "train"
print y_train

print "test"
print y_test

#build a model
#from keras.datasets.boston_housing import load_data
print "import keras stuff"
from keras.layers import Dense, Input, Activation
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers, losses
from keras.layers import Dropout, add, BatchNormalization
from matplotlib import pyplot as plt

nodes=30
alpha=0.01
regularizer=regularizers.l2(alpha)
#regularizer=None

def makeNetwork():
    # we define the input shape (i.e., how many input features) **without** the batch size
    x = Input(shape=(X.shape[1], ))

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


net = makeNetwork()

callbacks = [
    # if we don't have a decrease of the loss for 10 epochs, terminate training.
    EarlyStopping(verbose=True, patience=10, monitor='val_loss'), 
    # Always make sure that we're saving the model weights with the best val loss.
    ModelCheckpoint('model.h5', monitor='val_loss', verbose=True, save_best_only=True)]

history = net.fit(X_train, y_train, validation_split=0.2, epochs=300, verbose=1, callbacks=callbacks, batch_size=128)

plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['loss'], label='loss')
plt.legend()

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

_, bins, _ = plt.hist(y_test, histtype='step', label=r'$y_{\mathsf{true}}$')
plt.hist(yhat, bins=bins, histtype='step', label=r'$\hat{y}$')
plt.hist(correct,bins=bins, histtype='step', label=r'NXOR')
plt.legend()

#plt.show()
#raw_input()
