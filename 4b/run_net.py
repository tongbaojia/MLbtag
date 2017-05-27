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
from sklearn.metrics import roc_curve, auc, roc_auc_score
import glob, time, argparse
import ROOT


filepath = "hist-MiniNtuple.h5"

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir", default="")
    return parser.parse_args()

def getHyperParameters():      
    nodes=30      
    alpha=0.01        
    regularizer=regularizers.l2(alpha)        
    return (nodes, regularizer)

def makeNetwork(inputwidth, nodes, regularizer):
    # we define the input shape (i.e., how many input features) **without** the batch size
    x = Input(shape=(inputwidth, ))

    # all Keras Ops look like z = f(z) (like functional programming)
    h = Dense(nodes, kernel_regularizer=regularizer)(x)
    h = Activation('relu')(h)
    h = BatchNormalization()(h)

    h = Dense(nodes, kernel_regularizer=regularizer)(h)
    h = Activation('relu')(h)
    h = BatchNormalization()(h) ##modify turn on of the node's output

    h = Dense(nodes,kernel_regularizer=regularizer)(h)
    h = Activation('relu')(h)
    h = BatchNormalization()(h)

    # our output is a single number, the house price.
    y = Dense(1)(h)
    y = Activation('sigmoid')(y)

    net = Model(input=x, output=y)

    net.compile(optimizer='adam', loss=losses.binary_crossentropy)
    return net

def main():
    '''here is where everything is setup, basic options of plots and direcotries, fits'''
    start_time = time.time()
    ops = options()

    ##or just load the matricies
    print "load the npy file directly"
    X_train = np.load("X_sig_train.npy")
    X_test  = np.load("X_sig_test.npy")
    y_train = np.load("y_sig_train.npy")
    y_test  = np.load("y_sig_test.npy")
    Z_train = np.load("Z_sig_train.npy")
    Z_test  = np.load("Z_sig_test.npy")

    ##get the list
    lst_0b = []
    lst_2b = []
    for k in range(y_train.shape[0]):
        if y_train[k] == 0:
            lst_0b.append(k)
        else:
            lst_2b.append(k)


    ##check the variables
    inputs = ['j0_trk0_pt','j0_trk1_pt','j1_trk0_pt','j1_trk1_pt','j0_trkdr','j1_trkdr','j0_nTrk','j1_nTrk','detaHH','mHH', 'j1_m', 'j0_m']

    # ##seperate the two training
    # X_0b = X_train[lst_0b, :]
    # X_2b = X_train[lst_2b, :]

    # for i in range(X_train.shape[1]):
    #     bins = np.linspace(-5, 5, 100)
    #     plt.hist(X_0b[:, i], bins, alpha=0.5, label=inputs[i] + "_0b")
    #     plt.hist(X_2b[:, i], bins, alpha=0.5, label=inputs[i] + "_2b")
    #     plt.legend()
    #     plt.savefig(inputs[i] + "_var" + ".png")
    #     plt.clf()

    ##setup the constants
    nodes, regularizer = getHyperParameters()
    #regularizer=None
    ##setup the neutral net

    # ##setup the epoc
    # callbacks = [
    #     # if we don't have a decrease of the loss for 10 epochs, terminate training.
    #     EarlyStopping(verbose=True, patience=10, monitor='val_loss'), 
    #     # Always make sure that we're saving the model weights with the best val loss.
    #     ModelCheckpoint('model.h5', monitor='val_loss', verbose=True, save_best_only=True)]


    # net = makeNetwork(X_train.shape[1], nodes, regularizer)


    # ##train
    # history = net.fit(X_train, y_train, validation_split=0.2, epochs=40, verbose=1, callbacks=callbacks, batch_size=128)
    # plt.plot(history.history['val_loss'], label='val_loss')
    # plt.plot(history.history['loss'], label='loss')
    # plt.legend()
    # plt.savefig("loss.png")
    # plt.clf()

    #plt.show()
    #raw_input()

    # nodes, regularizer = getHyperParameters()
    # net = makeNetwork(X_train.shape[1], nodes, regularizer)
    # net.load_weights("model.h5")

    # yhat_test = net.predict(X_test)
    # yhat_test_round = np.array([1 if x>0.5 else 0 for x in yhat_test])
    # correct_test = np.logical_not(np.logical_xor(y_test,yhat_test_round))

    # yhat_train = net.predict(X_train)
    # yhat_train_round = np.array([1 if x>0.5 else 0 for x in yhat_train])
    # correct_train = np.logical_not(np.logical_xor(y_train,yhat_train_round))

    # print "(train) Fraction Correct =",np.average(correct_train),"+/-",correct_train.size**-0.5
    # print " (test) Fraction Correct =",np.average(correct_test),"+/-",correct_test.size**-0.5

    # _, bins, _ = plt.hist(y_test, histtype='step', label=r'$y_{\mathsf{true}}$')
    # plt.hist(yhat_test,   bins=bins,   histtype='step', label=r'$\hat{y}$')
    # plt.hist(correct_test,bins=bins, histtype='step', label=r'NXOR')
    # plt.legend()
    # plt.savefig("output.png")
    # plt.clf()



    # net2 = makeNetwork(2, nodes, regularizer)
    # callbacks2 = [
    #     # if we don't have a decrease of the loss for 10 epochs, terminate training.
    #     EarlyStopping(verbose=True, patience=10, monitor='val_loss'), 
    #     # Always make sure that we're saving the model weights with the best val loss.
    #     ModelCheckpoint('model2.h5', monitor='val_loss', verbose=True, save_best_only=True)]
    # ##train
    # history2   = net2.fit(X_train[:, -2:], y_train, validation_split=0.2, epochs=100, verbose=1, callbacks=callbacks2, batch_size=128)
    ##or, load the neutral net
    nodes, regularizer = getHyperParameters()
    net2 = makeNetwork(2, nodes, regularizer)
    net2.load_weights("model2.h5")

    yhat_test2 = net2.predict(X_test[:, -2:])
    ##make the roc curve
    #print y_test, yhat_test
    #fpr, tpr, thresholds = roc_curve(y_test, yhat_test)
    fpr2, tpr2, thresholds2 = roc_curve(y_test, yhat_test2)

    ##cut based
    temp_lst = []
    for k in X_test:
        #print k
        if (abs(k[-3] - 1) < 0.5):
            temp_lst.append(1)
            # if np.sqrt(((k[-2])/0.3) ** 2  + ((k[-1])/0.3) ** 2) < 1.6:
            #     temp_lst.append(1)
            # else:
            #     temp_lst.append(0)
        else:
            temp_lst.append(0)
    yhat_test_cut = np.array(temp_lst)
    fpr3, tpr3, thresholds3 = roc_curve(y_test, yhat_test_cut)

    #print fpr, tpr, thresholds
    #roc_auc  = auc(fpr, tpr)
    roc_auc2 = auc(fpr2, tpr2)
    roc_auc3 = auc(fpr3, tpr3)
    #plt.plot(fpr, tpr, color='green',  lw=2, label='Full curve (area = %0.2f)' % roc_auc)
    plt.plot(fpr2, tpr2, color='darkorange', lw=2, label='Slice curve (area = %0.2f)' % roc_auc2)
    #plt.plot(fpr3, tpr3, color='red', lw=2, label='Cut curve (area = %0.2f)' % roc_auc3)
    plt.plot([0, 0], [1, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate--BKG')
    plt.ylabel('True Positive Rate--Sig')
    plt.title('ROC curves for Signal vs BKG')
    plt.legend(loc="lower right")
    plt.savefig("roc.png")
    plt.clf()
    #plt.show()
    #raw_input()


    ##check the outputs
    canv = ROOT.TCanvas("test", "test", 800, 800)
    grid_Xtest = []
    for i in np.arange(-5, 5, 0.1):
        for j in np.arange(-5, 5, 0.1):
            grid_Xtest.append([i, j])
    grid_Xtest = np.array(grid_Xtest)
    grid_ytest = net2.predict(grid_Xtest)
    hist_mass = ROOT.TH2F("j0m_j1m", ";j0 m;j1 m ", 50, -5, 5, 50, -5, 5)
    for i in range(grid_Xtest.shape[0]):
        hist_mass.Fill(grid_Xtest[i][0], grid_Xtest[i][1], grid_ytest[i])
    hist_mass.Draw("colz")
    canv.SaveAs("mHH.png")

    ###check the weights
    # yhat_0b = net.predict(X_0b)
    # yhat_2b = net.predict(X_2b)

    # fig, ax = plt.subplots()
    # bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # plt.hist(yhat_0b, bins=bins, histtype='step', label=r'$\hat{y}_{0}$', normed=True)
    # plt.hist(yhat_2b, bins=bins, histtype='step', label=r'$\hat{y}_{1}$', normed=True)
    # plt.legend()
    # ax.set_xlim([0,1])
    # ax.set_xlabel("NN Score")
    # ax.set_ylabel("Arb. Units")
    # plt.savefig("separation.png")
    # plt.clf()



    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
