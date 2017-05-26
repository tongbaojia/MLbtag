import numpy as np
from numpy.lib.recfunctions import stack_arrays
from root_numpy import root2array, root2rec
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import time

##for more info, see here:
##https://github.com/tongbaojia/MakePlot/blob/master/TinyTree.h

directory="/afs/cern.ch/work/b/btong/bbbb/MoriondAnalysis/Output/TEST/data_test/"
sig_directory="/afs/cern.ch/work/b/btong/bbbb/MoriondAnalysis/Output/TEST/signal_G_hh_c10_M1500/"

##conver to array
def get_category(n0,n1):
    if n0 + n1 == 0: return 0
    if n0==1 and n1==1: return 2
    if n0==2 and n1==2: return 4
    return -1

def transform_data():
    print "Transforming and preparing training data!!"

    # #filepath = directory+"hist-MiniNTuple.h5"
    # filepath = "/afs/cern.ch/user/b/btong/public/hist-MiniNTuple.h5"
    # df = pd.read_hdf(filepath, 'data')

    # #skim data to get only the two categories being studied and get equal statistics in each
    # is0b = df['category'] == 0
    # is2b = df['category'] == 2

    # df_0b = df[is0b]
    # df_2b = df[is2b]


    filepath_bkg = "hist-MiniNTuple_bkg.h5"
    df_0b = pd.read_hdf(filepath_bkg, 'data')
    filepath_sig = "hist-MiniNTuple_1500.h5"
    df_2b = pd.read_hdf(filepath_sig, 'data')

    ##clean data a bit
    df_2b = df_2b[df_2b['j0_trk0_pt'] < 500]
    df_2b = df_2b[df_2b['j1_trk0_pt'] < 500]
    df_0b = df_0b[df_0b['j0_trk0_pt'] < 500]
    df_0b = df_0b[df_0b['j1_trk0_pt'] < 500]

    ##make the 0b and the 2b have the same number of events
    frames = [df_0b[:df_2b.shape[0]],df_2b] 
    df_skim = pd.concat(frames)

    print "skim"
    print df_skim[:2]

    #make X matrix
    #inputs = ['j0_trk0_pt','j0_trk1_pt','j1_trk0_pt','j1_trk1_pt','j0_trkdr','j1_trkdr','j0_nTrk','j1_nTrk','detaHH']
    #inputs = []
    inputs = ['j0_trk0_pt','j0_trk1_pt','j1_trk0_pt','j1_trk1_pt','j0_trkdr','j1_trkdr','j0_nTrk','j1_nTrk','detaHH','mHH', 'j1_m', 'j0_m']
    inputs2 = ['j0_trk0_pt','j0_trk1_pt','j1_trk0_pt','j1_trk1_pt','j0_trkdr','j1_trkdr','j0_nTrk','j1_nTrk','detaHH','j0_trk0_Mv2']
    Z=df_skim[inputs].as_matrix()
    X=df_skim[inputs].as_matrix()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Z = scaler.fit_transform(Z)

    #make y vector
    le = LabelEncoder()
    y = le.fit_transform(df_skim['category'])

    np.save("X_sig",X)
    np.save("y_sig",y)
    #split into training and testing samples
    X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(X, y, Z, train_size=0.6)
    np.save("X_sig_train", X_train)
    np.save("X_sig_test", X_test)
    np.save("y_sig_train", y_train)
    np.save("y_sig_test", y_test)
    np.save("Z_sig_train", Z_train)
    np.save("Z_sig_test", Z_test)
    

def ntupleToh5(directory = directory, outname=""):
    filepath = directory + "hist-MiniNTuple.root"
    #    data = root2array(filepath,treename="TinyTree",selection="Xhh>1.6 && j0_nTrk>1 && j1_nTrk>1 && Rhh<58")
    data = root2array(filepath,treename="TinyTree",selection="(j0_nb + j1_nb == 1) && j0_nTrk>1 && j1_nTrk>1")
    df   = pd.DataFrame(data)

    df['category'] = [ get_category(n0,n1) for (_,(n0,n1)) in df[['j0_nb','j1_nb']].iterrows()]
    #df['category'] = 0
    ##save this
    df.to_hdf(filepath.replace(".root", outname + ".h5"), 'data')
    n0b=0
    n2b=0
    n4b=0
    for category in df['category']: 
        if category == 0: n0b+=1
        if category == 2: n2b+=1
        if category == 4: n4b+=1
    print "n0b:",n0b
    print "n2b:",n2b
    print "n4b:",n4b

def main():
    ##conver root to h5 file
    #ntupleToh5(directory=sig_directory, outname="_1500")
    #ntupleToh5(directory=directory, outname="_bkg")
    #ntupleToh5()

    ##load the data and save it as numpy matrices
    transform_data()

    print "DONE!!!"

if __name__ == "__main__":
    main()
