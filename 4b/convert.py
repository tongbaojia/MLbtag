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

##conver to array
def get_category(n0,n1):
    if n0 + n1 == 0: return 0
    if n0 + n1 == 1: return 1
    if n0==1 and n1==1: return 2
    if n0 + n1 == 3: return 3
    if n0==2 and n1==2: return 4
    return -1

def transform_data():
    print "Transforming and preparing training data!!"

    filepath = "hist-MiniNTuple.h5"
    df = pd.read_hdf(filepath, 'data')

    #skim data to get only the two categories being studied and get equal statistics in each
    is0b = df['category'] == 0
    is2b = df['category'] == 2

    df_0b = df[is0b]
    df_2b = df[is2b]
    #print df_2b

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
    X=df_skim[
        [
        #'mHH',
        #'j0_pt', 
        #'j0_eta', 'j0_phi', 
        #'j0_m','j0_nTrk',
        'j0_trkdr', 'j0_trk0_pt', 'j0_trk1_pt', #'j0_trk0_m', 'j0_trk0_eta','j0_trk0_phi',
        #'j1_pt', 
        #'j1_m','j1_nTrk',
        #'j1_eta', 'j1_phi', 
        'j1_trkdr', 'j1_trk0_pt', 'j1_trk1_pt', #'j1_trk0_eta','j1_trk0_phi','j1_trk0_m',
        'j0_trk0_Mv2',
        #'j0_trk0_Mv2', 'j0_trk1_Mv2', 'j1_trk0_Mv2', 'j1_trk1_Mv2', 
         ]].as_matrix()


    Z=df_skim[
        [
        #'mHH',
        #'j0_pt', 
        #'j0_eta', 'j0_phi', 
        #'j0_m','j0_nTrk',
        'j0_trkdr', 'j0_trk0_pt', 'j0_trk1_pt', #'j0_trk0_m', 'j0_trk0_eta','j0_trk0_phi',
        #'j1_pt', 
        #'j1_m','j1_nTrk',
        #'j1_eta', 'j1_phi', 
        'j1_trkdr', 'j1_trk0_pt', 'j1_trk1_pt', #'j1_trk0_eta','j1_trk0_phi','j1_trk0_m',
         ]].as_matrix()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Z = scaler.fit_transform(Z)

    #make y vector
    le = LabelEncoder()
    y = le.fit_transform(df_skim['category'])

    #split into training and testing samples
    X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(X, y, Z, train_size=0.6)
    np.save("X_train", X_train)
    np.save("X_test", X_test)
    np.save("y_train", y_train)
    np.save("y_test", y_test)
    np.save("Z_train", Z_train)
    np.save("Z_test", Z_test)
    

def ntupleToh5():
    print "conver to h5"
    filepath = "/afs/cern.ch/work/b/btong/bbbb/MoriondAnalysis/Output/TEST/data_test/hist-MiniNTuple.root"
    data = root2array(filepath,treename="TinyTree",selection="Xhh>1.6 && j0_nTrk>1 && j1_nTrk>1 && Rhh<58")
    df   = pd.DataFrame(data)

    df['category'] = [ get_category(n0,n1) for (_,(n0,n1)) in df[['j0_nb','j1_nb']].iterrows()]
    ##save this
    df.to_hdf(filepath.replace(".root",".h5"), 'data')
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
    #ntupleToh5()

    ##load the data and save it as numpy matrices
    transform_data()

    print "DONE!!!"

if __name__ == "__main__":
    main()