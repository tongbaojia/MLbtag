import numpy as np
from numpy.lib.recfunctions import stack_arrays
from root_numpy import root2array, root2rec
import pandas as pd
import glob

# ======> EVENT:0
#  runNumber       = 310863
#  lbNumber        = 234
#  eventNumber     = 999258821
#  mHH             = 1513.58
#  mHH_pole        = 1456.72
#  detaHH          = 1.67986
#  dphiHH          = 3.08563
#  drHH            = 3.51326
#  j0_m            = 120.739
#  j0_pt           = 689.915
#  j0_eta          = 0.243731
#  j0_phi          = 2.14801
#  j0_nTrk         = 3
#  j0_nb           = 0
#  j0_trkdr        = 0.212511
#  j1_m            = 139.69
#  j1_pt           = 425.747
#  j1_eta          = 1.92359
#  j1_phi          = -0.937619
#  j1_nTrk         = 4
#  j1_nb           = 0
#  j1_trkdr        = 0.457759
#  j0_trk0_m       = 12.6915
#  j0_trk0_pt      = 182.33
#  j0_trk0_eta     = 0.158777
#  j0_trk0_phi     = 2.16322
#  j0_trk0_Mv2     = -0.718781
#  j1_trk0_m       = 5.70228
#  j1_trk0_pt      = 118.743
#  j1_trk0_eta     = 2.09753
#  j1_trk0_phi     = -0.882759
#  j1_trk0_Mv2     = -0.925586
#  j0_trk1_m       = 7.27628
#  j0_trk1_pt      = 173.139
#  j0_trk1_eta     = 0.289656
#  j0_trk1_phi     = 1.99579
#  j0_trk1_Mv2     = -0.793517
#  j1_trk1_m       = 4.89243
#  j1_trk1_pt      = 51.0089
#  j1_trk1_eta     = 1.67229
#  j1_trk1_phi     = -1.05222
#  j1_trk1_Mv2     = -0.311372
#  Xzz             = 4.47778
#  Xww             = 5.46898
#  Xhh             = 1.78798
#  Rhh             = 24.904
#  Xtt             = 5.12966
#  nresj           = 0
#  weight          = 1

filepath = "../../patrick_data_test16L.root"
data = root2array(filepath,treename="TinyTree",selection="Xhh>1.6 && j0_nTrk>1 && j1_nTrk>1 && Rhh<33")

df = pd.DataFrame(data)

def get_category(n0,n1):
    if n0+n1 == 0: return 0
    if n0==1 and n1==1: return 2
    if n0==2 and n1==2: return 4
    return -1

df['category'] = [ get_category(n0,n1) for (_,(n0,n1)) in df[['j0_nb','j1_nb']].iterrows()]

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
