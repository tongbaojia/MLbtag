import ROOT
import numpy as np
from numpy.lib.recfunctions import stack_arrays
from root_numpy import root2array, root2rec
import glob
import pandas as pd


filepath = '/atlas/local/BtagOptimizationNtuples/group.perf-flavtag.410000.PowhegPythiaEvtGen.AOD.e3698_s2997_r8903_r8906.v21-1.db-b5223bf2_Akt4EMTo/group.perf-flavtag.11010668.Akt4EMTo._000001.root'
rootfile = ROOT.TFile(filepath)
roottree = rootfile.Get("bTagAntiKt4EMTopoJets")

data = root2array(filepath, selection='njets>7')
df = pd.DataFrame(data)

def root2pandas(files_path, tree_name, **kwargs):
    '''
    Args:
    -----
        files_path: a string like './data/*.root', for example
        tree_name: a string like 'bTag_AntiKt4EMTopoJets' corresponding to the name of the folder inside the root 
                   file that we want to open
        kwargs: arguments taken by root2array, such as branches to consider, start, stop, step, etc
    Returns:
    --------    
        output_panda: a pandas dataframe like allbkg_df in which all the info from the root file will be stored
    
    Note:
    -----
        if you are working with .root files that contain different branches, you might have to mask your data
        in that case, return pd.DataFrame(ss.data)
    '''
    # -- create list of .root files to process
    files = glob.glob(files_path)
    
    # -- process ntuples into rec arrays
    ss = stack_arrays([root2array(fpath, tree_name, **kwargs).view(np.recarray) for fpath in files])

    try:
        return pd.DataFrame(ss)
    except Exception:
        return pd.DataFrame(ss.data)

df.to_hdf('testl_pd.h5', 'data')
print "DONE!!!"
