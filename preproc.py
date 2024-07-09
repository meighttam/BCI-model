from numpy import *
import numpy as np
import glob
import re
from pylab import *
from scipy.signal import *
import pandas as pd
import code

# Required folder structure
# .
# data
# ├── test
# │   ├── Data_S01_Sess01.csv
# │   ├── ...
# │   └── Data_S25_Sess05.csv
# └── train
#     ├── Data_S02_Sess01.csv
#     ├── ...
#     ├── Data_S26_Sess05.csv
#     └── TrainLabels.csv

# Params
# no touchy!
freq = 200.0
epoc_window = 1.3 * freq # 1.3 seconds
# bandpass filter
low = 1.0
high = 40.0

# Returns
#  epochs: (5440, 56, 260) -- [Epochs, Channels, Time]
#  infos: (2, 5440) -- [Labels, User] or [idFeedBack, User]


def bandpass(sig,band,fs):
    B,A = butter(5, array(band)/(fs/2), btype='bandpass')
    return lfilter(B, A, sig, axis=0)


for test in [False, True] : 

    prefix = '' if test is False else 'test_'
    DataFolder = '../data/train/' if test is False else '../data/test/'
    list_of_files = glob.glob(DataFolder + 'Data_*.csv')
    list_of_files.sort()
    print("Found %d files" % len(list_of_files))

    reg = re.compile('\d+')

    X = []
    User = []
    idFeedBack = []
    Session = []
    Feedback = []
    Letter = []
    Word = []
    FeedbackTot = []
    LetterTot = []
    WordTot = []

    for f in list_of_files:
        print(f)
        user,session = reg.findall(f)
        sig = np.array(pd.io.parsers.read_csv(f))

        EEG = sig[:, 1:-2]
        EOG = sig[:, -2]
        Trigger = sig[:,-1]
        # code.interact(local=locals())

        sigF = bandpass(EEG,[low, high], freq)
        sigF = np.concatenate((sigF, EOG[:,None]), axis=1)
    
        idxFeedBack = np.where(Trigger == 1)[0]
    
        for fbkNum, idx in enumerate(idxFeedBack):
            
            X.append(sigF[idx : idx+int(epoc_window), :])
            User.append(int(user))
            idFeedBack.append('S' + user + '_Sess' + session + '_FB' + '%03d' % (fbkNum+1) )
            Session.append(int(session))
            Feedback.append(fbkNum)
            Letter.append(mod(fbkNum,5) + 1)
            Word.append(floor(fbkNum/5)+1)
            FeedbackTot.append(fbkNum + (int(session)-1)*60)
            WordTot.append(floor(fbkNum/5)+1 + (int(session)-1)*12)
	
    if test is False:
        Labels = genfromtxt(DataFolder + 'TrainLabels.csv', delimiter=',', skip_header=1)[:,1]
        inf = array([Labels, User])
    else:
        inf = array([idFeedBack, User])

    X = array(X).transpose((0,2,1))

    save(prefix + 'infos.npy', inf)
    save(prefix + 'epochs.npy', X)