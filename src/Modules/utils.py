import os, sys, time
import json, pickle

import numpy as np

_LightGray = '\x1b[38;5;251m'
_Bold = '\x1b[1m'
_Underline = '\x1b[4m'
_Orange = '\x1b[38;5;215m'
_SkyBlue = '\x1b[38;5;38m'
_Reset = '\x1b[0m'

SEED = 0x06902001 ^ 0x06902029 ^ 0x06902039 ^ 0x06902103

class EventTimer():
    def __init__(self, name = '', verbose = True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(_LightGray + '------------------ Begin "' + _SkyBlue + _Bold + _Underline + self.name + _Reset + _LightGray + '" ------------------' + _Reset, file = sys.stderr)
        self.beginTimestamp = time.time()
        return self

    def __exit__(self, type, value, traceback):
        elapsedTime = time.time() - self.beginTimestamp
        if self.verbose:
            print(_LightGray + '------------------ End "' + _SkyBlue + _Bold + _Underline + self.name + _Reset + _LightGray + ' (Elapsed ' + _Orange + f'{elapsedTime:.4f}' + _Reset + 's)" ------------------' + _Reset + '\n', file = sys.stderr)

    def gettime(self):
        return time.time() - self.beginTimestamp

def jsonSave(obj, file):
    with open(file, 'w') as f:
        json.dump(obj, f)

def jsonLoad(file):
    with open(file) as f:
        return json.load(f)

def pickleSave(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

def pickleLoad(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    
def MAP(truth, prediction):
    """Calculate MAP score.

    Parameters
    ----------
    truth : list of iterables
        List of iterables containing relevant document ids.
    
    perdiction : list of iterables.
        List of iterables containing relevant document ids.

    Returns
    -------
    float
        MAP score.
    
    """
    def AP(t, p):
        precisions = 0.
        cnt = 0
        for i, d in enumerate(p):
            if d in t:
                cnt += 1
                precisions += cnt / (i + 1)
        return precisions / len(t)
    return np.mean([AP(t, p) for t, p in zip(truth, prediction)])
