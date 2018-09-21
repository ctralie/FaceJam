import dlib
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.misc
import time
import librosa
from scipy.interpolate import interp1d
from multiprocessing import Pool as PPool
from FaceTools import *

def getRNNDBNOnsets(filename):
    """
    Call Madmom's implementation of RNN + DBN beat tracking
    :param filename: Path to audio file
    """
    print("Computing madmom beats...")
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    proc = DBNBeatTrackingProcessor(fps=100)
    act = RNNBeatProcessor()(filename)
    b = proc(act)
    return b



songfilename = "AAF.mp3"
matfilename = songfilename[0:-3] + "mat"
if not os.path.exists(matfilename):
    b = getRNNDBNOnsets(songfilename)
    sio.savemat(matfilename, {'b':b})
else:
    b = sio.loadmat(matfilename)['b'].flatten()


# Interpolate a ramp that peaks at every beat and goes to a trough at every half beat
b2 = np.zeros(b.size*2-1)
b2[0::2] = b
b2[1::2] = 0.5*(b[1::] + b[0:-1])
vals = np.ones(b2.size)
vals[1::2] = -1
FPS = 30

X, Fs = librosa.load(songfilename)
songLen = X.size/float(Fs)
N = np.floor(songLen*FPS)
t = np.arange(N)/float(FPS)
f = interp1d(b2, vals)

idx = np.arange(t.size)
idx = idx[(t >= np.min(b2))*(t <= np.max(b2))]
res = np.zeros(t.size)
res[idx] = f(t[idx])



imgfilename = "therock.jpg"
face = MorphableFace(imgfilename)
eyebrow_range = 0.03*(np.max(face.XGrid[:, 1])-np.min(face.XGrid[:, 1]))

def makeWarpsBatch(args):
    (i) = args
    XKey2 = np.array(face.XKey)
    XKey2[eyebrow_idx, 1] += res[i]*eyebrow_range
    imgwarp = face.getForwardMap(XKey2)
    scipy.misc.imsave("%i.png"%i, imgwarp)

parpool = PPool(12)
parpool.map(makeWarpsBatch, (np.arange(t.size)))