"""
A program that animates faces to the music.  The eyebrows go to the beat, and
the rest of the face expression correlates with place in the song structure 
(e.g. verse and chorus should have a different expression)
"""
import dlib
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.misc
import time
import librosa
import argparse
import sys
sys.path.append("GraphDitty")
from SongStructure import getFusedSimilarity
from DiffusionMaps import getDiffusionMap
from scipy.interpolate import interp1d
from multiprocessing import Pool as PPool
from FaceTools import *
from ExpressionsModel import *

TEMP_STR = "musicvideo"

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

    
if __name__ == '__main__':
    """
    The glue that glues everything together
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--songfilename', type=str, required=True, help="Path to audio file")
    parser.add_argument('--imgfilename', type=str, required=True, help="Path to image with a face in it, which will be animated")
    parser.add_argument('--videoname', type=str, required=True, help="Path to which to save the music video result")
    ## TODO: Check range of number of components so it's less than # of keypoints and # of diffusion maps time indices
    parser.add_argument('--n_components', type=int, default=10, help="Number of diffusion maps and facial landmark principal coordinates to use.  This *roughly* corresponds to the number of structual indicator functions.")
    parser.add_argument('--FPS', type=int, default=30, help="Video framerate")
    parser.add_argument('--NThreads', type=int, default=4, help="Number of CPU threads to use when doing the warps")
    opt = parser.parse_args()
    songfilename, imgfilename, videoname = opt.songfilename, opt.imgfilename, opt.videoname
    n_components, FPS, NThreads = opt.n_components, opt.FPS, opt.NThreads

    ## Step 1: Load in the audio and perform the structure analysis
    print("Doing structural diffusion maps...")
    res = getFusedSimilarity(songfilename, sr=22050, hop_length=512, win_fac=10, \
                            wins_per_block=20, K=10, reg_diag=1.0, reg_neighbs=0.5, \
                            niters=10, neigs=8, do_animation=False, plot_result=False)
    XDiffusion = getDiffusionMap(res['WFused'], NEigs=n_components+1)
    time_interval = res['time_interval']
    # Put most important components first
    XDiffusion = np.fliplr(XDiffusion[:, 0:-1])
    # Sphere-normalize
    XDiffusion -= np.mean(XDiffusion, 0)[None, :]
    XDiffusion /= np.sqrt(np.sum(XDiffusion**2, 1))[:, None]
    XDiffusion /= np.max(np.abs(XDiffusion), 0)[None, :]
    print("Finished diffusion maps")
    plt.figure(figsize=(12, 4))
    plt.imshow(XDiffusion.T, aspect='auto', extent = (0, time_interval*XDiffusion.shape[0], 0, n_components))
    plt.ylabel("Component")
    plt.xlabel("Time")
    plt.title("Diffusion Maps")
    plt.yticks(0.5 + np.arange(n_components), ["%i"%i for i in range(1, n_components+1)])
    plt.savefig("%s_DiffusionMaps.png"%videoname)

    ## Step 2: Interpolate a ramp that peaks at every beat and goes to a trough 
    ## at every half beat, and use this for the eyebrows
    matfilename = songfilename[0:-4] + "_beats.mat"
    if not os.path.exists(matfilename):
        print("Computing beats...")
        # Cache beats since they take long to compute
        b = getRNNDBNOnsets(songfilename)
        sio.savemat(matfilename, {'b':b})
    else:
        b = sio.loadmat(matfilename)['b'].flatten()

    b2 = np.zeros(b.size*2-1)
    b2[0::2] = b
    b2[1::2] = 0.5*(b[1::] + b[0:-1])
    vals = np.ones(b2.size)
    vals[1::2] = -1

    X, Fs = librosa.load(songfilename)
    songLen = X.size/float(Fs)
    N = np.floor(songLen*FPS)
    tsvideo = np.arange(N)/float(FPS)
    f = interp1d(b2, vals)

    idx = np.arange(tsvideo.size)
    idx = idx[(tsvideo >= np.min(b2))*(tsvideo <= np.max(b2))]
    eyebrows_diff = np.zeros(tsvideo.size)
    eyebrows_diff[idx] = f(tsvideo[idx])


    ## Step 3: Setup the new keypoints and do the warps
    (modelface, XC, P, sv) = getFaceModel(n_components = n_components, doPlot = False)
    face = MorphableFace(imgfilename)
    eyebrow_range = 0.03*(np.max(face.XGrid[:, 1])-np.min(face.XGrid[:, 1]))
    eyebrows_diff *= eyebrow_range
    # Resample diffusion maps so that they're at the video sample rate
    tsdiffusion = time_interval*np.arange(XDiffusion.shape[0])
    XDiffusionVideo = np.zeros((tsvideo.size, n_components))
    for k in range(XDiffusion.shape[1]):
        f = interp1d(tsdiffusion, XDiffusion[:, k])
        idx = np.arange(tsvideo.size)
        idx = idx[(tsvideo >= tsdiffusion[0])*(tsvideo <= tsdiffusion[-1])]
        XDiffusionVideo[idx, k] = f(tsvideo[idx])
    # Transfer diffusion map eigenvectors to PCA coordinates
    x = sv[0]*XDiffusionVideo

    def makeWarpsBatch(args):
        (i) = args
        print("Processing frame %i"%i)
        (XKey2, imgwarp) = transferExpression(modelface, XC, P, face, x[i, :])
        XKey2[eyebrow_idx, 1] += eyebrows_diff[i]
        imgwarp = face.getForwardMap(XKey2)
        scipy.misc.imsave("%s%i.png"%(TEMP_STR, i), imgwarp)

    parpool = PPool(12)
    parpool.map(makeWarpsBatch, (np.arange(tsvideo.size)))

    ## Step 4: Make the music video
    command = [AVCONV_BIN,
                '-r', "%i"%FPS,
                '-i', TEMP_STR + '%d.png',
                '-r', "%i"%FPS,
                '-i', songfilename,
                '-b', '30000k',
                videoname]
    subprocess.call(command)
    #Clean up temp files
    for i in range(tsvideo.size):
        os.remove("%s%i.png"%(TEMP_STR, i))