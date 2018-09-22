import numpy as np
from sklearn.decomposition import PCA
import scipy.misc
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import os
import imageio
import subprocess
from FaceTools import *

AVCONV_BIN = 'ffmpeg'
TEMP_STR = "tempprefix"


def getProcrustesAlignment(X, Y, idx):
    """
    Given correspondences between two point clouds, to center
    them on their centroids and compute the Procrustes alignment to
    align one to the other
    Parameters
    ----------
    X: ndarray(2, M) 
        Matrix of points in X
    Y: ndarray(2, M) 
        Matrix of points in Y (the target point cloud)
    
    Returns
    -------
    (Cx, Cy, Rx):
        Cx: 3 x 1 matrix of the centroid of X
        Cy: 3 x 1 matrix of the centroid of corresponding points in Y
        Rx: A 3x3 rotation matrix to rotate and align X to Y after
        they have both been centered on their centroids Cx and Cy
    """
    Cx = np.mean(X, 1)[:, None]
    #Pull out the corresponding points in Y by using numpy
    #indexing notation along the columns
    YCorr = Y[:, idx]
    #Get the centroid of the *corresponding points* in Y
    Cy = np.mean(YCorr, 1)[:, None]
    #Subtract the centroid from both X and YCorr with broadcasting
    XC = X - Cx
    YCorrC = YCorr - Cy
    #Compute the singular value decomposition of YCorrC*XC^T
    (U, S, VT) = np.linalg.svd(YCorrC.dot(XC.T))
    R = U.dot(VT)
    return (Cx, Cy, R)    



def getVideo(path, computeKeypoints = True, doPlot = False):
    if not os.path.exists(path):
        print("ERROR: Video path not found: %s"%path)
        return None
    #Step 1: Figure out if path is a folder or a filename
    prefix = "%s/"%path
    isFile = False
    if os.path.isfile(path):
        isFile = True
        #If it's a filename, use avconv to split it into temporary frame
        #files and load them in
        prefix = TEMP_STR
        command = [AVCONV_BIN,
                    '-i', path,
                    '-f', 'image2',
                    TEMP_STR + '%d.png']
        subprocess.call(command)

    #Step 2: Load in frame by frame
    #First figure out how many images there are
    #Note: Frames are 1-indexed
    NFrames = 0
    while True:
        filename = "%s%i.png"%(prefix, NFrames+1)
        if os.path.exists(filename):
            NFrames += 1
        else:
            break
    if NFrames == 0:
        print("ERROR: No frames loaded")
        return (None, None)
    #Now load in the video
    allkeypts = []
    allframes = []
    print("Loading video.")
    for i in range(NFrames):
        if i%20 == 0:
            print(".")
        filename = "%s%i.png"%(prefix, i+1)
        allframes.append(mpimage.imread(filename))
        if computeKeypoints:
            face = MorphableFace(filename)
            allkeypts.append(face.getFaceKeypts())
            if doPlot:
                plt.clf()
                plt.subplot(121)
                plt.imshow(face.img)
                plt.axis('off')
                plt.title("Frame %i"%(i+1))
                plt.subplot(122)
                face.plotKeypoints()
                plt.title("BBox width = %i, height = %i"%(face.width, face.height))
                plt.axis('off')
                plt.savefig("Keypoints%i.png"%i, bbox_inches='tight')
        if isFile:
            #Clean up temporary files
            os.remove(filename)
    print("\nFinished loading %s"%path)
    return (allframes, allkeypts)

def makeProcrustesVideo():
    allkeypts = sio.loadmat("allkeypts.mat")["allkeypts"]
    Y = allkeypts[0, :, :].T
    plt.figure(figsize=(10, 5))
    face = MorphableFace("MyExpressions_InitialFrame.jpg")
    allframes, _ = getVideo("MyExpressions.webm", computeKeypoints=False)
    for i in range(1, allkeypts.shape[0]):
        X = allkeypts[i, :, :].T
        Cx, Cy, R = getProcrustesAlignment(X[:, 0:-4], Y[:, 0:-4], np.arange(X.shape[1]-4))
        XNew = X - Cx
        XNew = R.dot(XNew)
        XNew += Cy
        XNew[:, -4::] = Y[:, -4::]
        newImg = face.getForwardMap(XNew.T)

        plt.subplot(121)
        plt.imshow(allframes[i])
        plt.title("Frame %i"%i)
        plt.subplot(122)
        plt.imshow(newImg)
        plt.title("Procrustes warped")

        plt.savefig("Procrustes%i.png"%i)

def getFaceModel():
    allkeypts = sio.loadmat("allkeypts.mat")["allkeypts"]
    Y = allkeypts[0, :, :].T
    face = MorphableFace("MyExpressions_InitialFrame.jpg")
    
    ## Step 1: Do procrustes to align all frames to first frame
    for i in range(1, allkeypts.shape[0]):
        print("Warping %i of %i"%(i, allkeypts.shape[0]))
        X = allkeypts[i, :, :].T
        Cx, Cy, R = getProcrustesAlignment(X[:, 0:-4], Y[:, 0:-4], np.arange(X.shape[1]-4))
        XNew = X - Cx
        XNew = R.dot(XNew)
        XNew += Cy
        XNew[:, -4::] = Y[:, -4::]
        allkeypts[i, :, :] = XNew.T
    
    ## Step 2: Now do PCA on the keypoints
    X = np.reshape(allkeypts, (allkeypts.shape[0], allkeypts.shape[1]*allkeypts.shape[2]))
    XC = np.mean(X, 0)[None, :]
    X -= XC
    pca = PCA(n_components=10)
    pca.fit(X)
    P = pca.components_.T
    sv = np.sqrt(pca.singular_values_)

    plt.subplot(141)
    plt.stem(sv)
    plt.title("Principal Component Standard Deviation")
    for k in range(3):
        Y = XC + sv[k]*P[:, k]
        XKey2 = np.reshape(Y, (allkeypts.shape[1], allkeypts.shape[2]))
        plt.subplot(1, 4, k+2)
        img = face.getForwardMap(XKey2)
        plt.imshow(img)
        plt.title("Principal Component %i"%k)
    plt.show()


if __name__ == '__main__':
    #allkeypts = getAllKeypointsVideo("MyExpressions.webm", doPlot=True)
    #allkeypts = np.array(allkeypts)
    #sio.savemat("allkeypts.mat", {"allkeypts":allkeypts})
    #makeProcrustesVideo()
    getFaceModel()