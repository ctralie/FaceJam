import numpy as np
from scipy.spatial import tsearch
from sklearn.decomposition import PCA
import scipy.misc
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import os
import imageio
import subprocess
from FaceTools import *
from GeometryTools import *

AVCONV_BIN = 'ffmpeg'
TEMP_STR = "tempprefix"

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

def getFaceModel(n_components=10, doPlot = False):
    allkeypts = sio.loadmat("allkeypts.mat")["allkeypts"]
    Y = allkeypts[0, :, :].T
    
    ## Step 1: Do procrustes to align all frames to first frame
    for i in range(1, allkeypts.shape[0]):
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
    pca = PCA(n_components=n_components)
    pca.fit(X)
    P = pca.components_.T
    sv = np.sqrt(pca.singular_values_)

    face = MorphableFace("MyExpressions_InitialFrame.jpg")
    if doPlot:
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
    
    return (face, XC.flatten(), P, sv)

def transferExpression(modelface, XC, P, targetface, X):
    """
    Given a model face and its principal components, 
    apply a principal component warp in the model face
    and transfer it to a new face
    Parameters
    ----------
    modelface: MorphableFace
        An object of the model face
    XC: ndarray(NPrincipalComponents)
        Centroid of model keypoints
    P: ndarray(NKeypoints, NPrincipalComponents)
        Principal components for the model keypoints
    targetface: MorphableFace
        Face to be warped
    X: ndarray(NPrincipalComponents)
        Principal coordinates of the facial expression to realize
    """
    ## Step 1: Make keypoints in the model coordinate system
    Y = XC[None, :] + np.sum(X[None, :]*P, 1)
    XKey2 = np.reshape(Y, (modelface.XKey.shape[0], modelface.XKey.shape[1]))

    ## Step 2: Find barycentric coordinates in model coordinate system, 
    ## **using triangles from the target system**
    idxs = [getTriangleIdx(modelface.XKey, targetface.tri.simplices, XKey2[k, :]) for k in range(XKey2.shape[0])]
    idxs = np.array(idxs).flatten()
    bary = getBarycentricCoords(XKey2, idxs, targetface.tri, modelface.XKey)

    ## Step 3: Apply barycentric coordinates in the target face coordinate system
    XKey2 = getEuclideanFromBarycentric(idxs, targetface.tri, targetface.XKey, bary)

    ## Step 4: Apply the warp to the target face based on the new keypoints
    return (XKey2, targetface.getForwardMap(XKey2))


def testPCsTheRock():
    (modelface, XC, P, sv) = getFaceModel(doPlot = False)
    face = MorphableFace("therock.jpg")

    plt.subplot(221)
    plt.imshow(face.img)
    plt.title("Original")
    plt.axis('off')
    for k in range(3):
        plt.subplot(1, 3, k+1)
        x = np.array(sv)
        x = np.zeros_like(sv)
        x[k] = sv[0]
        (XKey2, newimg) = transferExpression(modelface, XC, P, face, x)
        plt.imshow(newimg)
        plt.title("PC %i"%(k+1))
        plt.axis('off')
    plt.show()

def showMyFaceOnTheRock():
    (modelface, XC, P, sv) = getFaceModel(doPlot = False)
    print("XC.shape = ", XC.shape)
    print("P.shape = ", P.shape)
    face = MorphableFace("therock.jpg")
    (allframes, allkeypts) = getVideo("MyExpressions.webm")
    plt.figure(figsize=(12, 6))
    for i in range(len(allframes)):
        keypts = allkeypts[i]
        keypts[-4::, :] = allkeypts[0][-4::, :]
        keypts = keypts.flatten()
        keypts -= XC
        coords = keypts[None, :].dot(P)
        
        XKey2, newimg = transferExpression(modelface, XC, P, face, coords.flatten())
        plt.clf()
        plt.subplot(121)
        plt.imshow(allframes[i])
        plt.title("Frame %i"%i)
        plt.subplot(122)
        plt.imshow(newimg)
        plt.title("The Rock Warped")
        plt.savefig("MeToTheRock%i.png"%i)


if __name__ == '__main__':
    #allkeypts = getAllKeypointsVideo("MyExpressions.webm", doPlot=True)
    #allkeypts = np.array(allkeypts)
    #sio.savemat("allkeypts.mat", {"allkeypts":allkeypts})
    #makeProcrustesVideo()
    #testPCsTheRock()
    showMyFaceOnTheRock()