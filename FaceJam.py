import dlib
import numpy as np
import matplotlib.pyplot as plt
import time
from FaceTools import *



filename = "TheRock.png"
face = MorphableFace(filename)

NFrames = 10
for f in range(NFrames):
    print("Warping frame %i of %i..."%(f+1, NFrames))
    XKey2 = np.array(face.XKey)
    XKey2[0:-4, :] += 2*np.random.randn(XKey2.shape[0]-4, 2)
    tic = time.time()
    imgwarp = face.mapForward(XKey2)
    print("Elapsed Time: %.3g"%(time.time()-tic))
    plt.clf()
    plt.imshow(imgwarp)
    plt.scatter(XKey2[:, 0], XKey2[:, 1], 5)
    plt.savefig("%i.png"%f)