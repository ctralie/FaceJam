import dlib
import numpy as np
import matplotlib.pyplot as plt
import time
from FaceTools import *



filename = "therock.jpg"
face = MorphableFace(filename)

## TODO: Use skimage transform
NFrames = 10
for f in range(NFrames):
    plt.clf()
    print("Warping frame %i of %i..."%(f+1, NFrames))
    XKey2 = np.array(face.XKey)
    XKey2[0:-4, :] += 2*np.random.randn(XKey2.shape[0]-4, 2)
    tic = time.time()
    face.plotMapForward(XKey2)
    print("Elapsed Time: %.3g"%(time.time()-tic))
    plt.scatter(XKey2[:, 0], XKey2[:, 1], 2)
    plt.savefig("%i.png"%f)