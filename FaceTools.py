import time
import dlib
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from GeometryTools import *

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

eyebrow_idx = np.concatenate((np.arange(17, 22), np.arange(22, 27)))

def shape_to_np(shape, dtype="int"):
    """
    Used to convert from a shape object returned by dlib to an np array
    """
    return np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=dtype)

class MorphableFace(object):
    """
    An object which stores a face along with facial landmarks, as computed by dlib,
    which can be used to warp the face into different expressions
    """
    def __init__(self, filename):
        """
        Constructor for the face object.  Initializes the keypoints, sets up some
        data structures that keep track of the locations of the keypoints to help
        later with interpolation, and computes barycentric coordinates
        Parameters
        ----------
        filename: string
            Path to image file with at least one face in it
        """
        self.img = dlib.load_rgb_image(filename)
        self.getFaceKeypts()
        self.tri = Delaunay(self.XKey)
        X, Y = np.meshgrid(np.arange(self.img.shape[1]), np.arange(self.img.shape[0]))
        XGrid = np.array([X.flatten(), Y.flatten()], dtype=np.float).T
        allidxs = self.tri.find_simplex(XGrid)
        self.idxs = allidxs[allidxs > -1] # Indices into the simplices
        XGrid = XGrid[allidxs > -1, :]
        imgidx = np.arange(self.img.shape[0]*self.img.shape[1])
        imgidx = imgidx[allidxs > -1]
        self.imgidxi, self.imgidxj = np.unravel_index(imgidx, (self.img.shape[0], self.img.shape[1]))
        colors = self.img[self.imgidxi, self.imgidxj, :]
        self.colors = colors/255.0
        self.pixx = np.arange(np.min(self.imgidxj), np.max(self.imgidxj)+1)
        self.pixy = np.arange(np.min(self.imgidxi), np.max(self.imgidxi)+1)
        self.grididxx, self.grididxy = np.meshgrid(self.pixx, self.pixy)
        self.XGrid = XGrid
        self.bary = getBarycentricCoords(XGrid, self.idxs, self.tri, self.XKey)

    def getFaceKeypts(self, pad = 0.1):
        """
        Return the keypoints of the first face detected in the image
        Parameters
        ----------
        img: ndarray(M, N, 3)
            An RGB image which contains at least one face
        pad: float (default 0.1)
            The factor by which to pad the bounding box of the facial landmarks
            by the 4 additional landmarks
        
        Returns
        -------
        XKey: ndarray(71, 2)
            Locations of the facial landmarks.  Last 4 are 4 corners
            of the expanded bounding box
        """

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(self.img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        d = dets[0]

        # Get the landmarks/parts for the face in box d.
        shape = predictor(self.img, d)
        XKey = shape_to_np(shape)
        # Add four points in a square around the face
        bds = [min(d.left(), np.min(XKey[:, 0])), max(d.right(), np.max(XKey[:, 0])) \
                    , min(d.top(), np.min(XKey[:, 1])), max(d.bottom(), np.max(XKey[:, 1])) ]
        x1, x2, y1, y2 = bds
        width = x2 - x1
        height = y2 - y1
        self.width = width
        self.height = height
        print("width = %i, height = %i"%(width, height))
        x1 -= pad*width
        x2 += pad*width
        y1 -= pad*height
        y2 += pad*height
        XKey = np.concatenate((XKey, np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])), 0)
        self.XKey = XKey
        return self.XKey
    
    def plotForwardMapSplat(self, XKey2):
        """
        Extend the map from they keypoints to these new keypoints to a refined piecewise
        affine map from triangles to triangles, and then splat the result via a scatterplot
        Parameters
        ----------
        XKey2: ndarray(71, 2)
            New locations of facial landmarks
        """
        XGrid2 = getEuclideanFromBarycentric(self.idxs, self.tri, XKey2, self.bary)
        plt.imshow(self.img)
        plt.scatter(XGrid2[:, 0], XGrid2[:, 1], 2, c=self.colors)
    
    def getForwardMap(self, XKey2):
        """
        Extend the map from they keypoints to these new keypoints to a refined piecewise
        affine map from triangles to triangles
        Parameters
        ----------
        XKey2: ndarray(71, 2)
            New locations of facial landmarks
        
        Returns
        -------
        imgwarped: ndarray(M, N, 3)
            An image warped according to the map
        """
        XGrid2 = getEuclideanFromBarycentric(self.idxs, self.tri, XKey2, self.bary)
        imgret = np.array(self.img)
        interpbox = griddata(XGrid2, self.colors, (self.grididxx, self.grididxy))
        interpbox = np.array(np.round(255*interpbox), dtype = np.uint8)
        for c in range(3):
            interpc = interpbox[:, :, c]
            imgret[self.imgidxi, self.imgidxj, c] = interpc.flatten()
        # Some weird stuff happens at the boundaries
        for k in [0, -1]:
            imgret[self.imgidxi[k], :, :] = self.img[self.imgidxi[k], :, :]
            imgret[:, self.imgidxj[k], :] = self.img[:, self.imgidxj[k], :]
        return imgret

    def plotKeypoints(self, drawLandmarks = True, drawTriangles = False):
        """
        Plot the image with the keypoints superimposed
        """
        plt.imshow(self.img)
        if drawLandmarks:
            plt.scatter(self.XKey[:, 0], self.XKey[:, 1])
        if drawTriangles:
            plt.triplot(self.XKey[:, 0], self.XKey[:, 1], self.tri.simplices)


def testWarp():
    """
    Make sure the warping is working by randomly perturbing the
    facial landmarks a bunch of times
    """
    filename = "therock.jpg"
    face = MorphableFace(filename)
    NFrames = 10
    for f in range(NFrames):
        plt.clf()
        print("Warping frame %i of %i..."%(f+1, NFrames))
        XKey2 = np.array(face.XKey)
        XKey2[0:-4, :] += 2*np.random.randn(XKey2.shape[0]-4, 2)
        tic = time.time()
        res = face.getForwardMap(XKey2)
        plt.imshow(res)
        print("Elapsed Time: %.3g"%(time.time()-tic))
        plt.scatter(XKey2[:, 0], XKey2[:, 1], 2)
        plt.savefig("WarpTest%i.png"%f)

if __name__ == '__main__':
    face = MorphableFace("therock.jpg")
    plt.subplot(121)
    face.plotKeypoints()
    plt.xlim([250, 500])
    plt.ylim([250, 0])
    plt.title("DLib Facial Landmarks")
    plt.subplot(122)
    face.plotKeypoints(False, True)
    plt.xlim([250, 500])
    plt.ylim([250, 0])
    plt.title("Delaunay Triangulation")
    plt.show()