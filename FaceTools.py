import dlib
import numpy as np
from scipy.spatial import Delaunay
from scipy import interpolate
from GeometryTools import *

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

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
        self.img = dlib.load_rgb_image(filename)
        self.getFaceKeypts()
        self.tri = Delaunay(self.XKey)
        X, Y = np.meshgrid(np.arange(self.img.shape[1]), np.arange(self.img.shape[0]))
        XGrid = np.array([X.flatten(), Y.flatten()], dtype=np.float).T
        allidxs = self.tri.find_simplex(XGrid)
        self.idxs = allidxs[allidxs > -1] # Indices into the simplices
        XGrid = XGrid[allidxs > -1, :]
        self.imgidx = np.arange(self.img.shape[0]*self.img.shape[1])
        self.imgidx = self.imgidx[allidxs > -1]
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
        x1 -= pad*width
        x2 += pad*width
        y1 -= pad*height
        y2 += pad*height
        XKey = np.concatenate((XKey, np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])), 0)
        self.XKey = XKey
        return self.XKey
    
    def mapForward(self, XKey2):
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
        imgwarped = np.array(self.img)
        XGrid2 = getEuclideanFromBarycentric(self.idxs, self.tri, XKey2, self.bary)
        idxi, idxj = np.unravel_index(self.imgidx, (self.img.shape[0], self.img.shape[1]))
        # Do interpolation for each color channel independently
        for c in range(3):
            f = interpolate.interp2d(XGrid2[:, 0], XGrid2[:, 1], self.img[idxi, idxj, c])
            imgwarped[idxi, idxj, c] = f(idxj, idxi)
        return imgwarped
    
    def plotKeypoints(self):
        plt.clf()
        plt.imshow(self.img)
        plt.scatter(self.XKey[:, 0], self.XKey[:, 1])