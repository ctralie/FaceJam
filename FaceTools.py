import time
import dlib
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import griddata, interpn
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
        Constructor for the face object.  Initializes the keypoints
        and their delaunay triangulation
        Parameters
        ----------
        filename: string
            Path to image file with at least one face in it
        """
        self.img = dlib.load_rgb_image(filename)
        self.get_face_keypts()

    def get_bbox(self):
        """
        Get the bounding box of the keypoints
        """
        j1, i1 = np.floor(np.min(self.XKey, axis=0))
        j2, i2 = np.ceil(np.max(self.XKey, axis=0))
        return clamp_bbox(np.array([i1, i2, j1, j2], dtype=int), self.img.shape)

    def setup_grid(self, bbox):
        """
        Setup a grid given a bounding box, and compute triangle
        indices and barycentric coordinates on this grid
        Parameters
        ----------
        bbox: ndarray([i1, i2, j1, j2])
            A bounding box
        """
        self.bbox = bbox
        self.pixx = np.arange(bbox[2], bbox[3]+1)
        self.pixy = np.arange(bbox[0], bbox[1]+1)
        self.XKeyWBbox = add_bbox_to_keypoints(self.XKey, bbox)
        self.tri = Delaunay(self.XKeyWBbox)
        X, Y = np.meshgrid(self.pixx, self.pixy)
        self.XGrid = np.array([X.flatten(), Y.flatten()], dtype=np.float).T
        self.idxs = self.tri.find_simplex(self.XGrid)
        self.bary = get_barycentric(self.XGrid, self.idxs, self.tri, self.XKeyWBbox)
        self.colors = self.img[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1, :]/255.0


    def get_face_keypts(self, pad = 0.1):
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
        self.XKey = XKey
        return self.XKey
    
    def get_forward_map(self, XKey2):
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
        [i1, i2, j1, j2] = self.bbox
        XKey2WBbox = add_bbox_to_keypoints(XKey2, self.bbox)
        XGrid2 = barycentric_to_euclidean(self.idxs, self.tri, XKey2WBbox, self.bary)
        diff = XGrid2 - self.XGrid
        XGrid2 = self.XGrid - diff
        XGrid2 = np.fliplr(XGrid2)
        # Numerical precision could cause coords to be out of bounds
        XGrid2[XGrid2[:, 0] <= np.min(self.pixy), 0] = np.min(self.pixy)
        XGrid2[XGrid2[:, 0] >= np.max(self.pixy), 0] = np.max(self.pixy)
        XGrid2[XGrid2[:, 1] <= np.min(self.pixx), 1] = np.min(self.pixx)
        XGrid2[XGrid2[:, 1] >= np.max(self.pixx), 1] = np.max(self.pixx)
        imgret = np.array(self.img)
        shape = (i2-i1+1, j2-j1+1)
        for c in range(3):
            interpbox = interpn((self.pixy, self.pixx), self.colors[:, :, c], XGrid2)
            interpbox = np.array(np.round(255*interpbox), dtype = np.uint8)
            imgret[i1:i2+1, j1:j2+1, c] = np.reshape(interpbox, shape)
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


def test_delaunay(filename):
    face = MorphableFace(filename)
    plt.subplot(121)
    face.plotKeypoints()
    plt.title("DLib Facial Landmarks")
    plt.subplot(122)
    face.plotKeypoints(False, True)
    plt.title("Delaunay Triangulation")
    plt.show()

def test_warp(filename):
    """
    Make sure the warping is working by randomly perturbing the
    facial landmarks a bunch of times
    """
    face = MorphableFace(filename)
    bbox = face.get_bbox()
    expand_bbox(bbox, 0.2, face.img.shape)
    face.setup_grid(bbox)
    NFrames = 10
    for f in range(NFrames):
        plt.clf()
        print("Warping frame %i of %i..."%(f+1, NFrames))
        XKey2 = np.array(face.XKey, dtype=float)
        XKey2 += 2*np.random.randn(XKey2.shape[0], 2)
        tic = time.time()
        res = face.get_forward_map(XKey2)
        plt.imshow(res)
        print("Elapsed Time: %.3g"%(time.time()-tic))
        plt.scatter(XKey2[:, 0], XKey2[:, 1], 2)
        plt.savefig("WarpTest%i.png"%f)

if __name__ == '__main__':
    filename = "CVPR2014Data/1/00001.jpg"
    test_warp(filename)