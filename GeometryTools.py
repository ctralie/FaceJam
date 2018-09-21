import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def getCSM(X, Y):
    XSqr = np.sum(X**2, 1)
    YSqr = np.sum(Y**2, 1)
    return XSqr[:, None] + YSqr[None, :] - 2*X.dot(Y.T)

def getBarycentricCoords(X, idxs, tri, XTri):
    """
    Get the barycentric coordinates of all points
    https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    
    Parameters
    ---------- 
    X: ndarray (N, 2)
        Points of which to compute barycentric coordinates
    idxs: ndarray(N)
        Indices of each point into the Delaunay triangulation.
    tri: object
        Delaunay triangulation data structure
    XTri: ndarray (K, 2)
        Landmark points into which the delaunay triangle indexes
    
    Returns
    -------
    bary: ndarray (N, 3)
        The barycentric coordinates of each point with respect to its triangle
    """
    a, b, c = [XTri[tri.simplices[idxs][:, k], :] for k in range(3)]
    v0 = b - a
    v1 = c - a
    v2 = X - a
    d00 = np.sum(v0*v0, 1)
    d01 = np.sum(v0*v1, 1)
    d11 = np.sum(v1*v1, 1)
    d20 = np.sum(v2*v0, 1)
    d21 = np.sum(v2*v1, 1)
    denom = d00*d11 - d01*d01
    v = (d11*d20 - d01*d21)/denom
    w = (d00*d21 - d01*d20)/denom
    u = 1.0 - v - w
    return np.array([u, v, w]).T

def getEuclideanFromBarycentric(idxs, tri, XTri, bary):
    """
    Return Euclidean coordinates from a barycentric coordinates
    idxs: ndarray(N)
        Indices of each point into the Delaunay triangulation.
    tri: object
        Delaunay triangulation data structure
    XTri: ndarray (K, 2)
        Landmark points into which the delaunay triangle indexes
    bary: ndarray(N, 3)
        Barycentric coordinates of each point in X with respect to its triangle
    """
    a, b, c = [XTri[tri.simplices[idxs][:, k], :] for k in range(3)]
    u, v, w = bary[:, 0], bary[:, 1], bary[:, 2]
    return u[:, None]*a + v[:, None]*b + w[:, None]*c


def testBarycentricTri():
    """
    Testing point location and barycentric coordinates.
    Plot the indices for the simplices to figure out how point
    location happened, and then go to barycentric and back
    to make sure it's the identity
    """
    np.random.seed(2)
    XTri = np.random.rand(6, 2)
    tri = Delaunay(XTri)
    N = 100
    pix = np.linspace(0, 1, N)
    J, I = np.meshgrid(pix, pix)
    XGrid = np.array([J.flatten(), I.flatten()]).T
    allidxs = tri.find_simplex(XGrid)
    idxs = allidxs[allidxs > -1]
    XGrid = XGrid[allidxs > -1, :]

    bary = getBarycentricCoords(XGrid, idxs, tri, XTri)
    XGridBack = getEuclideanFromBarycentric(idxs, tri, XTri, bary)

    print("Is Cartesian -> Barycentric -> Cartesian Identity?: %s"%np.allclose(XGrid, XGridBack))

    plt.subplot(121)
    plt.imshow(np.reshape(allidxs, (N, N)), extent=(0, 1, 1, 0))
    plt.triplot(XTri[:, 0], XTri[:, 1], triangles=tri.simplices.copy())
    plt.subplot(122)
    #plt.scatter(XGrid[:, 0], XGrid[:, 1], 20, bary)
    plt.scatter(XGridBack[:, 0], XGridBack[:, 1])
    plt.triplot(XTri[:, 0], XTri[:, 1], triangles=tri.simplices.copy(), c='C1')
    plt.show()
    pass

if __name__ == '__main__':
    testBarycentricTri()