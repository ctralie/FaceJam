import sys
import os
import dlib
import glob
import time
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import scipy.misc
from scipy.spatial import Delaunay
import scipy.misc
from GeometryTools import *

def getCSM(X, Y):
    XSqr = np.sum(X**2, 1)
    YSqr = np.sum(Y**2, 1)
    return XSqr[:, None] + YSqr[None, :] - 2*X.dot(Y.T)

def shape_to_np(shape, dtype="int"):
    return np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=dtype)

predictor_path = "shape_predictor_68_face_landmarks.dat"
filename = "TheRock.png"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

img = dlib.load_rgb_image(filename)

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))

d = dets[0]

# Get the landmarks/parts for the face in box d.
shape = predictor(img, d)
XKey = shape_to_np(shape)
# Add four points in a square around the face
bds = [min(d.left(), np.min(XKey[:, 0])), max(d.right(), np.max(XKey[:, 0])) \
            , min(d.top(), np.min(XKey[:, 1])), max(d.bottom(), np.max(XKey[:, 1])) ]
x1, x2, y1, y2 = bds
pad = 0.1
width = x2 - x1
height = y2 - y1
x1 -= pad*width
x2 += pad*width
y1 -= pad*height
y2 += pad*height
XKey = np.concatenate((XKey, np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])), 0)

tri = Delaunay(XKey)
X, Y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
XGrid = np.array([X.flatten(), Y.flatten()]).T
idx = tri.find_simplex(XGrid)
idx = np.reshape(idx, (img.shape[0], img.shape[1]))
plt.imshow(idx)
plt.show()

#plt.imshow(img)
#plt.scatter(X[:, 0], X[:, 1])
#plt.show()
    
