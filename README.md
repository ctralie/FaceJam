# FaceJam (<a href = "https://labrosa.ee.columbia.edu/hamr_ismir2018/">HAMR 2018</a>)

The goal of this 2 day hackathon was given a song and an image with a face in it, to make a program that automatically detects the face and makes a music video in which the face's eyebrows move to the beat, and in which the face changes its expression depending where we are in the song (e.g. verse vs chorus).  Click on the thumbnail below to show an example animating the face of <a href = "https://en.wikipedia.org/wiki/Dwayne_Johnson">Dwayne Johnson ("The Rock")</a> to go along with <a href = "https://www.youtube.com/watch?v=CDl9ZMfj6aE">Alien Ant Farm's Smooth Criminal</a>:


[![Animating Dwayne Johnson's face to Alien Ant Farm's 'Smooth Criminal'](https://img.youtube.com/vi/nCy7NGGN-3U/1.jpg)](https://www.youtube.com/watch?v=nCy7NGGN-3U)

## Dependencies
This requires that you have numpy/scipy installed, as well as <a href = "https://www.ffmpeg.org/">ffmpeg</a> for video loading and saving, <a href = "http://dlib.net/">dlib</a> for facial landmark detection (pip install dlib), <a href = "https://librosa.github.io/librosa/">librosa</a> for audio features (pip install librosa), and <a href = "https://github.com/CPJKU/madmom">madmom</a> for beat tracking (pip install madmom).

## Usage
To run this program on your own songs, first check it out as follows

~~~~~ bash
git clone --recursive https://github.com/ctralie/FaceJam.git
~~~~~

Then, type the following at the root of the FaceJam directory

~~~~~ bash
python FaceJam.py --songfilename (path to your song) --imgfilename (path to image with a face in it) --videoname (output name for the resulting music video, e.g. "myvideo.avi")
~~~~~

to see more options, including number of threads to make it faster on a multicore machine, please type
~~~~~ bash
python FaceJam.py --help
~~~~~

## Algorithm / Development

Below I will describe some of the key steps of the algorithm


### Piecewise Affine Face Warping (Main Code: FaceTools.py, GeometryTools.py)

We use the delaunay triangulation on <a href = "http://dlib.net/face_landmark_detection.py.html">facial landmarks</a> to create a bunch of triangles.  We then define a piecewise affine (triangle to triangle) warp to extend the map <b>f</b> from facial landmarks in one position to facial landmarks in anther position to a map <b>g</b> from all pixels in the face bounding box in one position to the pixels in the bounding box of a face in another position.  Below is an example of a Delaunay triangulation on The Rock's face

<img src = "http://www.ctralie.com/Research/FaceJam_HAMR2018/TheRockDelaunay.svg">



And below is an example using this Delaunay triangulation to construct piecewise affine maps for randomly perturbed landmarks 

[![Warping The Rock's Face with A Piecewise Affine Warp Based on Facial Landmarks'](https://img.youtube.com/vi/PEP8yz_msjw/1.jpg)](https://www.youtube.com/watch?v=PEP8yz_msjw)


### Facial Expressions PCA / Barycentric Face Expression Cloning (Main Code: ExpressionsModel.py)

Next, I took a video of myself making a bunch of facial expression to make a "facial expressions dictionary" of sorts.  I perform a <a href = "https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem">procrustes alignment</a> of all of the facial landmarks to the first frame to control for rigid motions of my head.  Then, considering the collections of xy positions of all of my facial landmarks as one big vector, I perform PCA on this collection of landmark positions to learn a lower dimensional coordinate system for the space of my expressions

<img src = "http://www.ctralie.com/Research/FaceJam_HAMR2018/PrincipalComponents.svg">

I then use barycentric coordinates of my facial landmarks relative to triangles drawn on The Rock's face to define a piecewise affine warp on his face

<BR><BR>
<img src = "http://www.ctralie.com/Research/FaceJam_HAMR2018/TheRockPCs.png">

<BR><BR>
Here's an entire video showing me "cloning" my face expression to The Rock's face this way


[![Cloning my expressions to The Rock's Face'](https://img.youtube.com/vi/DLe8c7b0GTE/1.jpg)](https://www.youtube.com/watch?v=DLe8c7b0GTE)

### Song Structure Diffusion Maps + Beat Tracking = FaceJam! (Main Code: GraphDitty submodule, FaceJam.py)

Finally, given a song, I perform beat tracking using <a href = "https://github.com/CPJKU/madmom">madmom</a>, the best beat tracking software I'm aware of, and I also perform diffusion maps on song structure (this is my ISMIR late breaking demo this year...please see my GraphDitty <a href = "http://www.covers1000.net/GraphDitty/">demo</a> <a href = "https://github.com/ctralie/GraphDitty">code</a>, and <a href = "http://www.covers1000.net/ctralie2018_GraphDitty.pdf">abstract</a>  NOTE that GraphDitty is a submodule of this repo).  Below is a plot of the coordinates of diffusion maps for Alien Ant Farm's "Smooth Criminal" cover:


<img src = "http://www.ctralie.com/Research/FaceJam_HAMR2018/therock_AAF_4Components_MaxNormPerComponent.avi_DiffusionMaps.png">

Verse/chorus separation is clearly visible in the second component, and there is some finer structure in the other components.  I put the diffusion maps coordinates in the PCA space of facial landmarks, add on an additional vertical eyebrow displacement depending on how close we are to a beat onset, do the warp, and synchronize the result to music.  And that's it!
