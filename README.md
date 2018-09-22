# FaceJam (<a href = "https://labrosa.ee.columbia.edu/hamr_ismir2018/">HAMR 2018</a>)

The goal of this 2 day hackathon was given a song and an image with a face in it, to make a program that automatically detects the face and makes a music video in which the face's eyebrows move to the beat, and in which the face changes its expression depending where we are in the song (e.g. verse vs chorus).  An example is below animating the face of <a href = "https://en.wikipedia.org/wiki/Dwayne_Johnson">Dwayne Johnson ("The Rock")</a> to go along with <a href = "https://www.youtube.com/watch?v=CDl9ZMfj6aE">Alien Ant Farm's Smooth Criminal</a>:


[![Alt text](https://img.youtube.com/vi/VnCy7NGGN-3U/0.jpg)](https://www.youtube.com/watch?v=nCy7NGGN-3U)

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
~~~~~ bash

to see more options, including number of threads to make it faster on a multicore machine, please type
~~~~~ bash
 python FaceJam.py --help
~~~~~ bash

## Algorithm / Development

Below I will describe some of the key steps of the algorithm
