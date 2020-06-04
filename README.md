# **Real time attentiveness detection**
![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  
  
*Singapore India Hackathon 2019*  
  
A computer vision based solution that tracks attentivenss of students on E-Learning platforms in real time  
  
1) [Demonstration](#demonstration)
   1) [Requirements](#demonstration)
   2) [Steps](#demonstration)
2) [Attentiveness Features](#attentiveness)
   
<p align="center">
<img src="https://github.com/jainamshah17/attentiveness-detection/blob/master/media/demo.gif" width="80%"/>
</p>  
  
*🚀 This code uses [Gaze Tracking](https://github.com/antoinelame/GazeTracking) library, credits to [Antoine Lamé](https://github.com/antoinelame)*
  
## Demonstration
Clone this repository using terminal or download zip
```
git clone https://github.com/jainamshah17/attentiveness-detection.git
```
Before running the code, make sure you have installed all the required librarires
```
numpy == 1.16.1
opencv_python == 3.4.5.20
dlib == 19.16.0
```
Now execute detect.py python script from terminal
```
python detect.py
```
  
  ## Attentiveness
  1) Eye Gaze - Direction in which user is looking i.e. center / right / left ; from center of the screen
  2) Forehead Position - Position of user's forehead i.e. up / down / center ; from center of the screen
  3) Lips Distance - User is speaking / silent / yawning
    
  Based on above 3 estimations following things are done : attentiveness is scored, tells whether the user is feeling sleepy or not
