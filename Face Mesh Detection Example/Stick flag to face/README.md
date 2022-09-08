# **Stick flag to face**

## Overview 
This module will replace face by flag image or any photo if you want

## Example

![example image](./Example%20image/DSC_6730.JPG/image_1/result.jpg)

## Requirements
```
OpenCV
mediapipe
numpy
```

## How to run 
- usage:
```
usage: stick_flag_to_face.py [-h] [--src SRC] [--flag FLAG] [--sav SAV]

Stick flag to face

options:
  -h, --help   show this help message and exit
  --src SRC    face is sticked
  --flag FLAG  image to stick into face
  --sav SAV    save dir
```
- running following command:
```
py stick_flag_to_face.py
```
- default source image is `DSC_6730.JPG`
- default flag image is `vietnam_flag.jpg`
- default saved image dir is `Stick Flag Saved Image`
- if you want to change arguments, you need to add selection to the instructions above 

