# **Glow Effect**

## Overview

Creates a shadow effect of a given set of strokes

The effect is created by drawing multiple strokes with each time getting bigger and darker

    - get selfie segmentation
    - find contours
    - change background color 
    - draw shadow for contours

## Example 

![girl_and_flower](./Example%20Image/girl_and_flower.jpg)

## Requirements
```
opencv
mediapipe
numpy
```

## How to run

- run following command:
```
py glow_effect.py [-h] [--sav SAV] [-img IMAGE] [--bg_cl {0,1,2}] [--sd_cl {0,1,2}] [--mode MODE]
```
