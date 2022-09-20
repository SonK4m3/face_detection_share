# **Single Lightning**

## Overview
This effect is created by many lines connecting each other and drawing shadows

## Example
![single lightning](./Example%20Image/single_lightning15_51_31.png)
![lightning bolts](./Example%20Image/single_lightning_effect_14_42_02.png)
![bigger lightning](./Example%20Image/bigger_lightning_23_09_14.png)
![bigger lightning in black image](./Example%20Image/single_lightning_effect_14_41_43.png)

## How to run it
- default saved image directory is `Single Lightning Saved Image` will be make when you run code
- run following command:
```
py single_lightning_effect.py
```
- you can select another options by change arguments in command
```
usage: single_lightning_effect.py [-h] [--sav SAV] [--color {0,1,2}] [--mode {1,2,3,4}] [--start START [START ...]] [--end END [END ...]] [--thickness THICKNESS]

single lightning

options:
  -h, --help            show this help message and exit
  --sav SAV             name of saved directory
  --color {0,1,2}       color of light
  --mode {1,2,3,4}      mode of light
  --start START [START ...]
                        header of the lightning
  --end END [END ...]   tail of the lightning
  --thickness THICKNESS
                        width of the lightning
