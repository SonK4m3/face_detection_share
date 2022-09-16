import math


try:
    import cv2 as cv
    import os
    import numpy as np
    import argparse
    from datetime import datetime
    import random
except Exception as e:
    print('Caught error while importing {}'.format(e))

_SAVE_DIR = 'Single Lightning Effect Saved Image'

_HIGHTLIGHT_COLOR = (255,255,255) #white
_SHADOW_COLOR_RED = (0,0,255) #red
_SHADOW_COLOR_BLUE = (255,0,0) #blue
_SHADOW_COLOR_GREEN = (0,255,0) #green

_COLOR = [_SHADOW_COLOR_RED, _SHADOW_COLOR_BLUE, _SHADOW_COLOR_GREEN]

_DIMENSION = [500,500,3]

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_file(name):
    current_path = __file__.replace(__file__.split('\\')[-1], name)
    return current_path

def resized_image(image, scale = 300):
    height, width = image.shape[:2]
    x = scale / width
    return cv.resize(image, (scale, int(height * x)))

def save_image(save_dir, save_name, image):
    now = datetime.now()
    file_name = save_dir + '/' + save_name + now.strftime('_%H_%M_%S') + '.png'
    cv.imwrite(file_name, image)

def draw_shadow(image, mask, color, size, shading_value, blending_value):
    not_mask = cv.bitwise_not(mask)
    image_not_mask = cv.bitwise_and(image, image, mask=not_mask)
    
    hl_image = np.zeros(image.shape, np.uint8)
    hl_image[:] = _HIGHTLIGHT_COLOR
    hl_mask = cv.bitwise_and(hl_image, hl_image, mask=mask)
    
    result = cv.add(image_not_mask, hl_mask)
    
    result_x = result
    new_mask = mask
    x = shading_value
    for i in range(2, size, 1):
        pre = new_mask
        kernel = np.ones((i,i), np.uint8)
        new_mask_d = cv.dilate(pre, kernel, iterations=1)
        new_mask_e = cv.erode(pre, kernel, iterations=1)

        new_mask_d = cv.bitwise_xor(new_mask_d, pre)
        new_mask_e = cv.bitwise_xor(new_mask_e, pre)
        new_mask = cv.add(new_mask_d, new_mask_e)

        hl_image[:] = color
        sd_mask = cv.bitwise_and(hl_image, hl_image, mask=new_mask)
        x = x * blending_value
        result_x = cv.addWeighted(result_x, 1.0, sd_mask, x, 0.9)

    return result_x

def make_single_lightning_mask():
    blank = np.zeros(_DIMENSION[:2], np.uint8)
    start_point = (250,50)
    x, y = start_point[:2]
    pre = start_point
    thickness = 3
    while y < 500:
        x = x + random.randint(-30, 30)
        y = y + random.randint(20, 40)
        point = (x, y)
        cv.line(blank, pre, point, (255,255,255), thickness)
        pre = point
        thickness = int(thickness * 0.9) if thickness > 1 else thickness
        
    return blank

def make_lightning_bolts_mask():
    blank = np.zeros(_DIMENSION[:2], np.uint8)
    
    cnt = random.randint(3, 5)
    thickness = 3
    while cnt > 0:
        start_point = (250,50)
        x, y = start_point[:2]
        
        pre = start_point
        while y < 500:
            x = x + random.randint(-30, 30)
            y = y + random.randint(20, 30)
            point = (x, y)
            cv.line(blank, pre, point, (255,255,255), thickness)
            pre = point
        
        cnt -= 1
        
    return blank

def make_bigger_lightning(dimension, start, end, thickness, bend=4):
    blank = np.zeros(dimension, np.uint8)
    
    # from center started point to make 2 left right started point 
    x_center,y_center = start[:2]
    
    start_point = (x_center - (thickness // 2), y_center)
    start_point_2 = (x_center + (thickness // 2), y_center)
    
    x, y = start_point[:2]
    x_2, y_2 = start_point_2[:2]    
    
    pre = start_point
    pre_2 = start_point_2
    
    depth = get_first_divisor(thickness) // 2
    
    cv.line(blank, start_point, start_point_2, (255,255,255), 1)
    while y < dimension[1]:
        x_ran = random.randint(-(thickness//2 + bend * 10), thickness//2 + bend * 10)
        y_ran = random.randint(dimension[1]//10 - 20, dimension[1]//10 - 10)
        
        # make next point with random position
        # if point is out of image, we reverse sign
        if x + x_ran + depth >= 0 and x_2+ x_ran + depth <= dimension[0]:
            x = x + x_ran + depth 
            x_2 = x_2 + x_ran - depth
        else: 
            x - x_ran + depth
            x_2 - x_ran + depth
        if(y + y_ran < dimension[1]):
            y = y + y_ran 
            y_2 = y_2 + y_ran 
        
        print("{} {}".format(x, y))
        
        if(x > x_2 and y < dimension[1]):
            break
        
        cv.line(blank, pre, (x, y), (255,255,255), 1)
        cv.line(blank, pre_2, (x_2, y_2), (255,255,255), 1)

        pre = (x, y)
        pre_2 = (x_2, y_2)
    
    blank_2 = np.zeros(dimension, np.uint8)
    contours,_ = cv.findContours(blank, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(blank_2, contours, -1, (255,255,255), cv.FILLED)
    
    # smooth line 
    # blank_2 = cv.medianBlur(blank_2, 5)
        
    return blank_2

def lightning_effect(save_dir, color):
    blank = np.zeros(_DIMENSION, np.uint8)
    # blank = cv.imread('./Blending/Photos/men_in_cities.jpg')
    # mask = make_lightning_bolts_mask()
    # mask = make_single_lightning_mask()
    mask = make_bigger_lightning(blank.shape[:2], (450,50), (360, 490), 20)
    
    blank = draw_shadow(blank, mask, _COLOR[color], 8, 0.7, 0.8)

    save_image(save_dir, 'bigger_lightning', blank)
    
    cv.imshow('blank', blank)
    cv.waitKey(0)

def get_first_divisor(n):
    if n < 2: return 1
    for i in range(2, n):
        if n % i == 0:
            return i

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single lightning')
    parser.add_argument('--sav', help='name of saved directory', default=_SAVE_DIR, type=str)
    parser.add_argument('--color', help="color of light", default=1,choices=[0,1,2], type=int)
    
    args = parser.parse_args()
    save_dir = find_file(args.sav)
    
    make_dir(save_dir)
    
    lightning_effect(save_dir, args.color)
