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
_MY_COLOR = (210, 170, 60)
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
        new_mask = cv.dilate(pre, kernel, iterations=1)
        new_mask = cv.bitwise_xor(new_mask, pre)

        hl_image[:] = color
        sd_mask = cv.bitwise_and(hl_image, hl_image, mask=new_mask)
            
        x = x * blending_value
        result_x = cv.addWeighted(result_x, 1.0, sd_mask, x, 0.0)

    return result_x

def make_single_lightning_mask():
    blank = np.zeros(_DIMENSION[:2], np.uint8)
    start_point = (250,50)
    x, y = start_point[:2]
    
    pre = start_point
    while y < 500:
        x = x + random.randint(-50, 50)
        y = y + random.randint(20, 50)
        point = (x, y)
        cv.line(blank, pre, point, (255,255,255), 1)
        pre = point
        
    return blank

def make_lightning_bolts_mask():
    blank = np.zeros(_DIMENSION[:2], np.uint8)
    
    cnt = random.randint(3, 5)
    
    while cnt > 0:
        start_point = (250,50)
        x, y = start_point[:2]
        
        pre = start_point
        while y < 500:
            x = x + random.randint(-30, 30)
            y = y + random.randint(20, 30)
            point = (x, y)
            cv.line(blank, pre, point, (255,255,255), 1)
            pre = point
        
        cnt -= 1
        
    return blank

def lightning_effect(save_dir, color):
    blank = np.zeros(_DIMENSION, np.uint8)
    mask = make_lightning_bolts_mask()

    blank = draw_shadow(blank, mask, _COLOR[color], 10, 0.75, 0.75)

    save_image(save_dir, 'lightning_bolt', blank)

    cv.imshow('blank', blank)
    cv.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single lightning')
    parser.add_argument('--sav', help='name of saved directory', default=_SAVE_DIR, type=str)
    parser.add_argument('--color', help="color of light", default=1,choices=[0,1,2], type=int)

    args = parser.parse_args()
    save_dir = find_file(args.sav)
    
    make_dir(save_dir)
    
    lightning_effect(save_dir, args.color)
