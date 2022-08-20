try:
    import cv2 as cv
    import mediapipe as mp
    import os
    import numpy as np
    import argparse
except Exception as e:
    print('Caught error while importing {}'.format(e))

_IMAGE_PATH = '\\Photos\\long.jpg'
_SAVE_DIR = 'Shadow Saved Image'
_HIGHTLIGHT_COLOR = (255,255,255) #white
_SHADOW_COLOR_RED = (0,0,255) #red
_SHADOW_COLOR_BLUE = (255,0,0) #blue
_SHADOW_COLOR_GREEN = (0,255,0) #green

_COLOR = [_SHADOW_COLOR_RED, _SHADOW_COLOR_BLUE, _SHADOW_COLOR_GREEN]

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_file(name):
    current_path = __file__.replace(__file__.split('\\')[-1], name)
    return current_path

def resized_image(image, scale = 500):
    height, width = image.shape[:2]
    x = scale / width
    return cv.resize(image, (scale, int(height * x)))

def sefile_segmentation(image):
    height, width = image.shape[:2]
    #selfie segmentation model
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        results = selfie_segmentation.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        
        condition = np.stack((results.segmentation_mask,)*3, axis=-1) > 0.45
        
        fg_image = np.zeros(image.shape, np.uint8)
        fg_image[:] = (255,255,255)
        bg_image = np.zeros(image.shape, np.uint8)
        
        output_image = np.where(condition, fg_image, bg_image)
        _, image_designed = cv.threshold(cv.cvtColor(output_image, cv.COLOR_BGR2GRAY), 100, 255, cv.THRESH_BINARY)  
        new_image = cv.bitwise_and(image, image, mask=image_designed)
        return new_image

def draw_shadow(image, mask, color, shading_value, blending_value):
    not_mask = cv.bitwise_not(mask)
    image_not_mask = cv.bitwise_and(image, image, mask=not_mask)
    
    hl_image = np.zeros(image.shape, np.uint8)
    hl_image[:] = _HIGHTLIGHT_COLOR
    hl_mask = cv.bitwise_and(hl_image, hl_image, mask=mask)
    
    result = cv.add(image_not_mask, hl_mask)
    
    result_x = result
    new_mask = mask
    x = shading_value
    for i in range(2, 10, 1):
        pre = new_mask
        kernel = np.ones((i,i), np.uint8)
        new_mask = cv.dilate(pre, kernel, iterations=1)
        new_mask = cv.bitwise_xor(new_mask, pre)

        hl_image[:] = _COLOR[color]
        sd_mask = cv.bitwise_and(hl_image, hl_image, mask=new_mask)
            
        x = x * blending_value
        result_x = cv.addWeighted(result_x, 1.0, sd_mask, x, 0.0)

    not_mask = cv.bitwise_not(mask)
    image_not_mask = cv.bitwise_and(result_x, result_x, mask=not_mask)
    
    hl_image[:] = _HIGHTLIGHT_COLOR
    hl_mask = cv.bitwise_and(hl_image, hl_image, mask=mask)
    
    result_x = cv.add(image_not_mask, hl_mask)
    return result_x

def glow_effect_image(image_path, save_path, name, color):

    img = cv.imread(image_path)
    new_img = sefile_segmentation(img)
    edges = get_contours(new_img)
    
    blank = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    
    #processing
    img = draw_shadow(img, edges, color, 0.75, 0.75)
    blank = draw_shadow(blank, edges, color, 0.75, 0.75)
    #write image
    cv.imwrite(save_path + '\\' + name.split('\\')[-1], img)
    cv.imwrite(save_path + '\\black_' + name.split('\\')[-1], blank)
    #show image
    cv.imshow('blank', resized_image(blank))
    cv.imshow('image', resized_image(img))
    cv.waitKey(0)

def get_contours(image):
    blank = np.zeros(image.shape[:2], np.uint8)
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
     
    _, image_designed = cv.threshold(img_gray, 100, 255, cv.THRESH_BINARY)
    #detect contours
    contours, _ = cv.findContours(image_designed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #draw
    cv.drawContours(blank, contours, -1, (255,255,255), 1, cv.LINE_AA)
    
    return blank

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='glow effect')
    parser.add_argument('--sav', help='name of saved directory', default=_SAVE_DIR, type=str)
    parser.add_argument('-img', '--image', help="name of image", default=_IMAGE_PATH, type=str)
    parser.add_argument('--color', help="color of light", default=1,choices=[0,1,2], type=int)

    args = parser.parse_args()
    image_path = find_file(args.image)
    save_path = find_file(args.sav)
    
    make_dir(save_path)
    
    glow_effect_image(image_path, save_path, args.image, args.color)