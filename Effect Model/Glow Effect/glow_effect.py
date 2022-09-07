try:
    import cv2 as cv
    import mediapipe as mp
    import os
    import numpy as np
    import argparse
except Exception as e:
    print('Caught error while importing {}'.format(e))

_IMAGE_PATH = 'Photos\\girl_and_flower.jpg'
_SAVE_DIR = 'Shadow Saved Image'
_HIGHTLIGHT_COLOR = (255,255,255) #white
_SHADOW_COLOR_BLUE = (255,0,0) #blue
_SHADOW_COLOR_GREEN = (0,255,0) #green
_SHADOW_COLOR_RED = (0,0,255) #red

_COLOR = [_SHADOW_COLOR_RED, _SHADOW_COLOR_BLUE, _SHADOW_COLOR_GREEN]

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
# find current file main path to get image path
def find_file(name):
    current_path = __file__.replace(__file__.split('\\')[-1], name)
    return current_path

def resized_image(image, scale = 500):
    height, width = image.shape[:2]
    x = scale / width
    return cv.resize(image, (scale, int(height * x)))

def sefile_segmentation(image):
    #selfie segmentation model
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    # parameters to get selfie segmentation
    alpha = 0.45    # alpha is larger -> selfie segmentation is smaller 
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        # process
        results = selfie_segmentation.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        # condition to remove SS from result to blank image 
        condition = np.stack((results.segmentation_mask,)*3, axis=-1) > alpha
        # make 2 image white and black to get selfie segmentation
        human_image = np.zeros(image.shape, np.uint8)
        human_image[:] = (255,255,255)
        bg_image = np.zeros(image.shape, np.uint8)
        # output
        output_image = np.where(condition, human_image, bg_image)
        # get image follow segment
        _, image_designed = cv.threshold(cv.cvtColor(output_image, cv.COLOR_BGR2GRAY), 150, 255, cv.THRESH_BINARY) 
        not_image_designed = cv.bitwise_not(image_designed) 
        human_segment = cv.bitwise_and(image, image, mask=image_designed)
        background_segment = cv.bitwise_and(image, image, mask=not_image_designed)

        return [human_segment, background_segment, output_image]

# find largest contour area
def maximum_contour(contours):
    max = contours[0]
    for c in contours:
        if(cv.contourArea(c) > cv.contourArea(max)):
            max = c
    return max

# draw contours follow maximum-contour 
def get_contours_gray_max(image):
    blank = np.zeros(image.shape[:2], np.uint8)
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (5,5), 0)
    _, image_designed = cv.threshold(img_blur, 50, 255, cv.THRESH_BINARY)
    #detect contours
    contours, _ = cv.findContours(image_designed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #find maximum contour
    max = maximum_contour(contours)
    #draw
    cv.drawContours(blank, [max], 0, (255,255,255), 1, cv.LINE_AA)
    
    return blank

# find and draw contours
def get_contours(image):
    blank = np.zeros(image.shape[:2], np.uint8)
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (5,5), 0)
    _, image_designed = cv.threshold(img_blur, 50, 255, cv.THRESH_BINARY)
    #detect contours
    contours, _ = cv.findContours(image_designed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #draw
    cv.drawContours(blank, contours, -1, (255,255,255), 1, cv.LINE_AA)
    
    return blank   

# we draw shadow of mask, mask is gray image
# @pragma size: shadow width
# @pragma shading_value: shadow chroma
# @pragma blending_value: shadow spread
def draw_shadow(image, mask, color, size, shading_value, blending_value):
    not_mask = cv.bitwise_not(mask)
    # stick mask to image
    image_not_mask = cv.bitwise_and(image, image, mask=not_mask)
    # make mask and fill color
    hl_image = np.zeros(image.shape, np.uint8)
    hl_image[:] = _HIGHTLIGHT_COLOR
    hl_mask = cv.bitwise_and(hl_image, hl_image, mask=mask)
    
    result = cv.add(image_not_mask, hl_mask)
    # loop to make shadow
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

def get_bg_edges(human_edges, bg_edges):
    #remove human edge from background edge
    bg_edges[human_edges == 255] = 255
    
    return bg_edges
    
def glow_effect_image(image_path, save_path, name, bg_color, sd_color, mode):
    try:
        img = cv.imread(image_path)
    except Exception as e:
        print('Error while reading image: {}'.format(e))
        
    new_image = img
    
    human_segment, bg_segment, human_mask = sefile_segmentation(img)
    human_edges = get_contours_gray_max(human_mask)                                                       
    bg_edges = get_contours(bg_segment)
    bg_edges = get_bg_edges(human_edges, bg_edges)
    
    shading_value = 0.7
    blending_value = 0.7
    size = 8
    
    img = darking_background(img, _COLOR[bg_color], human_mask)
    #processing
    img = draw_shadow(img, human_edges,_COLOR[sd_color], size, shading_value, blending_value)
    #write image
    cv.imwrite(save_path + '\\' + name.split('\\')[-1], img)
    #show image
    cv.imshow('image', resized_image(img))
    
    if mode == 1:
        blank = np.zeros(img.shape, np.uint8)
        new_image = darking_background(new_image, _COLOR[bg_color], human_mask)
        blank = draw_shadow(new_image, bg_edges,_COLOR[sd_color], size, shading_value, blending_value)
        cv.imwrite(save_path + '\\black_' + name.split('\\')[-1], blank)
        cv.imshow('blank', resized_image(blank))
    
    cv.waitKey(0)

def darking_background(image, color, human_mask):
    img_gray = cv.cvtColor(human_mask, cv.COLOR_BGR2GRAY)
    _, image_designed = cv.threshold(img_gray, 20, 255, cv.THRESH_BINARY)
    human = cv.bitwise_and(image, image, mask=image_designed)
    
    img_blur = cv.GaussianBlur(image, (5,5), 1)
    img_darker = darking_image(img_blur, color, 200, 155, beta=0.1)
    
    bg_mask = cv.bitwise_not(image_designed)
    bg = cv.bitwise_and(img_darker, img_darker, mask=bg_mask)
    
    result = cv.add(human, bg)
    
    return result
    
def darking_image(image, color, brightness=255, contrast=127, beta=0.1):
    black = np.zeros(image.shape, np.uint8)
    black[:] = color
    
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            max = 255
        else:
            shadow = 0
            max = 255 + brightness
        al_pha = (max - shadow) / 255
        ga_mma = shadow
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv.addWeighted(image, al_pha, black, beta, ga_mma)
    else:
        cal = image
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv.addWeighted(cal, Alpha, black, beta, Gamma)
    
    return cal
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='glow effect')
    parser.add_argument('--sav', help='saved directory', default=_SAVE_DIR, type=str)
    parser.add_argument('-img', '--image', help="image", default=_IMAGE_PATH, type=str)
    parser.add_argument('--bg_cl', help="color of background", default=1,choices=[0,1,2], type=int)
    parser.add_argument('--sd_cl', help="color of light", default=1,choices=[0,1,2], type=int)
    parser.add_argument('--mode', help="make up for background", default=0, type=int)

    args = parser.parse_args()
    image_path = find_file(args.image)
    save_path = find_file(args.sav)
    
    make_dir(save_path)
    
    glow_effect_image(image_path, save_path, args.image, args.bg_cl, args.sd_cl, args.mode)