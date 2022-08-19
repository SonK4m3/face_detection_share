try:
    import cv2 as cv
    import mediapipe as mp
    import os
    import numpy as np
    import argparse
except Exception as e:
    print('Caught error while importing {}'.format(e))

_IMAGE_PATH = 'Photos\\doctor_strange.jpg'
_BACKGROUND_IMAGE_PATH = 'Photos\\river_in_the_cattle.jpg'
_SAVE_DIR = 'Change Background Saved Image'
_MASK_COLOR = (255,255,255) #white

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

def get_selfie_segmentation(image_path, background_image_path, save_path):
    
    make_dir(save_path)
    #image
    origin_image = cv.imread(image_path)
    height, width = origin_image.shape[:2]
    #background
    origin_background_image = cv.imread(background_image_path)
    background_image = cv.resize(origin_background_image, (width,height))
    #selfie segmentation model
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        results = selfie_segmentation.process(cv.cvtColor(origin_image, cv.COLOR_BGR2RGB))
        
        condition = np.stack((results.segmentation_mask,)*3, axis=-1) > 0.4
        
        fg_image = np.zeros(origin_image.shape, np.uint8)
        fg_image[:] = _MASK_COLOR
        bg_image = np.zeros(origin_image.shape, np.uint8)
        
        output_image = np.where(condition, fg_image, bg_image)
        
        _, image_designed = cv.threshold(cv.cvtColor(output_image, cv.COLOR_BGR2GRAY), 100, 255, cv.THRESH_BINARY)
        image = cv.bitwise_and(origin_image, origin_image, mask=image_designed)
        image_designed = cv.bitwise_not(image_designed)
        background_image_mask = cv.bitwise_and(background_image, background_image, mask=image_designed)
        image = cv.add(image, background_image_mask)
        
        #wirte iamge
        cv.imwrite(save_path + '\\long_is_on_the_river.jpg', image)
        #show image
        cv.imshow('origin', resized_image(image))
        cv.waitKey(1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='change background')
    parser.add_argument('--sav', help='name of saved directory', default=_SAVE_DIR, type=str)
    parser.add_argument('-img', '--image', help="name of image", default=_IMAGE_PATH, type=str)
    parser.add_argument('-bg', '--background-image', help="name of background image", default=_BACKGROUND_IMAGE_PATH, type=str)
    args = parser.parse_args()
    
    image_path = find_file(args.image)
    background_path = find_file(args.background_image)
    save_path = find_file(args.sav)
    
    get_selfie_segmentation(image_path, background_path, save_path)