try:
    import cv2 as cv
    import numpy as np
    import mediapipe as mp
    import os
    import argparse
except Exception as e:
    print('Caught error while importing: {}'.format(e))

IMAGE_PATH = '/Face Mesh Detection Example/Photos/DSC_6730.JPG'
FLAG_PATH = '/Face Mesh Detection Example/Photos/vietnam_flag.jpg'
SAVE_DIR = '/Face Mesh Detection Example/Stick Flag Save Image'

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def resized_image(image, scale = 500):
    height, width = image.shape[:2]
    x = scale / width
    return cv.resize(image, (scale, int(height * x)))

def get_face_detection(image_path):
    mp_face_detection = mp.solutions.face_detection
    
    location_face_detection = []
    
    with mp_face_detection.FaceDetection(model_selection = 1, min_detection_confidence = 0.1) as face_detection:
        image = cv.imread(image_path)
        height, width = image.shape[:2]
        
        results = face_detection.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        
        if not results.detections:
            return []
        
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            location_face_detection.append([x,y,w,h])

        return location_face_detection
  
def get_face_mesh(image_path, xmin, ymin, width, height, flag_path, save_image):
        
    #import model face mesh
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(
        static_image_mode = True, 
        max_num_faces = 3, 
        refine_landmarks = True, 
        min_detection_confidence = 0.5) as face_mesh:
        image = cv.imread(image_path)
        image = image[ymin:ymin+height, xmin:xmin+width]
        
        flag = cv.imread(flag_path)
        flag = cv.resize(flag, (image.shape[1], image.shape[0]))
        
        mask = np.zeros(flag.shape[:2], dtype='uint8')
        
        face_mask = np.zeros(flag.shape[:2], dtype='uint8')
        right_eye_mask = np.zeros(flag.shape[:2], dtype='uint8')
        left_eye_mask = np.zeros(flag.shape[:2], dtype='uint8')
        lips_mask = np.zeros(flag.shape[:2], dtype='uint8')
        
        results = face_mesh.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return 
        
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            #get face landmark point 
            face_landmark_points = get_landmark_points(
                                image=annotated_image,
                                face_landmarks=face_landmarks, 
                                mask=mp_face_mesh.FACEMESH_FACE_OVAL)

            face_points = np.array(face_landmark_points, np.int32)
            face_convexhull = cv.convexHull(face_points)
            cv.fillConvexPoly(face_mask, face_convexhull, 255)

            # eyes
            right_eye_landmark_points = get_landmark_points(
                                image=annotated_image,
                                face_landmarks=face_landmarks, 
                                mask=mp_face_mesh.FACEMESH_RIGHT_EYE)

            right_eye_points = np.array(right_eye_landmark_points, np.int32)
            right_eye_convexhull = cv.convexHull(right_eye_points)
            cv.fillConvexPoly(right_eye_mask, right_eye_convexhull, 255)
            
            left_eye_landmark_points = get_landmark_points(
                                image=annotated_image,
                                face_landmarks=face_landmarks, 
                                mask=mp_face_mesh.FACEMESH_LEFT_EYE)

            left_eye_points = np.array(left_eye_landmark_points, np.int32)
            left_eye_convexhull = cv.convexHull(left_eye_points)
            cv.fillConvexPoly(left_eye_mask, left_eye_convexhull, 255)
            
            #lips
            lips_landmark_points = get_landmark_points(
                                image=annotated_image,
                                face_landmarks=face_landmarks, 
                                mask=mp_face_mesh.FACEMESH_LIPS)

            lips_points = np.array(lips_landmark_points, np.int32)
            lips_convexhull = cv.convexHull(lips_points)
            cv.fillConvexPoly(lips_mask, lips_convexhull, 255)
            #mask
            mask = cv.bitwise_or(mask, face_mask)
            mask = cv.bitwise_xor(mask, right_eye_mask)
            mask = cv.bitwise_xor(mask, left_eye_mask)
            mask = cv.bitwise_xor(mask, lips_mask)
            #next step
            stick_flag_into_face(image, mask, flag, save_image) 

def stick_flag_into_face(image, mask, flag, save_image):
    #only face mask
    only_face_mask = cv.bitwise_and(image, image, mask=mask)
            
    #no face
    img_head_mask = cv.bitwise_not(mask)
    img_noface = cv.bitwise_and(image, image, mask=img_head_mask)
    
    #face flag
    flag_img = np.zeros(image.shape, np.uint8)
    flag_img = cv.bitwise_and(flag, flag, mask=mask)
    
    #blend
    alpha = 0.5
    beta = 1 - alpha
    new_face = cv.addWeighted(only_face_mask, alpha, flag_img, beta, 0.0)
    
    #add 
    result = cv.add(img_noface, new_face)
    result2 = cv.add(img_noface, flag_img)
    
    monoMaskImage = cv.split(mask)[0] # reducing the mask to a monochrome
    br = cv.boundingRect(monoMaskImage) # bounding rect (x,y,width,height)
    centerOfBR = (br[0] + br[2] // 2, br[1] + br[3] // 2)

    final_result = cv.seamlessClone(result2, result, mask, centerOfBR, cv.MIXED_CLONE)
    
    #save image
    name = save_image + '/' + 'only_face_mask' + '.jpg'
    cv.imwrite(name, only_face_mask)
    name = save_image + '/' + 'flag_img' + '.jpg'
    cv.imwrite(name, flag_img)
    name = save_image + '/' + 'new_face' + '.jpg'
    cv.imwrite(name, new_face)
    name = save_image + '/' + 'img_noface' + '.jpg'
    cv.imwrite(name, img_noface)
    name = save_image + '/' + 'result' + '.jpg'
    cv.imwrite(name, result)
    name = save_image + '/' + 'result2' + '.jpg'
    cv.imwrite(name, result2)
    name = save_image + '/' + 'final_result' + '.jpg'
    cv.imwrite(name, final_result)

    #show image
    cv.imshow('only', only_face_mask)
    cv.imshow('flag', flag_img)
    cv.imshow('new', new_face)
    cv.imshow('no', img_noface)
    cv.imshow('final', result)
    cv.imshow('final 2', result2)
    cv.imshow('final s', final_result)

    cv.waitKey(1000)

def get_landmark_points(image, face_landmarks, mask):
    list_x, list_y = zip(*mask)
    list_mask = list(set(list_x + list_y))
    height, width = image.shape[:2]
    
    landmark_points = list([(face_landmarks.landmark[ind].x * width, face_landmarks.landmark[ind].y * height) for ind in list_mask])
    return landmark_points

def get_name_image(image_dir):
    name = image_dir.split('/')[-1]
    return name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stick flag to face')
    parser.add_argument('-src', '--source-image', help="face is sticked", default=IMAGE_PATH, type=str)
    parser.add_argument('-flag', help="image to stick into face", default=FLAG_PATH, type=str)
    parser.add_argument('--sav', help='save dir', default=SAVE_DIR, type=str)
    args = parser.parse_args()
    
    current_path = os.getcwd()
    
    face_detections = get_face_detection(current_path + args.source_image)
    #debug
    # print(face_detections)
    
    make_dir(current_path + args.sav)
    
    for num, face_detection in enumerate(face_detections):
        x, y, w, h = face_detection
        #debug
        # print("{} {} {} {}".format(x, y, w, h))
        save_image = current_path + args.sav + '/' + get_name_image(current_path + args.source_image) + '/' + 'image_' + str(num)
        make_dir(save_image)
        get_face_mesh(current_path + args.source_image, x, y, w, h, current_path + args.flag, save_image)
