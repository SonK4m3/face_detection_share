try:
    import cv2 as cv
    import mediapipe as mp
    import numpy as np
    import os
    import argparse
except Exception as e:
    print('Caught error while importing: {}'.format(e))
    
IMAGE_PATH = '/Photos/DSC_6730.JPG'
SAVE_DIR = '/Line Face Save Image'

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def resized_image(image, scale = 500):
    height, width = image.shape[:2]
    x = scale / width
    return cv.resize(image, (scale, int(height * x)))

def draw_line(image, points, positions):
    start = points[0]
    for point in points:
        if(point != start):
            cv.line(image, positions[start], positions[point], (255, 255, 255), 2)
            start = point

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
  
def get_face_mesh(image_path, xmin, ymin, width, height, save_image):
        
    #import model face mesh
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(
        static_image_mode = True, 
        max_num_faces = 3, 
        refine_landmarks = True, 
        min_detection_confidence = 0.5) as face_mesh:
        image = cv.imread(image_path)
        image = image[ymin:ymin+height, xmin:xmin+width]
       
        face_mask = np.zeros(image.shape[:2], dtype='uint8')
        
        results = face_mesh.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return 
        
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            landmark_points = face_landmarks.landmark
            positions = []
            for xyz in landmark_points:
                x = xyz.x * width
                y = xyz.y * height
                positions.append((int(x), int(y)))    
                # cv.circle(annotated_image, (int(x), int(y)), 1, (0,255,0), -1)

            lip = [272, 271, 268, 13, 38, 41, 42, 78, 95, 88, 178, 87, 15, 316, 403, 319, 325, 307, 375, 321, 405, 314, 17,
                   84, 181, 91, 146, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 324, 
                   318, 402, 317, 15, 16, 17, 18, 200]
            
            face = [199, 175, 152, 148, 176, 149, 150, 136, 172, 132, 93, 234, 127, 162, 21, 54, 
                    103, 67, 109, 10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1]
            
            left_eye = [370, 462, 250, 305, 326, 327, 294, 331, 279, 360, 363, 456, 399, 412, 465, 
                        464, 463, 385, 476, 380, 374, 474, 386, 387, 388, 466, 263, 359, 255, 254, 
                        253, 256, 341, 453, 452, 451, 450, 449, 448, 261]
            
            right_eye = [0, 164, 2, 141, 97, 99, 60, 75, 240, 64, 129, 49, 131, 134, 236, 174, 188,
                         245, 133, 173, 157, 158, 469, 153, 145, 471, 159, 160, 161, 246, 33, 130, 
                         25, 110, 24, 23, 22, 26, 112, 233, 232, 231, 230, 229, 228]
            
            brown_right_eye = [113, 225, 224, 223, 222, 221, 189, 55, 107, 66, 105, 63, 70, 
                               46, 53, 52, 65, 55]
            
            brown_left_eye = [300, 276, 283, 282, 295, 285, 336, 296, 334, 293, 300, 389, 356, 
                              454, 366, 401, 435, 397, 365, 379, 378, 400, 377]
            
            top_left_eye = [413, 441, 442, 443, 444, 445]
            
            points = lip + face + left_eye
            
            draw_line(annotated_image, points, positions)
            draw_line(annotated_image, right_eye, positions)
            draw_line(annotated_image, brown_right_eye, positions)
            draw_line(annotated_image, brown_left_eye, positions)
            draw_line(annotated_image, top_left_eye, positions)

            draw_line(face_mask, points, positions)
            draw_line(face_mask, right_eye, positions)
            draw_line(face_mask, brown_right_eye, positions)
            draw_line(face_mask, brown_left_eye, positions)
            draw_line(face_mask, top_left_eye, positions)
        #save image    
        cv.imwrite(save_image + '/color_image.jpg', annotated_image)
        cv.imwrite(save_image + '/gray_image.jpg', face_mask)
        #show image    
        cv.imshow('a', resized_image(annotated_image))
        cv.imshow('b', resized_image(face_mask))
        cv.waitKey(0)

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
    parser = argparse.ArgumentParser(description='draw line in face')
    parser.add_argument('-src', '--source-image', help="face is drawed", default=IMAGE_PATH, type=str)
    parser.add_argument('--sav', help='save dir', default=SAVE_DIR, type=str)
    args = parser.parse_args()
    
    face_detections = get_face_detection(args.source_image)

    current_path = os.getcwd()

    make_dir(current_path + args.sav)
    
    for num, face_detection in enumerate(face_detections):
        x, y, w, h = face_detection
        
        save_image = current_path + args.sav + '/' + get_name_image(args.source_image) + '/' + 'image_' + str(num)
        
        make_dir(save_image)
        
        get_face_mesh(current_path + args.source_image, x - 50, y - 50, w + 100, h + 100, save_image)