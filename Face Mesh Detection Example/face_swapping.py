try:
    import cv2 as cv
    import numpy as np
    import dlib
    import argparse
    import os
except Exception as e:
    print('Caught error while importing: {}'.format(e))
    
SAVE_DIR = '/Face Mesh Detection Example/face_swapped'
SRC_IMAGE = '/Face Mesh Detection Example/Photos/long.jpg'
DST_IMAGE = '/Face Mesh Detection Example/Photos/man_2.jpg'
DAT_FILE = '/Face Mesh Detection Example/shape_predictor_68_face_landmarks.dat'

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
   
def resized_image(image, scale=500):
    height, width = image.shape[:2]
    x = scale/width
    return cv.resize(image, (scale, int(height * x)))

def draw_circle(image, x, y):
    return cv.circle(image, (x, y), 1, (0,255,0), 1)

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def FaceSwap(image, image2, save_dir):
    
    make_dir(save_dir)
    
    #convert to gray image
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    #create blank to get face in source image
    img2_new_face = np.zeros(image2.shape, np.uint8)
    
    # load face detector and face landmarks predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DAT_FILE)
    
    #Find landmark points of source image
    faces = detector(img_gray)
    
    landmarks_points = []
    for face in faces:
        landmarks = predictor(img_gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
            
            #draw face landmark points
            # cv.circle(image, (x,y), 3, (0,0,255), -1)
    points = np.array(landmarks_points, np.int32)
    convexhull = cv.convexHull(points)
        
    # delaunay triangulation
    rect = cv.boundingRect(convexhull)
    subdiv = cv.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangle = subdiv.getTriangleList()
    triangle = np.array(triangle).astype(int)
    
    #get the landmark points indexes of each triangle
    indexes_triangles = []
    for t in triangle:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
        
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)
        
    #Find landmark points of destination image   
    faces2 = detector(img2_gray)
    
    landmarks_points2 = []
    for face in faces2:
        landmarks = predictor(img2_gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points2.append((x, y))
    
    points2 = np.array(landmarks_points2, np.int32)
    convexhull2 = cv.convexHull(points2)
    
    #Triangulation of bot faces
    for triangle_index in indexes_triangles:
        #triangulation of the first face
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
        
        rect1 = cv.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = image[y:y+h, x:x+w]
        cropped_tr1_mask = np.zeros((h,w), np.uint8)
        
        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                          [tr1_pt2[0] - x, tr1_pt2[1] - y],
                          [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
        
        cv.fillConvexPoly(cropped_tr1_mask, points, 255)
        cropped_triangle = cv.bitwise_and(cropped_triangle, cropped_triangle,
                                          mask=cropped_tr1_mask)
        
        #triangulation of second face
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        rect2 = cv.boundingRect(triangle2)
        (x2, y2, w2, h2) = rect2
        cropped_triangle2 = image2[y2:y2+h2, x2:x2+w2]
        cropped_tr2_mask = np.zeros((h2,w2), np.uint8)
        
        points2 = np.array([[tr2_pt1[0] - x2, tr2_pt1[1] - y2],
                          [tr2_pt2[0] - x2, tr2_pt2[1] - y2],
                          [tr2_pt3[0] - x2, tr2_pt3[1] - y2]], np.int32)
        
        cv.fillConvexPoly(cropped_tr2_mask, points2, 255)
        cropped_triangle2 = cv.bitwise_and(cropped_triangle2, cropped_triangle2,
                                          mask=cropped_tr2_mask)
       
        #warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv.getAffineTransform(points, points2)
        warped_triangle = cv.warpAffine(cropped_triangle, M, (w2, h2), None, flags=cv.INTER_NEAREST, borderMode=cv.BORDER_REFLECT_101)
        warped_triangle = cv.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)
        
        # reconstruct destination face
        img2_new_face_rect_area = img2_new_face[y2:y2+h2, x2:x2+w2]
        img2_new_face_rect_area_gray = cv.cvtColor(img2_new_face_rect_area, cv.COLOR_BGR2GRAY)
    
        #remove the lines between the triangles
        _, mask_triangles_designed = cv.threshold(img2_new_face_rect_area_gray, 1, 255, cv.THRESH_BINARY_INV)
        warped_triangle = cv.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y2:y2+h2, x2:x2+w2] = img2_new_face_rect_area

        #face swapped
        img2_face_mask = np.zeros(img2_gray.shape[:2], np.uint8)
        img2_head_mask = cv.fillConvexPoly(img2_face_mask, convexhull2, 255)
        img2_face_mask = cv.bitwise_not(img2_head_mask)
        img2_head_noface = cv.bitwise_and(image2, image2, mask=img2_face_mask)
        
        #result
        result = cv.add(img2_head_noface, img2_new_face)
        
        #seamless cloning
        (x, y, w, h) = cv.boundingRect(convexhull2)
        center_face2 = (int((x+x+w)/2), int((y+y+h)/2))
        
        final_result = cv.seamlessClone(result, image2, img2_head_mask,
                                    center_face2, cv.NORMAL_CLONE)
    #save the result image
    cv.imwrite(save_dir + '/face_swap_image_68_landmark_points.jpg', final_result) 
    #show the result image
    cv.imshow('source image', resized_image(image))
    cv.imshow('destination image', resized_image(image2))
    cv.imshow('result image', resized_image(final_result))
    cv.waitKey(0) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='swapping image face 68 face lanmark points')
    parser.add_argument('--sav', help='save dir', default=SAVE_DIR, type=str)
    parser.add_argument('-src', '--source-image', help="face to swap", default=SRC_IMAGE, type=str)
    parser.add_argument('-dst', '--destination-image', help='background to swap', default=DST_IMAGE, type=str)
    args = parser.parse_args()
    
    current_path = os.getcwd()
    
    #take two image
    #source image
    img1 = cv.imread(current_path + args.source_image)
    #destination image
    img2 = cv.imread(current_path + args.destination_image)

    #swapping face
    FaceSwap(img1, img2, current_path + args.sav)
 