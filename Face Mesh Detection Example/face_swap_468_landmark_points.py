try:
    import cv2 as cv
    import mediapipe as mp
    import numpy as np
    import argparse
    import os
except Exception as e:
    print('Caught error while importing: {}'.format(e))
    
SAVE_DIR = '/Face Mesh Detection Example/face_swapped'
SRC_IMAGE = '/Face Mesh Detection Example/Photos/long.jpg'
DST_IMAGE = '/Face Mesh Detection Example/Photos/man_2.jpg'

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def get_face_mesh(img, face_mesh):
    results = face_mesh.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return []
    
    for face_landmark in results.multi_face_landmarks:
        landmark_points = []
        #we get points location of face mesh in 2D model
        for xyz in face_landmark.landmark:
            x = xyz.x * img.shape[1]
            y = xyz.y * img.shape[0]
            landmark_points.append((x, y))
    points = np.array(landmark_points, np.int32)
    convexhull = cv.convexHull(points)
    
    return [landmark_points, points, convexhull]    
        

def face_swap(src, dts, save_dir):
    
    make_dir(save_dir)
    #read 2 images
    img = cv.imread(src)
    img2 = cv.imread(dts)
    #new image to get face from source image
    img2_new_face = np.zeros(img2.shape, np.uint8)
    
    #model face mesh
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.1) as face_mesh:
        #get face mesh landmark points
        landmark_points, points, convexhull = get_face_mesh(img, face_mesh)
        landmark_points2, points2, convexhull2 = get_face_mesh(img2, face_mesh)

        #delauany triangulation
        rect = cv.boundingRect(convexhull)
        subdiv = cv.Subdiv2D(rect)
        subdiv.insert(landmark_points)
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

        #Triangulation of both faces
        for triangle_index in indexes_triangles:
            #triangulation of the first face
            #location of 3 vertices
            tr1_pt1 = landmark_points[triangle_index[0]]
            tr1_pt2 = landmark_points[triangle_index[1]]
            tr1_pt3 = landmark_points[triangle_index[2]]
            #implement triangle by array
            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
            
            rect1 = cv.boundingRect(triangle1)
            (x, y, w, h) = rect1
            cropped_triangle = img[y:y+h, x:x+w]
            cropped_tr1_mask = np.zeros((h,w), np.uint8)    #make blank to get triangle
            
            points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                            [tr1_pt2[0] - x, tr1_pt2[1] - y],
                            [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
            
            cv.fillConvexPoly(cropped_tr1_mask, points, 255)    #draw triangle into blank
            cropped_triangle = cv.bitwise_and(cropped_triangle, cropped_triangle,
                                            mask=cropped_tr1_mask)  # get image in triangle

            #triangulation of second face
            tr2_pt1 = landmark_points2[triangle_index[0]]
            tr2_pt2 = landmark_points2[triangle_index[1]]
            tr2_pt3 = landmark_points2[triangle_index[2]]
            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

            rect2 = cv.boundingRect(triangle2)
            (x2, y2, w2, h2) = rect2
            cropped_triangle2 = img2[y2:y2+h2, x2:x2+w2]
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
            M = cv.getAffineTransform(points, points2)  # transform triangle 1 shape to triangle 2 shape
            # we use flag INTER_NEAREST to fill black pixels
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

            # #face swapped
            img2_face_mask = np.zeros(img2.shape[:2], np.uint8)
            img2_head_mask = cv.fillConvexPoly(img2_face_mask, convexhull2, 255)
            img2_face_mask = cv.bitwise_not(img2_head_mask)
            img2_head_noface = cv.bitwise_and(img2, img2, mask=img2_face_mask)
            # #result
            result = cv.add(img2_head_noface, img2_new_face)
            
            #seamless cloning
            (x, y, w, h) = cv.boundingRect(convexhull2)
            center_face2 = (int((x+x+w)/2), int((y+y+h)/2))
            
            final_result = cv.seamlessClone(result, img2, img2_head_mask,
                                        center_face2, cv.NORMAL_CLONE)
         #save the result image
        cv.imwrite(save_dir + '/face_swap_image_68_landmark_points.jpg', final_result) 
        #show image
        cv.imshow('1', final_result)
        cv.waitKey(0)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='swapping image face 684 face landmark points')
    parser.add_argument('--sav', help='save dir', default=SAVE_DIR, type=str)
    parser.add_argument('-src', '--source-image', help="face to swap", default=SRC_IMAGE, type=str)
    parser.add_argument('-dst', '--destination-image', help='background to swap', default=DST_IMAGE, type=str)
    args = parser.parse_args()
    
    current_path = os.getcwd()
    
    #process
    face_swap(current_path + args.source_image, current_path + args.destination_image, current_path + args.sav)
    