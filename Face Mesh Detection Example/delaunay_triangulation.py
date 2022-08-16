try: 
    import cv2 as cv
    import numpy as np
    import mediapipe as mp
    import argparse
    import os
except Exception as e:
    print('Caught error while importing: {}'.format(e))

IMAGE = '/Face Mesh Detection Example/Photos/man.jpg'
SAVE_DIR = '/Face Mesh Detection Example/Delaunay triangle'

#face_mesh_model
STATIC_IMAGE_MODE = True
MAX_NUM_FACES = 5
REFINE_LANDMARKS = True
MIN_DETECTION_CONFIDENCE = 0.1

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

#we get first element of 4D array
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def delaunay_triangle(img_path, save_dir):
    
    make_dir(save_dir)

    #import model face mesh
    mp_face_mesh = mp.solutions.face_mesh

    img = cv.imread(img_path)
    height, width = img.shape[:2]

    with mp_face_mesh.FaceMesh(static_image_mode=True,
                            max_num_faces=2,
                            refine_landmarks=True,
                            min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        
        blank = np.zeros(img.shape[:2], np.uint8)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmark_points = []
                for xyz in face_landmarks.landmark:
                    x = int(xyz.x * width)
                    y = int(xyz.y * height)
                    landmark_points.append((x, y))
                    # print(x, y)
                    cv.circle(img, (x, y), 1, (0, 0, 255), -1)
                
                points = np.array(landmark_points, np.int32)
                convexhull = cv.convexHull(points)
                cv.polylines(blank, [convexhull], True, 255, thickness=1)  
                
                #delaunay triangulation
                rect = cv.boundingRect(points)
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
                            
                for triangle_index in indexes_triangles:
                    pt1 = landmark_points[triangle_index[0]]
                    pt2 = landmark_points[triangle_index[1]]
                    pt3 = landmark_points[triangle_index[2]]
            
                    #draw
                    cv.line(blank, pt1, pt2, 255, 1)
                    cv.line(blank, pt2, pt3, 255, 1)
                    cv.line(blank, pt3, pt1, 255, 1)
            
            #save
            cv.imwrite(save_dir + '/delaunay_triangulation.jpg', blank)
            #show
            cv.imshow('blank', blank)
            cv.imshow('image', img)
            cv.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='delaunay triangulation')
    parser.add_argument('--sav', help='save dir', default=SAVE_DIR, type=str)
    parser.add_argument('-src', '--source-image', help="face to swap", default=IMAGE, type=str)
    args = parser.parse_args()
    
    current_path = os.getcwd()
    
    delaunay_triangle(current_path + args.source_image, current_path + args.sav)