try:
    import cv2 as cv
    import numpy as np
    import mediapipe as mp
    import os
    import argparse
except Exception as e:
    print('Caught error while importing: {}'.format(e))
    
IMAGE_PATH = '/Face Mesh Detection Example/Photos/long.jpg'
SAVE_DIR = '/Face Mesh Detection Example/Geometric Art Save Image'

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def resized_image(image, scale = 500):
    height, width = image.shape[:2]
    x = scale / width
    return cv.resize(image, (scale, int(height * x)))

def area(p1, p2, p3):
    return abs((p1[0] * (p2[1] - p3[1]) + 
                p2[0] * (p3[1] - p1[1]) + 
                p3[0] * (p1[1] - p2[1])) / 2.0)

def is_inside(p1, p2, p3, p):
    A = area(p1, p2, p3)
    A1 = area(p, p2, p3)
    A2 = area(p1, p, p3)
    A3 = area(p1, p2, p)
    
    if A == A1 + A2 + A3:
        return True
    else:
        return False

def fill_triangle_max(image, blank, vertices):
    p1, p2, p3 = vertices[:3]
    b = g = r = 0
    index = 0
    for i in range(0, 500):
        for j in range(0, 500):
            p = (i, j)
            if is_inside(p1, p2, p3, p):
                x,y,z = image[i,j]
                b += x
                g += y
                r += z
                index += 1
    
    b = b//index
    g = g//index
    r = r//index 
    
    color = (int(b), int(g), int(r))
    tri = np.array([[p1, p2, p3]]).astype(int)
    cv.fillPoly(blank, tri, tuple(color))
    return blank

def fill_triangle_min(image, blank, vertices):
    p1, p2, p3 = vertices[:3]
    
    b = (int(image[p1[1], p1[0]][0]) + int(image[p2[1], p2[0]][0]) + int(image[p3[1], p3[0]][0])) // 3
    g = (int(image[p1[1], p1[0]][1]) + int(image[p2[1], p2[0]][1]) + int(image[p3[1], p3[0]][1])) // 3
    r = (int(image[p1[1], p1[0]][2]) + int(image[p2[1], p2[0]][2]) + int(image[p3[1], p3[0]][2])) // 3
    
    color = (int(b), int(g), int(r))
    tri = np.array([[p1, p2, p3]]).astype(int)
    cv.fillPoly(blank, tri, tuple(color))
    return blank

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
            new_img = image[y - 50 : y + h + 50, x - 50 : x + w + 50]
            location_face_detection.append(new_img)

        return location_face_detection
  
def get_face_mesh(img, model, save_dir):
    
    mp_face_mesh = mp.solutions.face_mesh
    img = resized_image(img)
    height, width = img.shape[:2]
    
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                            max_num_faces=2,
                            refine_landmarks=True,
                            min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        
        triangle_blank = np.zeros(img.shape[:2], np.uint8)
        painting_blank = img.copy()
        index = 0
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmark_points = []
                for xyz in face_landmarks.landmark:
                    x = int(xyz.x * width)
                    y = int(xyz.y * height)
                    landmark_points.append((x, y))
                    # print(x, y)
                    # cv.circle(img, (x, y), 1, (0, 0, 255), -1)
                
                points = np.array(landmark_points, np.int32)
                convexhull = cv.convexHull(points)
                # cv.polylines(img, [convexhull], True, 255, thickness=1)  
                
                #delaunay triangulation
                rect = cv.boundingRect(points)
                subdiv = cv.Subdiv2D(rect)
                subdiv.insert(landmark_points)
                triangle = subdiv.getTriangleList()
                triangle = np.array(triangle).astype(int)
                
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
                    cv.line(triangle_blank, pt1, pt2, 255, 1)
                    cv.line(triangle_blank, pt2, pt3, 255, 1)
                    cv.line(triangle_blank, pt3, pt1, 255, 1)
                    
                    if model == 1:
                        fill_triangle_min(img, painting_blank, [pt1, pt2, pt3])
                    elif model == 0:
                        fill_triangle_max(img, painting_blank, [pt1, pt2, pt3])
                        index += 1
                        if index == 300:
                            break
            #save image
            cv.imwrite(save_dir + '/geometric_art.jpg', painting_blank)
            #show image
            cv.imshow('paint', painting_blank)
            cv.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Geometric art')
    parser.add_argument('-src', '--source-image', help="face is make up", default=IMAGE_PATH, type=str)
    parser.add_argument('--model', help='algorithms to get color: 1 is min and 0 is max', default=1, type=int)
    parser.add_argument('--sav', help='save dir', default=SAVE_DIR, type=str)

    args = parser.parse_args()
    
    current_path = os.getcwd()
    
    images = get_face_detection(current_path + args.source_image)
    
    make_dir(current_path + args.sav)

    for image in images:
        get_face_mesh(image, current_path + args.model, current_path + args.sav)