try:
    import cv2 as cv
    import os
    import numpy as np
    import argparse
    from datetime import datetime
    import random
    import math
except Exception as e:
    print('Caught error while importing {}'.format(e))

_SAVE_DIR = 'Single Lightning Effect Saved Image'

_HIGHTLIGHT_COLOR = (255,255,255) #white
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
        new_mask_d = cv.dilate(pre, kernel, iterations=1)
        new_mask_e = cv.erode(pre, kernel, iterations=1)

        new_mask_d = cv.bitwise_xor(new_mask_d, pre)
        new_mask_e = cv.bitwise_xor(new_mask_e, pre)
        new_mask = cv.add(new_mask_d, new_mask_e)

        hl_image[:] = color
        sd_mask = cv.bitwise_and(hl_image, hl_image, mask=new_mask)
        x = x * blending_value
        result_x = cv.addWeighted(result_x, 1.0, sd_mask, x, 0.9)

    return result_x

def make_single_lightning_mask(start):
    blank = np.zeros(_DIMENSION[:2], np.uint8)
    x, y = start
    pre = start
    while y < blank.shape[1]:
        x = x + random.randint(-30, 30)
        y = y + random.randint(20, 40)
        cv.line(blank, pre, (x, y), (255,255,255), 1)
        pre = (x, y)
        
    return blank

def make_lightning_bolts_mask(start):
    blank = np.zeros(_DIMENSION[:2], np.uint8)
    
    cnt = random.randint(3, 5)
    thickness = 2
    while cnt > 0:
        x, y = start
                
        pre = start
        while y < blank.shape[1]:
            x = x + random.randint(-30, 30)
            y = y + random.randint(20, 30)
            cv.line(blank, pre, (x, y), (255,255,255), thickness)
            pre = (x, y)
        
        cnt -= 1
        
    return blank

def distance(p1, p2):
    return math.sqrt((p1[1] - p2[1]) * (p1[1] - p2[1]) + (p1[0] - p2[0]) * (p1[0] - p2[0]))

def cosin(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if (x1*x1 + y1*y1 == 0) or (x2*x2 + y2*y2 == 0):
        return 1
    else:
        return (x1 * x2 + y1 * y2) / (math.sqrt(x1*x1 + y1*y1) * math.sqrt(x2*x2 + y2*y2))
    
def side_point_lie(A, B, I):
    x1,y1 = A
    x2,y2 = B
    x,y = I
    d = (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)
    return 1 if d > 1 else -1

def rotate(start, end, b, c):
    #find cosin and sin of the angle to be rotated
    cos = cosin((b[0] - start[0], b[1] - start[1]), (end[0] - start[0], end[1] - start[1]))
    sin = math.sqrt(1 - cos*cos)
    #rotated point of c
    C1 = ((c[0] - start[0])*cos - side_point_lie(start, end, b)*(c[1] - start[1])*sin, side_point_lie(start, end, b)*(c[0] - start[0])*sin + (c[1] - start[1])*cos)
    #tranform to equaly distance
    Ob = distance(start, b)
    OB = distance(start, end)
    x = OB/Ob
    C = (int(x * C1[0]) + start[0], int(x * C1[1]) + start[1])
    return C

def rotate_line(points, points_2, start, end): 
    p = []
    p2 = []   
    for (point, point_2) in zip(points, points_2):
        p.append(list(rotate(start, end, points[-1], point)))
        p2.append(list(rotate(start, end, points_2[-1], point_2)))
    
    return [p, p2]
       
def make_bigger_lightning(dimension, start, end, thickness, bend=4):
    blank = np.zeros(dimension, np.uint8)
    # from center started point to make 2 left right started point 
    x_center,y_center = start
    # length of lightning we will make
    # depth is number lightning zigzag
    length = dimension[1] - y_center
    depth = get_first_divisor(thickness)//2
    
    start_point = (x_center - (thickness // 2), y_center)
    start_point_2 = (x_center + (thickness // 2), y_center)
    
    x, y = start_point[:2]
    x_2, y_2 = start_point_2[:2]    
    
    pre = start_point
    pre_2 = start_point_2
    #get left, right zigzag line points sets
    points = list([pre])
    points_2 = list([pre_2])
    while y < length:
        x_ran = random.randint(-(thickness//2 + bend*10), thickness//2 + bend*10)
        y_ran = random.randint(length//10 - 10, length//10)
        # make next point with random position
        # if point is out of image, we reverse sign
        if x + x_ran + depth >= 0 and x_2 + x_ran + depth <= dimension[0]:
            x = x + x_ran + depth 
            x_2 = x_2 + x_ran - depth
        else: 
            x - x_ran + depth
            x_2 - x_ran + depth
        if(y + y_ran < dimension[1]):
            y = y + y_ran 
            y_2 = y_2 + y_ran       
        
        if(x > x_2 and y < dimension[1]):
            break
            
        pre = (x, y)
        pre_2 = (x_2, y_2)
        
        points.append(pre)
        points_2.append(pre_2)
    #rotate and transfer end point of lightning to end point
    points, points_2 = rotate_line(points, points_2, start, end)
    #draw lightning
    cv.line(blank, points[0], points_2[0], (255,255,255), 1, cv.LINE_AA)
    pre = points[0]
    pre_2 = points_2[0]
    for (point, point_2) in zip(points, points_2):
        if(point[0] < 0):   point[0] = 0
        elif(point[0] > dimension[0]):  point[0] = dimension[0]
        
        if(point_2[0] < 0): point_2[0] = 0
        elif(point_2[0] > dimension[0]):  point_2[0] = dimension[0]
        
        if(point[1] < 0):   point[1] = 0
        elif(point[1] > dimension[1]):  point[1] = dimension[1]
        
        if(point_2[1] < 0): point_2[1] = 0
        elif(point_2[1] > dimension[1]):  point_2[1] = dimension[1]
        # print(point)
        cv.line(blank, pre, point, (255,255,255), 1, cv.LINE_AA)
        cv.line(blank, pre_2, point_2, (255,255,255), 1, cv.LINE_AA)
        #set up current point to connect next point
        pre = point
        pre_2 = point_2
    #filled lightning shape
    blank_2 = np.zeros(dimension, np.uint8)
    contours,_ = cv.findContours(blank, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(blank_2, contours, -1, (255,255,255), cv.FILLED)

    return blank_2

def make_lightning_tree(dimension, start, thickness, bend=4, check = True, edge = 1):
    blank = np.zeros(dimension, np.uint8)
    # from center started point to make 2 left right started point 
    x_center,y_center = start[:2]

    start_point = (x_center - (thickness // 2), y_center)
    start_point_2 = (x_center + (thickness // 2), y_center)

    x, y = start_point[:2]
    x_2, y_2 = start_point_2[:2]    

    pre = start_point
    pre_2 = start_point_2
    #creat zigzag degree and length of lightning
    depth = get_first_divisor(thickness) // 2

    cv.line(blank, start_point, start_point_2, (255,255,255), 1)
    sub_lightnings = []
    while y < dimension[1]:
        x_ran = random.randint(-(thickness//2 + bend*10), thickness//2 + bend*10)
        y_ran = random.randint(dimension[1]//10 - 20, dimension[1]//10 - 10)
        # make next point with random position
        # if point is out of image, we reverse sign
        if x + x_ran + depth >= 0 and x_2+ x_ran + depth <= dimension[0]:
            x = x + x_ran + depth 
            x_2 = x_2 + x_ran - depth
        else: 
            x - x_ran + depth
            x_2 - x_ran + depth
        if(y + y_ran < dimension[1]):
            y = y + y_ran 
            y_2 = y_2 + y_ran 
        
        if(y > y_center and check == True and edge > 0):
            check = random.choice([True, False])
            sub_lightning = make_lightning_tree(dimension, ((x + x_2)//2, (y + y_2)//2), x_2 - x, 4, check, edge - 1)
            sub_lightnings.append(sub_lightning)
        
        if(x > x_2 and y < dimension[1]):
            break

        cv.line(blank, pre, (x, y), (255,255,255), 1)
        cv.line(blank, pre_2, (x_2, y_2), (255,255,255), 1)

        pre = (x, y)
        pre_2 = (x_2, y_2)
    #we find contours and draw filled to get shape of lightning
    blank_2 = np.zeros(dimension, np.uint8)
    contours,_ = cv.findContours(blank, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(blank_2, contours, -1, (255,255,255), cv.FILLED)
    #add sub ligjtning to root
    for lightning in sub_lightnings:
        blank_2 = cv.add(blank_2, lightning)
        
    return blank_2

def lightning_effect(save_dir, color, mode, start, end, thickness):
    blank = np.zeros(_DIMENSION, np.uint8)
    if mode == 1:
        mask = make_bigger_lightning(blank.shape[:2], start, end, thickness, 4)
    elif mode == 2:
        mask = make_lightning_tree(blank.shape[:2], (150,40), 20, 4)
    elif mode == 3:  
        mask = make_lightning_bolts_mask(start)
    else:
        mask = make_single_lightning_mask(start)
        
    blank = draw_shadow(blank, mask, _COLOR[color], 8, 0.7, 0.8)
    #save image
    save_image(save_dir, 'single_lightning_effect', blank)
    #show image
    cv.imshow('blank', blank)
    cv.waitKey(0)

def get_first_divisor(n):
    if n <= 3: return 2
    for i in range(2, n):
        if n % i == 0:
            return i

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single lightning')
    parser.add_argument('--sav', help='name of saved directory', default=_SAVE_DIR, type=str)
    parser.add_argument('--color', help="color of light", default=1,choices=[0,1,2], type=int)
    parser.add_argument('--mode', help="mode of light", default=1,choices=[1,2,3,4], type=int)
    parser.add_argument('--start', help="header of the lightning", nargs='+', default=[200,50], type=int)
    parser.add_argument('--end', help="tail of the lightning", nargs='+', default=[480,480], type=int)
    parser.add_argument('--thickness', help="width of the lightning", default=20, type=int)

    args = parser.parse_args()
    save_dir = find_file(args.sav)
    
    make_dir(save_dir)
    
    lightning_effect(save_dir, args.color, args.mode, list(args.start), list(args.end), args.thickness)