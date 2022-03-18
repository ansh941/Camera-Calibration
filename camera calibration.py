# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:30:02 2022

@author: ASH
"""

import numpy as np
import cv2


def define_location(event, x, y, flags, params):
    global points_real
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Mouse Position: '+str(x)+', '+str(y))
        points_real.append([x,y,1])

def extract_coordinates_from_image(img, world):
    global points_real, image_points, world_points
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', define_location)
    cv2.imshow('image', img)
    
    while True:
        if cv2.waitKey()==27 and len(points_real) == 25:
            cv2.destroyAllWindows()
            break

    image_points.extend(points_real)
    world_points.extend(world[:len(points_real)])
    
np.set_printoptions(suppress=True)
gap = 12

image_points = []
world_points = []

img = cv2.imread('./pattern.png', cv2.IMREAD_COLOR)

# Define XY
world_XY = [[gap*i, gap*j, 0, 1] for j in range(5) for i in range(5)]

points_real = [[761, 533, 1], [832, 576, 1], [907, 625, 1], [986, 670, 1], [1070, 721, 1],
               [695, 575, 1], [765, 619, 1], [844, 669, 1], [921, 719, 1], [1007, 772, 1],
               [624, 619, 1], [695, 668, 1], [773, 714, 1], [854, 771, 1], [937, 827, 1],
               [549, 664, 1], [620, 716, 1], [700, 770, 1], [779, 826, 1], [863, 885, 1],
               [470, 714, 1], [545, 769, 1], [621, 824, 1], [705, 884, 1], [789, 945, 1]]

image_points.extend(points_real)
world_points.extend(world_XY[:len(points_real)])

# points_real = []
# extract_coordinates_from_image(img, world_XY)

# Define YZ
world_YZ = [[0, gap*i, gap*j, 1] for j in range(5) for i in range(5)]

points_real = [[761, 533, 1], [834, 574, 1], [908, 625, 1], [987, 670, 1], [1070, 720, 1],
               [765, 454, 1], [841, 497, 1], [919, 543, 1], [999, 590, 1], [1085, 642, 1],
               [770, 375, 1], [848, 415, 1], [929, 461, 1], [1013, 507, 1], [1100, 556, 1],
               [776, 283, 1], [857, 328, 1], [939, 371, 1], [1027, 417, 1], [1117, 464, 1],
               [783, 194, 1], [864, 232, 1], [950, 277, 1], [1042, 323, 1], [1136, 369, 1]]

# points_real = []
# extract_coordinates_from_image(img, world_YZ)

image_points.extend(points_real)
world_points.extend(world_YZ[:len(points_real)])

# Define XZ
world_XZ = [[gap*i, 0, gap*j, 1] for j in range(5) for i in range(5)]

points_real = [[762, 533, 1], [695, 572, 1], [623, 619, 1], [549, 664, 1], [468, 712, 1],
               [766, 456, 1], [698, 493, 1], [623, 538, 1], [546, 585, 1], [465, 633, 1],
               [770, 373, 1], [700, 410, 1], [625, 455, 1], [546, 498, 1], [460, 544, 1],
               [775, 285, 1], [704, 323, 1], [628, 363, 1], [545, 406, 1], [458, 453, 1],
               [782, 194, 1], [706, 230, 1], [629, 271, 1], [545, 309, 1], [454, 352, 1]]

# points_real = []
# extract_coordinates_from_image(img, world_XZ)

image_points.extend(points_real)
world_points.extend(world_XZ[:len(points_real)])

# Points normalization
image_points = np.array(image_points, dtype=np.float32)
world_points = np.array(world_points, dtype=np.float32)

std_image = np.sum(np.std(image_points[:,:2], axis=0))/2
std_world = np.sum(np.std(world_points[:,:3], axis=0))/3

mean_image = np.mean(image_points, axis=0)
mean_world = np.mean(world_points, axis=0)

T_image = np.array([
    [1/std_image, 0, -mean_image[0]/std_image],
    [0, 1/std_image, -mean_image[1]/std_image],
    [0,0,1]
    ])
T_world = np.array([
    [1/std_world, 0, 0,-mean_world[0]/std_world],
    [0, 1/std_world, 0, -mean_world[1]/std_world],
    [0, 0, 1/std_world, -mean_world[2]/std_world],
    [0,0,0,1]
    ])

image_points_n = np.matmul(T_image, image_points.T).T
world_points_n = np.matmul(T_world, world_points.T).T

N = len(image_points_n)
A = np.zeros((2*N, 12), dtype=np.float64)

for i in range(N):
    X,Y,Z,_ = world_points_n[i]
    x,y,_ = image_points_n[i]
    A[2*i] = np.array([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
    A[(2*i)+1] = np.array([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])
    print('World(3D): {}, Image(2D): {}'.format((X,Y,Z),(x,y)))
    
print('----- Data Matrix Constructed ----------------------')
print('A:', A.shape)

# SVD
U, s, Vh = np.linalg.svd(A)
p = Vh[np.argmin(s)]
proj = p.reshape(3, 4)

proj_ori = np.linalg.inv(T_image)@proj@T_world
print('proj_ori : ', proj_ori)

def null(a):
    u, s, v = np.linalg.svd(a)
    
    return v[-1].T.copy()

# Camera center
C = null(proj_ori)
Co = C/C[-1]

print('camera center:', Co)
print('PC:', np.matmul(proj_ori, Co))

# QR factorization
M_ = proj_ori[:3, :3].copy()
R_inv, K_inv = np.linalg.qr(np.linalg.inv(M_))  # q, r
R = np.linalg.inv(R_inv)

K_temp = np.linalg.inv(K_inv)
K = K_temp / K_temp[2, 2]

proj_final = proj_ori / K_temp[2, 2]
t = np.matmul(K_inv, proj_final.T[-1])

print('----- Camera Params ----------------------')
print('proj_final:', proj_final)
print('K:', K)
print('R:', R)
print('t:', t)