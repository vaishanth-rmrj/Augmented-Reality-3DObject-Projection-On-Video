import numpy as np
import cv2

def sort_corners(corners):
    if len(corners) == 4:
        sort_by_x = sorted(corners, key=lambda x: x[0])
        bottom_points = [sort_by_x[0], sort_by_x[1]]
        top_points = [sort_by_x[2], sort_by_x[3]]

        top_points_sorted = sorted(top_points, key=lambda x: x[1])
        bottom_points_sorted = sorted(bottom_points, key=lambda x: x[1])

        t_l = top_points_sorted[0]
        t_r = top_points_sorted[1]
        b_l = bottom_points_sorted[0]
        b_r = bottom_points_sorted[1]

        return [t_l, t_r, b_r, b_l]
    
    print("Error: You must pass only 4 corners values")

def compute_tag_corners_homography_matrix(tag_corners, desired_corners):

    if (len(tag_corners) != 4) or (len(desired_corners) != 4):
        print("Need only four points to compute SVD.")
        return 0

    tag_corners = np.array(sort_corners(tag_corners))
    # print(tag_corners)
    desired_corners = np.array(sort_corners(desired_corners))    

    x = tag_corners[:, 0]
    y = tag_corners[:, 1]
    xp = desired_corners[:, 0]
    yp = desired_corners[:, 1]
    
    A = []
    for i in range(len(x)):
        row1 = np.array([-x[i], -y[i], -1, 0, 0, 0, x[i]*xp[i], y[i]*xp[i], xp[i]])
        A.append(row1)
        row2 = np.array([0, 0, 0, -x[i], -y[i], -1, x[i]*yp[i], y[i]*yp[i], yp[i]])
        A.append(row2)

    A = np.array(A)
    U, E, V_T = np.linalg.svd(A)
    V = V_T.transpose()
    H_vertical = V[:, V.shape[1] - 1]
    H = H_vertical.reshape([3,3])
    H = H / H[2,2]

    return H

def inverse_warp_image(H_matrix, image, output_size):

    H_inv = np.linalg.inv(H_matrix)

    original_points_x = []
    original_points_y = []

    for row in range(output_size[0]):
        for col in range(output_size[1]):

            pos_vect = np.array([row, col, 1]).T
            desired_pos_vect = H_inv.dot(pos_vect)
            desired_pos_vect /= desired_pos_vect[2] #normalizing with scaling factor

            original_points_x.append(int(desired_pos_vect[0]))
            original_points_y.append(int(desired_pos_vect[1]))

    warped_img = []
    for x,y in zip(original_points_x, original_points_y):        
        warped_img.append(image[y, x])

    warped_img= np.reshape(warped_img, output_size)

    return warped_img

def superimpose_image(H_matrix, image1, image2):

    H_inv = np.linalg.inv(H_matrix)

    original_points_x = []
    original_points_y = []
    image2_flattened = []

    for row in range(image2.shape[0]):
        for col in range(image2.shape[1]):

            pos_vect = np.array([row, col, 1]).T
            desired_pos_vect = H_inv.dot(pos_vect)
            desired_pos_vect /= desired_pos_vect[2] #normalizing with scaling factor

            original_points_x.append(int(desired_pos_vect[0]))
            original_points_y.append(int(desired_pos_vect[1]))

            # flatten image 2 
            image2_flattened.append(list(image2[row, col]))
    
    
    for index, (x,y) in enumerate(zip(original_points_x, original_points_y)):        
        image1[y, x] = image2_flattened[index]           

    return image1


def compute_projection_matrix(H_matrix, K_matrix):
    K_inv = np.linalg.inv(K_matrix)

    B_tilda = np.dot(K_inv, H_matrix)
    B_tilda_mod = np.linalg.norm(B_tilda)

    if B_tilda_mod < 0:
        B = -1  * B_tilda
    else:
        B =  B_tilda

    b1 = B[:,0]
    b2 = B[:,1]
    b3 = B[:,2]

    lambda_ = (np.linalg.norm(b1) + np.linalg.norm(b2))/2
    lambda_ = 1 / lambda_

    r1 = lambda_ * b1
    r2 = lambda_ * b2
    r3 = np.cross(r1, r2)
    t = lambda_ * b3

    P = np.array([r1,r2, r3, t]).T
    P = np.dot(K_matrix, P)
    P = P / P[2,3]
    return P

def compute_projected_points(points, P_matrix):
    x_pts = points[:, 0]
    y_pts = points[:, 1]
    z_pts = points[:, 2]

    points_3d_matrix = np.stack((x_pts, y_pts, z_pts, np.ones(x_pts.size)))
    transformed_points_matrix = P_matrix.dot(points_3d_matrix)
    transformed_points_matrix /= transformed_points_matrix[2,:]

    x_2d = np.int32(transformed_points_matrix[0,:])
    y_2d = np.int32(transformed_points_matrix[1,:])  

    projected_2d_points = np.dstack((x_2d, y_2d)).reshape(4,2)
    return projected_2d_points

def draw_cube(image, bottom_points, top_points):

    bottom_points = sort_corners(bottom_points)
    bottom_points = np.array(bottom_points, np.int32)
    image = cv2.polylines(image, [bottom_points],True, (55, 66, 219), 2)

    top_points = sort_corners(top_points)
    top_points = np.array(top_points, np.int32)
    image = cv2.polylines(image, [top_points],True, (55, 66, 219), 2)

    for i in range(0, bottom_points.shape[0]):
        cv2.line(image, (bottom_points[i,0], bottom_points[i,1]), (top_points[i,0], top_points[i,1]), (55, 66, 219), 3)

    return image