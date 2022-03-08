import cv2
import numpy as np
import matplotlib.pyplot as plt

from tag_detector import *
from tag_processor import *
from homography_utils import *


if __name__ == "__main__":
    img_color = cv2.imread("assets/image_2.png")
    img_color = cv2.cvtColor(img_color , cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # detecting the tag from the image
    detector = TagDetector()
    detector.set_tag_image(img_gray)
    tag_corners = detector.detect_tag_corners()    

    K = np.array([[1346.100595, 0, 932.1633975], [0, 1355.933136, 654.8986796], [0, 0, 1]])

    desired_tag_img_size = 200
    desired_corners = [(0,0), (desired_tag_img_size,0), (0,desired_tag_img_size), (desired_tag_img_size, desired_tag_img_size)]

    cube_height = np.array([-(desired_tag_img_size), -(desired_tag_img_size), -(desired_tag_img_size), -(desired_tag_img_size)]).reshape(-1,1)
    cube_top_corners = np.concatenate((desired_corners, cube_height), axis = 1)

    Hmatrix_3 = compute_tag_corners_homography_matrix(np.float32(desired_corners), np.float32(tag_corners))
    P = compute_projection_matrix(Hmatrix_3, K)
    cube_top_projected_corners = compute_projected_points(cube_top_corners, P)

    ar_image = draw_cube(img_color.copy(), tag_corners, cube_top_projected_corners)

    plt.figure(figsize=(50,50)) 
    plt.subplot(2, 1, 1), plt.imshow(img_color) 
    plt.subplot(2, 1, 1), plt.scatter(np.array(tag_corners)[:, 0], np.array(tag_corners)[:, 1], color="blue") 
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 1, 2), plt.imshow(ar_image) 
    plt.subplot(2, 1, 2), plt.scatter(np.array(tag_corners)[:, 0], np.array(tag_corners)[:, 1], color="blue") 
    plt.subplot(2, 1, 2), plt.scatter(cube_top_projected_corners[:, 0], cube_top_projected_corners[:, 1], color="red") 
    plt.title('AR Cube Image'), plt.xticks([]), plt.yticks([])
    plt.show()