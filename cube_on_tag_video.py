import cv2
import numpy as np
import matplotlib.pyplot as plt

from tag_detector import *
from tag_processor import *
from homography_utils import *

if __name__ == "__main__":
    vid_cap = cv2.VideoCapture("assets/tagvideo.mp4")
    detector = TagDetector()

    desired_tag_img_size = 200
    desired_corners = [(0,0), (desired_tag_img_size,0), (0,desired_tag_img_size), (desired_tag_img_size, desired_tag_img_size)]

    cube_height = np.array([-(desired_tag_img_size), -(desired_tag_img_size), -(desired_tag_img_size), -(desired_tag_img_size)]).reshape(-1,1)
    cube_top_corners = np.concatenate((desired_corners, cube_height), axis = 1)

    K = np.array([[1346.100595, 0, 932.1633975], [0, 1355.933136, 654.8986796], [0, 0, 1]])

    frame_counter = 0
    while True:
        _, frame = vid_cap.read()
        img_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_counter % 5 == 0:

            detector.set_tag_image(img_grayscale)
            tag_corners = detector.detect_tag_corners() 

            if len(tag_corners) == 4:
                Hdt = compute_tag_corners_homography_matrix(np.float32(desired_corners), np.float32(tag_corners))
                P = compute_projection_matrix(Hdt, K)
                cube_top_projected_corners = compute_projected_points(cube_top_corners, P)

                draw_cube(frame, tag_corners, cube_top_projected_corners)
            

            cv2.imshow("canny_edge", frame)

            if cv2.waitKey(5) == ord("q"):
                break
                
    vid_cap.release()
    cv2.destroyAllWindows()


    
    
