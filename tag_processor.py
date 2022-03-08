import cv2
import numpy as np

class TagProcessor:

    def __init__(self):
        pass

    def set_tag_image(self, tag_image):
        self.image = tag_image

    def decode_tag(self):

        tag_orientaion = "up"
        tag_id = 0

        img_width, img_height = self.image.shape
        kernel_width = int(img_width/8)
        kernel_height = int(img_height/8)

        tag = []

        for row in range(int(kernel_width/2), img_width, kernel_width):
            for col in range(int(kernel_height/2), img_height, kernel_height):
                kernel = self.image[row - int(kernel_width/2) : row + int(kernel_width/2), col - int(kernel_height/2) : col + int(kernel_height/2)] 

                if np.median(kernel) == 255: 
                    tag.append(int(1))
                else:
                    tag.append(int(0))

        tag_matrix = np.reshape(tag, (8,8))
        info_matrix = tag_matrix[2:6, 2:6]

        if info_matrix[0,0] == 1:
            tag_orientaion = "down"
            info_matrix = np.rot90(info_matrix, 2)
        elif info_matrix[0,3] == 1:
            tag_orientaion = "right"
            info_matrix = np.rot90(info_matrix, 3)
        elif info_matrix[3,0] == 1:
            info_matrix = np.rot90(info_matrix, 1)
            tag_orientaion = "left"

        tag_data = info_matrix[1:3, 1:3].flatten()

        for i in range(len(tag_data)):
            if tag_data[i]:
                tag_id += 2**i

        return tag_id, tag_orientaion