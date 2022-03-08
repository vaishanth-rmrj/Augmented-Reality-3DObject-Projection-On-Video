import cv2
import numpy as np

class TagDetector:

    def __init__(self, corner_confidence = 0.1, corner_dist = 50, max_corners = 15):
        self.tag_corners = []
        self.fft_image = np.array([])
        self.fshift_mask_mag = np.array([])

        # params for good feature to track
        self.corner_confidence = corner_confidence
        self.corner_dist = corner_dist
        self.max_corners = max_corners

    def set_tag_image(self, tag_image):
        self.image = tag_image

    def apply_binary_threshholding(self, image):
        _,thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        self.image = thresh

    def apply_morphology(self, image):
        kernel = np.ones((5,5), np.uint8)
        img_erosion = cv2.erode(image, kernel, iterations=1) 
        self.image = img_erosion

    def get_circular_mask(self, img_shape, radius):
        rows, cols = img_shape

        mask = np.ones((rows, cols, 2), np.uint8)
        h, k = int(rows / 2), int(cols / 2)
        for x in range(0,rows):
            for y in range(0,cols):
                if (x - h) ** 2 + (y - k) ** 2 < radius**2:
                        mask[x,y] = 0   

        return mask

    def perform_fft(self, image, circular_mask_radius):

        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT) 
        dft_shift = np.fft.fftshift(dft) 

        mask = self.get_circular_mask(image.shape, circular_mask_radius)
        
        fshift = dft_shift * mask #masking the fft of the image
        self.fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1])) 
        f_ishift = np.fft.ifftshift(fshift) 

        fft_image = cv2.idft(f_ishift) #inverse fft to get the original image
        self.fft_image = cv2.magnitude(fft_image[:, :, 0], fft_image[:, :, 1])

    def get_cluster_mean(self, x_pts, y_pts):

        mean = (np.mean(x_pts), np.mean(y_pts))

        while True:
            x_pts.append(mean[0])
            y_pts.append(mean[1])

            new_mean = (np.mean(x_pts), np.mean(y_pts))
            if new_mean == mean:
                mean = new_mean
                break

            mean = new_mean

        return mean

    def reject_unwanted_points(self, mean, x_pts, y_pts):
        tag_corners = []
        for x,y in zip(x_pts, y_pts):            
            dist = ((x-mean[0])**2 + (y-mean[1])**2)**0.5
            if dist < 200 and dist > 100:
                tag_corners.append((x, y))

        return tag_corners

    def detect_corners(self):
        corners = cv2.goodFeaturesToTrack(self.fft_image, self.max_corners, self.corner_confidence, self.corner_dist)

        # separate x and y
        x_corners = []
        y_corners = []
        for corner in corners:
            x_corners.append(corner[0][0])
            y_corners.append(corner[0][1])

        mean = self.get_cluster_mean(x_corners, y_corners)
        self.tag_corners = self.reject_unwanted_points(mean, x_corners, y_corners) 
    
    def detect_tag_corners(self):
        self.apply_binary_threshholding(self.image)
        self.apply_morphology(self.image)
        self.perform_fft(self.image, circular_mask_radius=220)
        self.detect_corners()

        return self.tag_corners

    def get_inverse_fft(self):
        return self.fft_image

    def get_fft_masked(self):
        return self.fshift_mask_mag