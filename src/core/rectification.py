import cv2
import numpy as np

class StereoRectifier:
    def __init__(self, K_left, K_right, D_left, D_right, R, T, image_size):
        self.K_left = K_left
        self.K_right = K_right
        self.D_left = D_left
        self.D_right = D_right
        self.R = R
        self.T = T
        self.image_size = image_size
        self.map1_left = None
        self.map2_left = None
        self.map1_right = None
        self.map2_right = None
        self.new_K_left = None
        self.new_K_right = None
        
    def compute_rectification_maps(self):
        R_left, R_right, P_left, P_right, self.Q = cv2.fisheye.stereoRectify(
            self.K_left, self.D_left,
            self.K_right, self.D_right,
            self.image_size,
            self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            fov_scale=1.0,
            balance=0.0
        )
        
        self.map1_left, self.map2_left = cv2.fisheye.initUndistortRectifyMap(
            self.K_left, self.D_left, R_left, P_left, self.image_size, cv2.CV_16SC2
        )
        
        self.map1_right, self.map2_right = cv2.fisheye.initUndistortRectifyMap(
            self.K_right, self.D_right, R_right, P_right, self.image_size, cv2.CV_16SC2
        )
        
        self.new_K_left = P_left[:3, :3]
        self.new_K_right = P_right[:3, :3]
        
    def rectify_stereo_pair(self, img_left, img_right):
        if self.map1_left is None:
            self.compute_rectification_maps()
            
        img_left_rect = cv2.remap(img_left, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        img_right_rect = cv2.remap(img_right, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        
        return img_left_rect, img_right_rect
    
    def get_rectified_camera_matrices(self):
        if self.new_K_left is None:
            self.compute_rectification_maps()
        return self.new_K_left, self.new_K_right