import numpy as np
import cv2


def triangulate_point(K_left, K_right, pt_left, pt_right, R, T):
    """
    Triangulate a single point from stereo image pairs.
    Args:
        K_left (np.ndarray): Intrinsic matrix of the left camera.
        K_right (np.ndarray): Intrinsic matrix of the right camera.
        pt_left (tuple): 2D point in the left image (x, y).
        pt_right (tuple): 2D point in the right image (x, y).
        R (np.ndarray): Rotation matrix from left to right camera.
        T (np.ndarray): Translation vector from left to right camera.
    Returns:
        np.ndarray: 3D point in the world coordinate system, or None if triangulation fails.
    """
    disparity = pt_left[0] - pt_right[0]
    
    if disparity <= 0.5 or disparity > 500:
        return None
    
    P_left = K_left @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P_right = K_right @ np.hstack(( R, T.reshape(-1,1)))

    pts_left_hom = np.array([[pt_left[0]], [pt_left[1]]])
    pts_right_hom = np.array([[pt_right[0]], [pt_right[1]]])

    points_4d_hom = cv2.triangulatePoints(P_left, P_right, pts_left_hom, pts_right_hom)

    if abs(points_4d_hom[3, 0]) < 1e-10:
        return None
        
    points_3d = points_4d_hom[:3] / points_4d_hom[3]  
    point_3d = points_3d.flatten()
    
    if point_3d[2] <= 0 or point_3d[2] < 0.1 or point_3d[2] > 50.0:
        return None
    
    if abs(point_3d[0]) > 50 or abs(point_3d[1]) > 50:
        return None
        
    return point_3d


