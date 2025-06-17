import cv2
import numpy as np

def pose_estimation_ransac(pts_3d, pts_2d, K, ransac_threshold=2.0, confidence=0.99, max_iterations=1000):
    """
    Robust pose estimation using RANSAC PnP.
    
    Parameters:
    - pts_3d: np.array of shape (N, 3), 3D points in world coordinates
    - pts_2d: np.array of shape (N, 2), 2D points in image coordinates  
    - K: np.array of shape (3, 3), intrinsic camera matrix
    - ransac_threshold: float, reprojection error threshold for RANSAC
    - confidence: float, confidence level for RANSAC
    - max_iterations: int, maximum RANSAC iterations
    
    Returns:
    - success: bool, whether pose estimation succeeded
    - R: np.array of shape (3, 3), rotation matrix (or None if failed)
    - t: np.array of shape (3,), translation vector (or None if failed)
    - inliers: np.array, indices of inlier correspondences (or None if failed)
    """
    try:
        if len(pts_3d) < 4 or len(pts_2d) < 4:
            return False, None, None, None
            
        # Ensure input arrays are the right type
        pts_3d = pts_3d.astype(np.float32)
        pts_2d = pts_2d.astype(np.float32)
        
        # Use RANSAC for robust pose estimation
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, K, None,
            iterationsCount=max_iterations,
            reprojectionError=ransac_threshold,
            confidence=confidence,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success and rvec is not None and tvec is not None:
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.flatten()
            return True, R, t, inliers
        else:
            return False, None, None, None
            
    except Exception as e:
        print(f"Pose estimation failed with error: {e}")
        return False, None, None, None

def pose_estimation(pts_3d, pts_2d, K):
    """
    Basic pose estimation without RANSAC (for backward compatibility).
    
    Parameters:
    - pts_3d: np.array of shape (N, 3), 3D points in world coordinates
    - pts_2d: np.array of shape (N, 2), 2D points in image coordinates
    - K: np.array of shape (3, 3), intrinsic camera matrix
    
    Returns:
    - R: np.array of shape (3, 3), rotation matrix
    - t: np.array of shape (3,), translation vector
    """
    try:
        if len(pts_3d) < 4 or len(pts_2d) < 4:
            raise ValueError("Need at least 4 point correspondences")
            
        pts_3d = pts_3d.astype(np.float32)
        pts_2d = pts_2d.astype(np.float32)
        
        success, rvec, tvec = cv2.solvePnP(
            pts_3d, pts_2d, K, None, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success and rvec is not None and tvec is not None:
            R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to rotation matrix
            t = tvec.flatten()
            return R, t
        else:
            raise ValueError("Pose estimation failed: solvePnP returned failure")
            
    except Exception as e:
        print(f"Pose estimation error: {e}")
        raise ValueError(f"Pose estimation failed: {e}")
