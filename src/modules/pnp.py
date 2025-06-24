import cv2
import numpy as np
from src.core.pose_estimation import pose_estimation_ransac




class PnP:
    def __init__(self, extractor, K_left, verbose=True):
        self.extractor = extractor
        self.K_left = K_left
        self.verbose = verbose
        

    def estimate_pose(self, prev_descriptors, prev_keypoints, prev_3d_points, prev_feature_indices,
                      curr_descriptors, curr_keypoints, frame_id=None):
        """
        Estimate the pose using PnP with RANSAC.
        Args:
            prev_descriptors (np.ndarray): Descriptors from the previous frame.
            prev_keypoints (list): Keypoints from the previous frame.
            prev_3d_points (np.ndarray): 3D points corresponding to the previous keypoints.
            prev_feature_indices (np.ndarray): Indices of features in the previous frame.
            curr_descriptors (np.ndarray): Descriptors from the current frame.
            curr_keypoints (list): Keypoints from the current frame.
            frame_id (int, optional): Frame ID for logging purposes.
        Returns:
            R (np.ndarray): Rotation matrix if pose estimation is successful, None otherwise.
            t (np.ndarray): Translation vector if pose estimation is successful, None otherwise.
            inliers (np.ndarray): Indices of inliers if pose estimation is successful, None otherwise.
            matched_3d_2d (tuple): Tuple of matched 3D points and 2D points if pose estimation is successful, None otherwise.
        """
        matched_3d = []
        matched_2d = []

        if prev_descriptors is None or curr_descriptors is None:
            return None, None, None, None

        prev_query_idx, curr_train_idx = self.extractor.match(
            prev_descriptors, curr_descriptors, min_cossim=0.6
        )

        if len(prev_query_idx) < 8:
            if self.verbose:
                print(f"Not enough temporal matches: {len(prev_query_idx)}")
            return None, None, None, None

        for prev_idx, curr_idx in zip(prev_query_idx, curr_train_idx):
            match_3d_idx = np.where(prev_feature_indices == prev_idx)[0]
            if len(match_3d_idx) > 0:
                pt3d = prev_3d_points[match_3d_idx[0]]
                pt2d = curr_keypoints[curr_idx].cpu().numpy()

                matched_3d.append(pt3d)
                matched_2d.append(pt2d)

        if len(matched_3d) < 8:
            if self.verbose:
                print(f"Not enough 3D-2D correspondences after filtering: {len(matched_3d)}")
            return None, None, None, None

        matched_3d = np.array(matched_3d)
        matched_2d = np.array(matched_2d)

        success, R, t, inliers = pose_estimation_ransac(
            matched_3d, matched_2d, self.K_left,
            ransac_threshold=5.0, confidence=0.95, max_iterations=2000
        )

        if not success or inliers is None or len(inliers) == 0:
            if self.verbose:
                print("PnP failed.")
            return None, None, None, None

        if self.verbose:
            prefix = f"Frame {frame_id}:" if frame_id is not None else ""
            print(f"{prefix} PnP successful with {len(inliers)} inliers out of {len(matched_3d)}")
            print(f"Translation: {t}")
            rotation_angles = np.degrees(cv2.Rodrigues(R)[0].flatten())
            print(f"Rotation angles (degrees): {rotation_angles}")

            projected_points, _ = cv2.projectPoints(
                matched_3d[inliers.flatten()],
                cv2.Rodrigues(R)[0],
                t.reshape(-1, 1),
                self.K_left,
                None
            )
            reprojection_error = np.mean(np.linalg.norm(
                projected_points.reshape(-1, 2) - matched_2d[inliers.flatten()],
                axis=1
            ))
            print(f"Mean reprojection error: {reprojection_error:.2f} pixels")

        return R, t, inliers, (matched_3d, matched_2d)
