import numpy as np
from src.core.triangulation import triangulate_point



class Triangulator:
    def __init__(self, K_left, K_right, R, T, depth_range=(-12.0, 50.0)):
        # Use the camera matrices passed in (should be rectified when called from execute.py)
        self.K_left = K_left
        self.K_right = K_right
        self.R = R
        self.T = T
        self.depth_min, self.depth_max = depth_range

    def triangulate(self, output_left, output_right, query_idx, train_idx):
        current_3d_points = []
        current_2d_points = []
        feature_indices = []
        valid_indices = []

        for j in range(len(query_idx)):
            pt_left = output_left['keypoints'][query_idx[j]].cpu().numpy().astype(np.float32)
            pt_right = output_right['keypoints'][train_idx[j]].cpu().numpy().astype(np.float32)

            coord = triangulate_point(
                self.K_left, self.K_right, 
                pt_left, pt_right, 
                self.R, self.T
            )

            if coord is not None:
                if self.depth_min < coord[2] < self.depth_max:
                    current_3d_points.append(coord)
                    current_2d_points.append(pt_left)
                    feature_indices.append(query_idx[j])  # For future temporal matching
                    valid_indices.append(j)

        if len(current_3d_points) < 8:
            return None, None, None, None

        return (
            np.array(current_3d_points),
            np.array(current_2d_points),
            np.array(feature_indices),
            valid_indices  # Used for visualization
        )