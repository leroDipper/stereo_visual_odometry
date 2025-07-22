import numpy as np
import matplotlib.pyplot as plt

from src.core.feature_extractor import XFeat
from src.core.outlier_removal import remove_statistical_outliers

from src.modules.frame_loader import FrameLoader
from src.modules.stereo_matcher import StereoMatcher
from src.modules.triangulator import Triangulator
from src.modules.pnp import PnP
from src.utils.visualisation import Visualiser

if __name__ == "__main__":
    # --- Dataset paths ---
    left_images_path = "dataset-stereo/dso/cam0/images"
    right_images_path = "dataset-stereo/dso/cam1/images"
    images = [left_images_path, right_images_path]

    # --- EuRoC camera intrinsics (already rectified images) ---
    K_left = np.array([[380.8104, 0, 510.2947],
                       [0, 380.8119, 514.3305],
                       [0, 0, 1]])

    K_right = np.array([[379.2870, 0, 505.5667],
                        [0, 379.2658, 510.2841],
                        [0, 0, 1]])

    R = np.array([[0.99999943, -0.00083618, -0.00066128],
                  [0.00080424, 0.99889894, -0.04690684],
                  [0.00069977, 0.04690628, 0.99889905]])

    T = np.array([-0.10092123, -0.00196454, -0.00146635])

    # --- Initialize pipeline components ---
    extractor = XFeat()
    extractor.initiate_model()

    frame_loader = FrameLoader(images, max_images=50)  # No rectification needed
    stereo_matcher = StereoMatcher(extractor, K_left, K_right, rectifier=None, top_k=1000)
    triangulator = Triangulator(K_left, K_right, R, T)
    pnp_solver = PnP(extractor, K_left)

    visualiser = None
    cumulative_R = np.eye(3)
    cumulative_t = np.zeros((3, 1))

    # --- Tracking state ---
    previous_output_left = None
    previous_3d_points = None
    previous_descriptors = None
    previous_keypoints = None
    previous_feature_indices = None
    all_points = []

    # --- Main processing loop ---
    for i, (left_rect, right_rect) in enumerate(frame_loader.get_frame_pairs()):
        query_idx, train_idx, output_left, output_right = stereo_matcher.match_frames(left_rect, right_rect)
        if query_idx is None:
            continue

        pts_3d, pts_2d, feature_indices, valid_indices = triangulator.triangulate(
            output_left, output_right, query_idx, train_idx
        )
        if pts_3d is None:
            continue

        all_points.extend(pts_3d)

        if visualiser is None:
            visualiser = Visualiser(left_rect, right_rect, output_left['keypoints'], output_right['keypoints'], [])

        # --- Pose estimation ---
        if previous_output_left is not None:
            R_rel, t_rel, inliers, _ = pnp_solver.estimate_pose(
                prev_descriptors=previous_descriptors,
                prev_keypoints=previous_keypoints,
                prev_3d_points=previous_3d_points,
                prev_feature_indices=previous_feature_indices,
                curr_descriptors=output_left['descriptors'],
                curr_keypoints=output_left['keypoints'],
                frame_id=i
            )

            if R_rel is not None and t_rel is not None:
                # Accumulate pose
                #cumulative_t = cumulative_R @ t_rel + cumulative_t
                #cumulative_R = R_rel @ cumulative_R

                visualiser.trajectory_plot(R_rel, t_rel, i)

        # Update for next iteration
        previous_output_left = output_left
        previous_3d_points = pts_3d
        previous_descriptors = output_left['descriptors']
        previous_keypoints = output_left['keypoints']
        previous_feature_indices = feature_indices

        print(f"Processed frame {i + 1}")

    # --- Final 3D visualization ---
    if all_points:
        all_points = np.array(all_points)
        all_points = remove_statistical_outliers(all_points)
        print(f"Triangulated {all_points.shape[0]} total 3D points")

    if visualiser is not None and visualiser.fig is not None:
        print("Trajectory plotting complete. Close the plot window to exit.")
        plt.ioff()
        plt.show()
