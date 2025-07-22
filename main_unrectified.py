import numpy as np
import matplotlib.pyplot as plt
from src.core.feature_extractor import XFeat
from src.core.rectification import StereoRectifier
from src.core.outlier_removal import remove_statistical_outliers

# Import the modular components
from src.modules.frame_loader import FrameLoader
from src.modules.stereo_matcher import StereoMatcher
from src.modules.triangulator import Triangulator
from src.modules.pnp import PnP  

from src.utils.visualisation import Visualiser


if __name__ == "__main__":
    # --- Calibration and dataset paths ---

    # local paths for the developer: comment out the next two lines
    #left_images_path = "/home/leroy-marewangepo/Masters_Stuff/dataset-stereo/dso/cam0/images"
    #right_images_path = "/home/leroy-marewangepo/Masters_Stuff/dataset-stereo/dso/cam1/images"

    # relatives paths for users: uncomment the next two lines
    left_images_path = "dataset-stereo/dso/cam0/images"
    right_images_path = "dataset-stereo/dso/cam1/images"

    # If you want to use your own dataset, change the paths above to your dataset paths

    images = [left_images_path, right_images_path]

    K_left = np.array([[380.8104, 0, 510.2947],
                                [0, 380.8119, 514.3305],
                                [0, 0, 1]])
            
    K_right = np.array([[379.2870, 0, 505.5667],
                                [0, 379.2658, 510.2841],
                                [0, 0, 1]])
        
    D_left = np.array([0.0101710798924, -0.0108164400299, 
                            0.00594278176941, -0.00166228466786])
        
    D_right = np.array([0.01371679169245271, -0.015567360615942622,
                                0.00905043103315326, -0.002347858896562788])
        
    R = np.array([[0.9999994317488622, -0.0008361847221513937, -0.0006612844045898121],
                        [0.0008042457277382264, 0.9988989443471681, -0.04690684567228134],
                        [0.0006997790813734836, 0.04690628718225568, 0.9988990492196964]])
        
    T = np.array([-0.10092123225528335, -0.001964540595211977, -0.0014663556043866572])

    image_size = (1024, 1024)  #


    # --- Initialize pipeline components ---
    extractor = XFeat()
    extractor.initiate_model()

    rectifier = StereoRectifier(K_left, K_right, D_left, D_right, R, T, image_size)

    # Get rectified camera matrices 
    K_left_rect, K_right_rect = rectifier.get_rectified_camera_matrices()

    frame_loader = FrameLoader(images, rectifier, max_images=50)
    stereo_matcher = StereoMatcher(extractor, K_left_rect, K_right_rect, rectifier, top_k=1000)  # Use rectified matrices
    triangulator = Triangulator(K_left_rect, K_right_rect, R, T)  # Use rectified matrices
    pnp_solver = PnP(extractor, K_left_rect)  # Use rectified matrix
    visualiser = None  # Will be initialized with first frame




    # --- Processing loop ---
    previous_output_left = None
    previous_3d_points = None
    previous_descriptors = None
    previous_keypoints = None
    previous_feature_indices = None

    all_points = []

    for i, (left_rect, right_rect) in enumerate(frame_loader.get_frame_pairs()):
        # Match stereo pair
        query_idx, train_idx, output_left, output_right = stereo_matcher.match_frames(left_rect, right_rect)
        if query_idx is None:
            continue

        # Triangulate matched features
        pts_3d, pts_2d, feature_indices, valid_indices = triangulator.triangulate(
            output_left, output_right, query_idx, train_idx
        )
        if pts_3d is None:
            continue

        
        all_points.extend(pts_3d)

        if visualiser is None:
             visualiser = Visualiser(left_rect, right_rect, output_left['keypoints'], output_right['keypoints'], [])



        # Perform PnP if we have a previous frame
        if previous_output_left is not None:
            R, t, inliers, _ = pnp_solver.estimate_pose(
                prev_descriptors=previous_descriptors,
                prev_keypoints=previous_keypoints,
                prev_3d_points=previous_3d_points,
                prev_feature_indices=previous_feature_indices,
                curr_descriptors=output_left['descriptors'],
                curr_keypoints=output_left['keypoints'],
                frame_id=i
            )

            
            if R is not None and t is not None:
                # Accumulate the poses 
               # cumulative_t = cumulative_R @ t + cumulative_t
               # cumulative_R = R @ cumulative_R
                
                # Plot the cumulative pose
                visualiser.trajectory_plot(R, t, i)


        # Update for next iteration
        previous_output_left = output_left
        previous_3d_points = pts_3d
        previous_descriptors = output_left['descriptors']
        previous_keypoints = output_left['keypoints']
        previous_feature_indices = feature_indices

        print(f"Processed frame {i+1}")

    # --- Final 3D point visualization ---
    if all_points:
        all_points = np.array(all_points)
        all_points = remove_statistical_outliers(all_points)
        print(f"Triangulated {all_points.shape[0]} total 3D points")
        #visualise_3d_points(all_points)

        # Keep the trajectory plot open
    if visualiser is not None and visualiser.fig is not None:
        print("Trajectory plotting complete. Close the plot window to exit.")
        plt.ioff()  # Turn off interactive mode
        plt.show()  
        