import os
import cv2
import numpy as np
from feature_extractor import XFeat
from triangulation import triangulate_point
from visualisation import visualise_matches, visualise_3d_points
from rectification import StereoRectifier
from outlier_removal import remove_statistical_outliers
from pipeline import Pipeline, PipelineStep
from pose_estimation import pose_estimation_ransac



class FeatureMatchingStep(PipelineStep):
    def __init__(self, extractor, images_path, rectifier, top_k=None, max_images=None, min_cossim=-1):
        self.extractor = extractor
        self.images_path = images_path
        self.rectifier = rectifier
        self.top_k = top_k
        self.max_images = max_images
        self.min_cossim = min_cossim
        self.points_3d = []
        
        self.K_left, self.K_right = self.rectifier.get_rectified_camera_matrices()
        

    def process(self, data):
        left_images = sorted([os.path.join(self.images_path[0], f) for f in os.listdir(self.images_path[0]) if f.endswith(('.png', '.jpg', '.jpeg'))])
        right_images = sorted([os.path.join(self.images_path[1], f) for f in os.listdir(self.images_path[1]) if f.endswith(('.png', '.jpg', '.jpeg'))])

        if self.max_images is not None:
            left_images = left_images[:self.max_images]
            right_images = right_images[:self.max_images]

        previous_left = None
        previous_3d_points = None
        frame_pairs = []
        x, y, z = [], [], []

        for i in range(len(left_images)):
            left_frame = cv2.imread(left_images[i], cv2.IMREAD_COLOR)
            right_frame = cv2.imread(right_images[i], cv2.IMREAD_COLOR)

            if left_frame is None or right_frame is None:
                continue

            left_frame_rect, right_frame_rect = self.rectifier.rectify_stereo_pair(left_frame, right_frame)
            
            img_tensor_left = self.extractor.convert_to_tensor(left_frame_rect)
            img_tensor_right = self.extractor.convert_to_tensor(right_frame_rect)

            output_left = self.extractor.detect_and_compute(img_tensor_left, top_k=self.top_k)
            output_right = self.extractor.detect_and_compute(img_tensor_right, top_k=self.top_k)

            if (output_left is None or output_right is None or 
                'descriptors' not in output_left or 'descriptors' not in output_right or
                'keypoints' not in output_left or 'keypoints' not in output_right):
                continue
            
            # Stereo matching
            query_idx, train_idx = self.extractor.match(
                output_left['descriptors'], 
                output_right['descriptors'], 
                min_cossim=0.7
            )

            
            if len(query_idx) < 8:
                continue   
            pts_left = output_left['keypoints'][query_idx].cpu().numpy().astype(np.float32)
            pts_right = output_right['keypoints'][train_idx].cpu().numpy().astype(np.float32)

            # RANSAC filtering
            F, mask = cv2.findFundamentalMat(pts_left, pts_right,
                        cv2.FM_RANSAC,  # method parameter (positional)
                        1.0,            # ransacReprojThreshold  
                        0.99            # confidence
            )

            # Keep only inlier matches
            if mask is not None:
                inlier_mask = mask.ravel() == 1
                query_idx = query_idx[inlier_mask]
                train_idx = train_idx[inlier_mask]
            
            # Triangulate current frame
            current_3d_points = []
            current_2d_points = []
            valid_indices = []

            for j in range(len(query_idx)):
                pt_left = output_left['keypoints'][query_idx[j]].cpu().numpy()
                pt_right = output_right['keypoints'][train_idx[j]].cpu().numpy()

                coord = triangulate_point(
                    self.K_left, self.K_right, 
                    pt_left, pt_right, self.rectifier.R, self.rectifier.T
                )
                
                if coord is not None:
                    current_3d_points.append(coord)
                    current_2d_points.append(pt_left)
                    valid_indices.append(j)
                    x.append(coord[0])
                    y.append(coord[1])
                    z.append(coord[2])

            if len(current_3d_points) < 8:
                continue

            current_3d_points = np.array(current_3d_points)
            current_2d_points = np.array(current_2d_points)

            # Visual odometry with previous frame
            if previous_left is not None and previous_3d_points is not None:
                try:
                    # Match current left frame with previous left frame
                    prev_query_idx, curr_train_idx = self.extractor.match(
                        previous_left['descriptors'],
                        output_right['descriptors'],
                        min_cossin=0.7
                    )

                    if len(prev_query_idx) >= 8:
                        # Find corresponding 3D-2D pairs
                        matched_3d_points = []
                        matched_2d_points = []

                        for prev_idx, curr_idx in zip(prev_query_idx, curr_train_idx):
                             # Check if this feature was triangulated in previous frame
                             if prev_idx < len(previous_3d_points):
                                # Find corresponding 2D point in current frame
                                curr_2d = output_left['keypoints'][curr_idx].cpu().numpy()
                                matched_3d_points.append(previous_3d_points[prev_idx])
                                matched_2d_points.append(curr_2d)
                        
                        if len(matched_3d_points) >= 8:
                            matched_3d_points = np.array(matched_3d_points)
                            matched_2d_points = np.array(matched_2d_points)

                            # Estimate pose using PnP
                            success, R, t, inliers = pose_estimation_ransac(matched_3d_points, matched_2d_points, self.K_right)

                            if success and inliers is not None:
                                print(f"Frame {i}: PnP successful with {len(inliers)} inliers")
                                print(f"Translation: {t}")
                                print(f"Rotation angles (degrees): {np.degrees(cv2.Rodrigues(R)[0].flatten())}")
                            else:
                                print(f"Frame {i}: PnP failed")

                except Exception as e:
                    print(f"Frame {i}: Visual odometry error: {e}")

            # Store current frame data for next iteration
            previous_left = output_left
            previous_3d_points = current_3d_points

            # Create visualization data
            kp_left = [cv2.KeyPoint(float(output_left['keypoints'][query_idx[j]][0]), 
                                   float(output_left['keypoints'][query_idx[j]][1]), 1) 
                      for j in valid_indices]
            kp_right = [cv2.KeyPoint(float(output_right['keypoints'][train_idx[j]][0]), 
                                    float(output_right['keypoints'][train_idx[j]][1]), 1) 
                       for j in valid_indices]
            matches_lr = [cv2.DMatch(_queryIdx=j, _trainIdx=j, _distance=0) 
                         for j in range(len(kp_left))]

            frame_pairs.append((kp_left, kp_right, matches_lr))

            
            if i % 10 == 0:
                visualise_matches(left_frame_rect, right_frame_rect, kp_left, kp_right, matches_lr)
            
            print(f"Processed frame {i+1}/{len(left_images)}")

        if len(x) > 0:
            self.points_3d = np.stack([x, y, z], axis=1)
            self.points_3d = remove_statistical_outliers(self.points_3d)  # Add this line
            print(f"Triangulated {self.points_3d.shape[0]} 3D points")
            visualise_3d_points(self.points_3d)

        data['frame_pairs'] = frame_pairs
        data['points_3d'] = self.points_3d
        return data
