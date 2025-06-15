import os
import cv2
import numpy as np
from feature_extractor import XFeat
from triangulation import triangulate_point
from visualisation import visualise_matches, visualise_3d_points
from rectification import StereoRectifier
from outlier_removal import remove_statistical_outliers
from pipeline import Pipeline, PipelineStep


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
        # self.baseline = abs(self.rectifier.T[0])

    def process(self, data):
        left_images = sorted([os.path.join(self.images_path[0], f) for f in os.listdir(self.images_path[0]) if f.endswith(('.png', '.jpg', '.jpeg'))])
        right_images = sorted([os.path.join(self.images_path[1], f) for f in os.listdir(self.images_path[1]) if f.endswith(('.png', '.jpg', '.jpeg'))])

        if self.max_images is not None:
            left_images = left_images[:self.max_images]
            right_images = right_images[:self.max_images]

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

            query_idx, train_idx = self.extractor.match(
                output_left['descriptors'], 
                output_right['descriptors'], 
                min_cossim=0.7
            )

            
            if len(query_idx) > 8:
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
                


            if len(train_idx) == 0 or len(query_idx) == 0:
                continue

            kp_left = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in output_left['keypoints'][query_idx]]
            kp_right = [cv2.KeyPoint(float(x[0]), float(x[1]), 1) for x in output_right['keypoints'][train_idx]]
            matches_lr = [cv2.DMatch(_queryIdx=j, _trainIdx=j, _distance=0) for j in range(len(kp_left))]

            frame_pairs.append((kp_left, kp_right, matches_lr))

            for j in range(len(matches_lr)):
                coord = triangulate_point(
                    self.K_left, self.K_right, 
                    kp_left[j].pt, kp_right[j].pt, self.rectifier.R, self.rectifier.T
                )
                
                if coord is not None:
                    x.append(coord[0])
                    y.append(coord[1])
                    z.append(coord[2])
            


            visualise_matches(left_frame_rect, right_frame_rect, kp_left, kp_right, matches_lr)
            print(f"Processed frame {i+1}/{len(left_images)}")
            #if i % 10 == 0:
                #visualise_matches(left_frame_rect, right_frame_rect, kp_left, kp_right, matches_lr)

        if len(x) > 0:
            self.points_3d = np.stack([x, y, z], axis=1)
            self.points_3d = remove_statistical_outliers(self.points_3d)  # Add this line
            print(f"Triangulated {self.points_3d.shape[0]} 3D points")
            visualise_3d_points(self.points_3d)

        data['frame_pairs'] = frame_pairs
        data['points_3d'] = self.points_3d
        return data

if __name__ == "__main__":
    left_images_path = "/home/leroy-marewangepo/Masters_Stuff/dataset-room1_1024/dso/cam0/images"
    right_images_path = "/home/leroy-marewangepo/Masters_Stuff/dataset-room1_1024/dso/cam1/images"
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



    extractor = XFeat()
    extractor.initiate_model()
    
    rectifier = StereoRectifier(K_left, K_right, D_left, D_right, R, T, image_size)
    
    pipeline = Pipeline(steps=[
        FeatureMatchingStep(extractor, images, rectifier, top_k=1000, max_images=1500)
    ])

    final_data = pipeline.run(data={})