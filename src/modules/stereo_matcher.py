import cv2
import numpy as np


class StereoMatcher:
    def __init__(self, extractor, K_left, K_right, rectifier, top_k=None):
        self.extractor = extractor
        self.K_left = K_left   # Should be rectified matrices
        self.K_right = K_right  # Should be rectified matrices
        self.rectifier = rectifier
        self.top_k = top_k

    def match_frames(self, img_left, img_right):
        img_tensor_left = self.extractor.convert_to_tensor(img_left)
        img_tensor_right = self.extractor.convert_to_tensor(img_right)

        output_left = self.extractor.detect_and_compute(img_tensor_left, top_k=self.top_k)
        output_right = self.extractor.detect_and_compute(img_tensor_right, top_k=self.top_k)

        if (output_left is None or output_right is None or 
            'descriptors' not in output_left or 'descriptors' not in output_right):
            return None, None, None, None

        query_idx, train_idx = self.extractor.match(
            output_left['descriptors'],
            output_right['descriptors'],
            min_cossim=0.7
        )

        if len(query_idx) < 8:
            return None, None, None, None

        # RANSAC fundamental matrix filtering
        pts_left = output_left['keypoints'][query_idx].cpu().numpy().astype(np.float32)
        pts_right = output_right['keypoints'][train_idx].cpu().numpy().astype(np.float32)

        F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC, 1.0, 0.99)

        if mask is not None:
            inlier_mask = mask.ravel() == 1
            query_idx = query_idx[inlier_mask]
            train_idx = train_idx[inlier_mask]

            return query_idx, train_idx, output_left, output_right
        else:
            return None, None, None, None
        