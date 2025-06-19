import os
import cv2
import numpy as np
from feature_extractor import XFeat
from triangulation import triangulate_point
from visualisation import visualise_matches, visualise_3d_points
from rectification import StereoRectifier
from outlier_removal import remove_statistical_outliers
from pipeline import Pipeline, PipelineStep
from steps import FeatureMatchingStep




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
        FeatureMatchingStep(extractor, images, rectifier, top_k=1500, max_images=100)
    ])

    final_data = pipeline.run(data={})