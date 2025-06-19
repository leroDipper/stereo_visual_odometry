import os
import cv2


class FrameLoader:
    def __init__(self, images_path, rectifier, max_images=None):
        self.images_path = images_path
        self.rectifier = rectifier
        self.max_images = max_images

    def get_frame_pairs(self):
        left_images = sorted([os.path.join(self.images_path[0], f)
                              for f in os.listdir(self.images_path[0])
                              if f.endswith(('.png', '.jpg', '.jpeg'))])
        right_images = sorted([os.path.join(self.images_path[1], f)
                               for f in os.listdir(self.images_path[1])
                               if f.endswith(('.png', '.jpg', '.jpeg'))])

        if self.max_images is not None:
            left_images = left_images[:self.max_images]
            right_images = right_images[:self.max_images]

        for left_path, right_path in zip(left_images, right_images):
            left_frame = cv2.imread(left_path, cv2.IMREAD_COLOR)
            right_frame = cv2.imread(right_path, cv2.IMREAD_COLOR)

            if left_frame is None or right_frame is None:
                continue

            left_rect, right_rect = self.rectifier.rectify_stereo_pair(left_frame, right_frame)
            yield left_rect, right_rect