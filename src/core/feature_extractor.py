import torch
import cv2

class XFeat:
    def __init__(self):
        self.xfeat = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def initiate_model(self):
        try:
            self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
            self.xfeat = self.xfeat.to(self.device)
            self.xfeat.eval()
        except Exception as e:
            print(f"Error loading XFeat model: {e}")
            exit()
    
    def convert_to_tensor(self, image):
        if len(image.shape) == 2:
            frame_gray = image
        else:
            frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        frame_gray_expanded = frame_gray[:, :, None]  # Add channel dimension
        frame_gray_expanded = frame_gray_expanded[None, ...]  # Add batch dimension

        img_tensor = self.xfeat.parse_input(frame_gray_expanded)
        return img_tensor
    
    def detect_and_compute(self, img_tensor, top_k=None):
        output = self.xfeat.detectAndCompute(img_tensor, top_k=top_k)[0]
        return output
    
    def match(self, descriptors1, descriptors2, min_cossim=-1):
        """
        Match descriptors between two images
        Returns indices for matching keypoints - exactly like the original working code
        """
        query_idx, train_idx = self.xfeat.match(descriptors1, descriptors2, min_cossim=min_cossim)
        return query_idx, train_idx