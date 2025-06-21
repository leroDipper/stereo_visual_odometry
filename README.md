# Stereo Vision Pipeline

A modular stereo vision pipeline for 3D reconstruction and visual odometry using XFeat feature extraction, stereo rectification, triangulation, and PnP pose estimation.

## Features

- **Feature Extraction**: XFeat deep learning-based feature detector and descriptor
- **Stereo Rectification**: Fisheye camera model support with stereo rectification
- **3D Triangulation**: Robust triangulation with depth filtering and outlier removal
- **Visual Odometry**: Frame-to-frame pose estimation using PnP with RANSAC
- **Modular Design**: Clean separation of concerns with reusable components
- **Visualization**: 3D point cloud visualization and stereo match visualization

## Project Structure

```
├── execute.py                    # Main execution script
├── src/
│   ├── core/                     # Core algorithms
│   │   ├── feature_extractor.py  # XFeat wrapper
│   │   ├── rectification.py      # Stereo rectification
│   │   ├── triangulation.py      # 3D point triangulation
│   │   ├── pose_estimation.py    # PnP pose estimation
│   │   └── outlier_removal.py    # Statistical outlier removal
│   ├── modules/                  # Pipeline modules
│   │   ├── frame_loader.py       # Image loading and preprocessing
│   │   ├── stereo_matcher.py     # Stereo feature matching
│   │   ├── triangulator.py       # Triangulation module
│   │   └── pnp.py                # PnP pose estimation module
│   ├── pipeline/                 # Pipeline framework
│   │   ├── pipeline.py           # Base pipeline classes
│   │   └── steps.py              # Pipeline step implementations
│   └── utils/                    # Utilities
│       └── visualisation.py     # Visualization functions
└── README.md
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for XFeat)
- OpenCV 4.5+
- PyTorch 1.9+
- NumPy, SciPy, Matplotlib

See `requirements.txt` for complete dependencies.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stereo_visual_odometry
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. The XFeat model will be automatically downloaded from PyTorch Hub on the first run.

## Usage

### Basic Usage

1. **Prepare your dataset**: Organise stereo image pairs in separate directories:
```
dataset/
├── cam0/images/  # Left camera images
└── cam1/images/  # Right camera images
```

2. **Configure camera calibration**: Update the camera matrices and distortion parameters in `execute.py`:
```python
K_left = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
K_right = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
D_left = np.array([k1, k2, k3, k4])  # Fisheye distortion
D_right = np.array([k1, k2, k3, k4])
R = np.array([...])  # Rotation between cameras
T = np.array([...])  # Translation between cameras
```

3. **Update image paths**:
Code is currently set to work with the developer's local dataset, uncomment the
   paths for the dataset for users
```python
 # local paths for the developer: comment out the next two lines
    left_images_path = "/home/leroy-marewangepo/Masters_Stuff/dataset-stereo/dso/cam0/images"
    right_images_path = "/home/leroy-marewangepo/Masters_Stuff/dataset-stereo/dso/cam1/images"

    # relatives paths for users: uncomment the next two lines
    # left_images_path = "dataset-stereo/dso/cam0/images"
    # right_images_path = "dataset-stereo/dso/cam1/images"

    # If you want to use your own dataset, change the paths above to your dataset paths
```
or  
```python
left_images_path = "path/to/cam0/images"
right_images_path = "path/to/cam1/images"
```

4. **Run the pipeline**:
```bash
python execute.py
```

### Module Usage

You can also use individual components:

```python
from src.core.feature_extractor import XFeat
from src.core.rectification import StereoRectifier
from src.modules.triangulator import Triangulator

# Initialize components
extractor = XFeat()
extractor.initiate_model()

rectifier = StereoRectifier(K_left, K_right, D_left, D_right, R, T, image_size)
triangulator = Triangulator(K_left_rect, K_right_rect, R, T)

# Process images
left_rect, right_rect = rectifier.rectify_stereo_pair(left_img, right_img)
# ... feature extraction and matching
pts_3d, pts_2d, feature_indices, valid_indices = triangulator.triangulate(
    output_left, output_right, query_idx, train_idx
)
```

## Configuration Parameters

### Feature Extraction
- `top_k`: Maximum number of features to extract (default: 1000)
- `min_cossim`: Minimum cosine similarity for feature matching (default: 0.7 for stereo, 0.6 for temporal)

### Triangulation
- `depth_range`: Valid depth range for 3D points (default: 0.1-50.0 meters)
- Disparity filtering: Filters invalid disparities (≤0.5 or >500 pixels)

### Pose Estimation
- `ransac_threshold`: RANSAC reprojection error threshold (default: 5.0 pixels)
- `confidence`: RANSAC confidence level (default: 0.95)
- `max_iterations`: Maximum RANSAC iterations (default: 2000)

### Outlier Removal
- `k`: Number of nearest neighbors for statistical filtering (default: 50)
- `std_ratio`: Standard deviation ratio for outlier threshold (default: 2.0)

## Output

The pipeline generates:

1. **3D Point Cloud**: Triangulated 3D points from stereo pairs
2. **Pose Estimates**: Camera poses for each frame via visual odometry
3. **Visualizations**: 
   - 3D point cloud scatter plot
   - Stereo feature matches (optional)
4. **Console Output**: Processing progress and pose estimation results

## Camera Calibration

The pipeline expects fisheye camera calibration parameters:

- **K_left, K_right**: 3x3 intrinsic camera matrices
- **D_left, D_right**: 4-element fisheye distortion coefficients [k1, k2, k3, k4]
- **R**: 3x3 rotation matrix between cameras
- **T**: 3-element translation vector between cameras
- **image_size**: (width, height) tuple

Use OpenCV's fisheye calibration functions to obtain these parameters.

## Performance Notes

- **GPU Acceleration**: XFeat runs significantly faster on CUDA-enabled GPUs
- **Memory Usage**: Large datasets may require processing in batches
- **Processing Speed**: ~1-2 seconds per stereo pair on modern hardware
- **Feature Count**: Higher `top_k` values improve accuracy but increase computation time

## Troubleshooting

### Common Issues

1. **XFeat model loading fails**: Ensure internet connection for PyTorch Hub download
2. **CUDA out of memory**: Reduce `top_k` parameter or use CPU mode
3. **Few triangulated points**: Check camera calibration and stereo baseline
4. **PnP fails**: Verify sufficient feature matches and 3D point quality

### Debug Output

Enable verbose output in PnP module:
```python
pnp_solver = PnP(extractor, K_left_rect, verbose=True)
```

### Visualisation Issues

If 3D visualisation doesn't appear:
```python
# Add at the end of visualise_3d_points function
plt.show(block=True)  # Force blocking display
```

## Extensions

The modular design allows easy extensions:

- **Trajectory plotting**: Plot camera motion
- **Loop Closure Detection**: Add descriptor database for place recognition
- **Bundle Adjustment**: Integrate global optimization
- **Dense Reconstruction**: Add stereo dense matching
- **Real-time Processing**: Implement streaming pipeline
- **Different Cameras**: Adapt rectification for pinhole cameras
- **Implement Photogrammetry**: Create maps for absolute localisation

## Dependencies

Key dependencies:
- `torch`: XFeat neural network
- `opencv-python`: Computer vision operations
- `numpy`: Numerical computations
- `scipy`: Statistical outlier removal
- `matplotlib`: 3D visualization

## License

[N/A]

## Citation

If you use this code in your research, please cite:
- XFeat: Accelerated Features for Lightweight Image Matching

## Contributing

[N/A]
