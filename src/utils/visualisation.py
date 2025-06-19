import cv2
import matplotlib.pyplot as plt

def visualise_matches(left_frame, right_frame, kp_left, kp_right, matches, max_display=400):
    matches_subset = matches[:min(max_display, len(matches))]
    match_img = cv2.drawMatches(left_frame, kp_left,
                                right_frame, kp_right,
                                matches_subset, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Stereo Matches", match_img)
    cv2.waitKey(1)

def visualise_3d_points(points_3d):
    print(f"Visualising {points_3d.shape[0]} 3D points")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='r', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Triangulated 3D Points")
    plt.show()
