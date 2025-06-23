import cv2
import matplotlib.pyplot as plt


class Visualiser:
    def __init__(self, left_frame, right_frame, kp_left, kp_right, matches, points_3d=None):
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.kp_left = kp_left
        self.kp_right = kp_right
        self.matches = matches
        self.points_3d = points_3d

        # Trajectory plot variables
        self.positions = []
        self.frame_ids = []
        self.fig = None
        self.ax = None
        self.trajectory_line = None
        self.current_pos_point = None

    def visualise_matches(self, max_display=400):
        matches_subset = self.matches[:min(max_display, len(self.matches))]
        match_img = cv2.drawMatches(self.left_frame, self.kp_left,
                                    self.right_frame, self.kp_right,
                                    matches_subset, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Stereo Matches", match_img)
        cv2.waitKey(1)

    def visualise_3d_points(self):
        print(f"Visualising {self.points_3d.shape[0]} 3D points")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.points_3d[:, 0], self.points_3d[:, 1], self.points_3d[:, 2], c='r', s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("Triangulated 3D Points")
        plt.show()

    def trajectory_plot(self, R, t, frame_id, real_time=True):
        camera_position = (-R.T @ t).reshape(-1)
        self.positions.append(camera_position)
        self.frame_ids.append(frame_id)

        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_title("Camera Trajectory (Real-Time)")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
            self.trajectory_line, = self.ax.plot([], [], [], 'b-', linewidth=2, label="Trajectory")
            self.current_pos_point, = self.ax.plot([], [], [], 'ro', markersize=6, label="Current Position")
            self.ax.legend()

        if real_time:
            xs, ys, zs = zip(*self.positions)
            self.trajectory_line.set_data(xs, ys)
            self.trajectory_line.set_3d_properties(zs)
            self.current_pos_point.set_data([xs[-1]], [ys[-1]])
            self.current_pos_point.set_3d_properties([zs[-1]])
            
            
            if len(self.positions) > 1:
                # Get the range of your data
                x_range = max(xs) - min(xs)
                y_range = max(ys) - min(ys)
                z_range = max(zs) - min(zs)
                
                # Add some padding around your data
                padding = 0.1  # 10% padding
                x_center = (max(xs) + min(xs)) / 2
                y_center = (max(ys) + min(ys)) / 2
                z_center = (max(zs) + min(zs)) / 2
                
                # Set limits with padding
                self.ax.set_xlim(x_center - x_range/2 - padding, x_center + x_range/2 + padding)
                self.ax.set_ylim(y_center - y_range/2 - padding, y_center + y_range/2 + padding)
                self.ax.set_zlim(z_center - z_range/2 - padding, z_center + z_range/2 + padding)
            
            plt.draw()
            plt.pause(0.001)