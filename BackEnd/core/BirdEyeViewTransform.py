import cv2
import numpy as np
import json


class BirdEyeViewTransform:
    def __init__(self, src_points=None, target_size=(640, 480), target_corners=None):
        self.src_points = np.float32(src_points)
        self.target_size = target_size
        self.target_corners = np.float32(target_corners)
        self.__hography_matrix = None
        self.points = []  # List to store points selected by the user

    def apply(self, image):
        """
        Apply the bird's eye view transformation to the input image.

        """
        if self.target_corners is None:
            w, h = self.target_size
            self.target_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        # Compute the perspective transform matrix
        if self.__hography_matrix is None:
            self.__hography_matrix, _ = cv2.findHomography(self.src_points, self.target_corners)
        # self.hography_matrix = cv2.getPerspectiveTransform(self.src_points, self.target_corners)
        # Apply the perspective warp
        transformed_image = cv2.warpPerspective(image, self.__hography_matrix, self.target_size)
        return transformed_image

    def get_hography_matrix(self):
        """
        Get the computed homography matrix.

        Returns:
            np.ndarray: The homography matrix.
        """
        if self.__hography_matrix is None:
            self.__hography_matrix, _ = cv2.findHomography(self.src_points, self.target_corners)
        return self.__hography_matrix

    def set_hography_matrix(self, src_points=None, target_corners=None):
        """
        Set the homography matrix using source points and target corners.

        Args:
            src_points (list): Source points in the original image.
            target_corners (list): Target corners in the transformed image.
        """
        if not src_points is None:
            self.src_points = src_points
        if not target_corners is None:
            self.target_corners = np.float32(target_corners)
        self.__hography_matrix, _ = cv2.findHomography(self.src_points, self.target_corners)

    def calculate_distance(self, point1_px, point2_px):
        """
        Calculate the distance between two points in the transformed image.

        Args:
            point1_px (tuple): Coordinates of the first point in pixel space.
            point2_px (tuple): Coordinates of the second point in pixel space.

        Returns:
            float: The distance between the two points in the transformed image.
        """
        if self.__hography_matrix is None:
            self.__hography_matrix = self.set_hography_matrix()

        p1_px, p2_px = np.array(point1_px, dtype='float32'), np.array(point2_px, dtype='float32')

        point_src_reshaped = np.array([[p1_px, p2_px]], dtype='float32')
        point_dst_transformed = cv2.perspectiveTransform(point_src_reshaped, self.__hography_matrix)
        # Trích xuất tọa độ từ kết quả
        pdt1 = point_dst_transformed[0][0]
        pdt2 = point_dst_transformed[0][1]
        distance = cv2.norm(pdt1, pdt2)
        return distance

    def save_config_BEV(self, filename):
        """
        Save the configuration of the BirdEyeViewTransform1 object to a file.

        Args:
            filename (str): The name of the file to save the configuration.
        """
        config = {
            'src_points': self.src_points.tolist(),
            'target_size': self.target_size,
            'target_corners': self.target_corners.tolist() if self.target_corners is not None else None,
            'hography_matrix': self.get_hography_matrix().tolist()
        }
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)

    def load_config_BEV(self, filename):
        """
        Load the configuration of the BirdEyeViewTransform1 object from a file.

        Args:
            filename (str): The name of the file to load the configuration from.
        """
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
                self.src_points = np.float32(config['src_points'])
                self.target_size = tuple(config['target_size'])
                self.target_corners = np.float32(config['target_corners']) if config['target_corners'] else None
                self.__hography_matrix = np.float32(config['hography_matrix'])
                print(f"loaded file {filename} configuration")
        except FileNotFoundError:
            print(f"Không tìm thấy file cấu hình: {filename}")

    def mouse_handler1(self, event, x, y, flags, param):
        """Hàm xử lý sự kiện nhấp chuột để chọn điểm."""

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                print(f"Đã chọn điểm {len(self.points)}: ({x}, {y})")

    def mouse_handler2(self, event, x, y, flags, param):
        """Hàm xử lý sự kiện nhấp chuột để chọn điểm."""

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
                print(f"Đã chọn điểm {len(self.points)}: ({x}, {y})")

    def demo(self, img_path):
        """
        Set the points selected by the user.

        Args:
            img_path (str): Path to the image to load.
        """
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Không tìm thấy file ảnh: {img_path}")
            clone = image.copy()
        except FileNotFoundError as e:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            clone = image.copy()
            cv2.putText(clone, "Image not found", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_handler1)
        print("Nhấp chuột vào 2 điểm trên ảnh để đo khoảng cách.")
        print("Nhấn 'r' để chọn lại điểm. Nhấn 'q' để thoát.")

        while True:
            # Vẽ các điểm đã chọn và đường nối
            display_image = clone.copy()
            if len(self.points) > 0:
                for point in self.points:
                    cv2.circle(display_image, point, 5, (0, 255, 0), -1)

            if len(self.points) == 2:
                cv2.line(display_image, self.points[0], self.points[1], (0, 0, 255), 2)

                # Tính toán và hiển thị khoảng cách
                distance = self.calculate_distance(self.points[0], self.points[1])

                # Hiển thị kết quả lên màn hình
                text = f"Distance: {distance:.2f} meters"  # Giả sử đơn vị là mét
                cv2.putText(display_image, text, (self.points[0][0], self.points[0][1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.imshow("Image", display_image)

            key = cv2.waitKey(1) & 0xFF
            # Nhấn 'r' để reset
            if key == ord('r'):
                self.points = []
                print("Đã xóa các điểm. Vui lòng chọn lại.")
            # Nhấn 'q' để thoát
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

    def set_src_points_by_monitor(self, config):
        """
        Set the source points by selecting them on the monitor.

        Args:
            img_path (str): Path to the image to load.
        """

        try:
            source = config['source']
            if isinstance(source, str) and source.isdigit(): source = int(source)
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise FileNotFoundError(f"Không thể mở video hoặc ảnh: {config['source']}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['frame_width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['frame_height'])
            ret, image = cap.read()
            if image is None:
                raise FileNotFoundError(f"Không tìm thấy file ảnh: {config['source']}")
            clone = cv2.resize(image.copy(), (1280, 720))  # Resize for better visibility
        except FileNotFoundError as e:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            clone = image.copy()
            cv2.putText(clone, "Image not found", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_handler2)
        print("Nhấp chuột vào 2 điểm trên ảnh để đo khoảng cách.")
        print("Nhấn 'r' để chọn lại điểm. Nhấn 'q' để thoát. Nhấn 's' để lưu điểm đã chọn.")
        self.points = []

        while True:
            # Vẽ các điểm đã chọn và đường nối
            display_image = clone.copy()
            if len(self.points) > 0:
                for point in self.points:
                    cv2.circle(display_image, point, 5, (0, 255, 0), -1)

            if len(self.points) == 4:
                self.draw_points(display_image, self.points)

                # Hiển thị kết quả lên màn hình
                text = "Points selected. Press 's' to save."
                cv2.putText(display_image, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.imshow("Image", display_image)

            key = cv2.waitKey(1) & 0xFF
            # Nhấn 'r' để reset
            if key == ord('r'):
                self.points = []
                print("Đã xóa các điểm. Vui lòng chọn lại.")
            # Nhấn 'q' để thoát
            elif key == ord('q'):
                break
            elif key == ord('s'):
                if len(self.points) == 4:
                    self.src_points = np.float32(self.points)
                    print("Đã lưu các điểm đã chọn.")

                    break
                else:
                    print("Vui lòng chọn đủ 4 điểm trước khi lưu.")
        cv2.destroyAllWindows()

    def draw_points(selt, image, points, color=(0, 225, 0), radius=8, thickness=-1):
        """
        Draws points on the image.
        Args:
            image: The input image (numpy array).
            points: List of (x, y) coordinates.
            color: BGR color tuple for the points.
            radius: Radius of the points.
            thickness: Thickness of the points (-1 for filled).
        Returns:
            Image with points drawn.
        """
        for pt in points:
            cv2.circle(image, (int(pt[0]), int(pt[1])), radius, color, thickness)
        return image

    def set_target_corners_by_monitor(self):
        """
        Set the target corners by entering values through a PyQt5 GUI window.
        """
        from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                                     QLabel, QLineEdit, QPushButton, QGridLayout,
                                     QMessageBox, QGroupBox)
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QFont
        import sys

        class TargetCornersDialog(QWidget):
            def __init__(self, parent_transform):
                super().__init__()
                self.transform = parent_transform
                self.entries = []
                self.initUI()

            def initUI(self):
                self.setWindowTitle('Nhập Target Corners')
                self.setFixedSize(450, 400)

                # Main layout
                main_layout = QVBoxLayout()

                # Title
                title = QLabel('Nhập tọa độ 4 góc của target corners')
                title_font = QFont()
                title_font.setPointSize(12)
                title_font.setBold(True)
                title.setFont(title_font)
                title.setAlignment(Qt.AlignCenter)
                main_layout.addWidget(title)

                # Subtitle
                subtitle = QLabel('(Top-Left, Top-Right, Bottom-Right, Bottom-Left)')
                subtitle.setAlignment(Qt.AlignCenter)
                main_layout.addWidget(subtitle)

                # Input group
                input_group = QGroupBox("Tọa độ các góc")
                input_layout = QGridLayout()

                # Headers
                input_layout.addWidget(QLabel("Góc"), 0, 0)
                input_layout.addWidget(QLabel("X (m)"), 0, 1)
                input_layout.addWidget(QLabel("Y (m)"), 0, 2)

                # Create input fields
                corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

                for i in range(4):
                    # Corner name
                    label = QLabel(f"{corner_names[i]}:")
                    input_layout.addWidget(label, i + 1, 0)

                    # X coordinate input
                    x_input = QLineEdit()
                    x_input.setPlaceholderText("X")
                    x_input.setMaximumWidth(100)
                    input_layout.addWidget(x_input, i + 1, 1)

                    # Y coordinate input
                    y_input = QLineEdit()
                    y_input.setPlaceholderText("Y")
                    y_input.setMaximumWidth(100)
                    input_layout.addWidget(y_input, i + 1, 2)

                    self.entries.append([x_input, y_input])

                input_group.setLayout(input_layout)
                main_layout.addWidget(input_group)

                # Current target size info
                info_label = QLabel(f"Target size hiện tại: {self.transform.target_size}")
                info_label.setAlignment(Qt.AlignCenter)
                main_layout.addWidget(info_label)

                # Buttons
                button_layout = QHBoxLayout()

                # Save button
                save_btn = QPushButton("Lưu")
                save_btn.setStyleSheet("QPushButton { background-color: #90EE90; }")
                save_btn.clicked.connect(self.save_corners)
                button_layout.addWidget(save_btn)

                # Cancel button
                cancel_btn = QPushButton("Hủy")
                cancel_btn.setStyleSheet("QPushButton { background-color: #FFB6C1; }")
                cancel_btn.clicked.connect(self.close)
                button_layout.addWidget(cancel_btn)

                main_layout.addLayout(button_layout)

                # Set main layout
                self.setLayout(main_layout)

                # Set default values on startup
                self.set_default_values()

            def set_default_values(self):
                """Set default corner values based on target size"""
                w, h = self.transform.target_size
                default_values = [[0, 0], [w, 0], [w, h], [0, h]]

                for i in range(4):
                    self.entries[i][0].setText(str(default_values[i][0]))
                    self.entries[i][1].setText(str(default_values[i][1]))

            def save_corners(self):
                """Save the entered corner values"""
                try:
                    corners = []
                    for i in range(4):
                        x = float(self.entries[i][0].text())
                        y = float(self.entries[i][1].text())
                        corners.append([x, y])

                    # Validate corners
                    if len(corners) == 4:
                        self.transform.target_corners = np.float32(corners)
                        # Update target size based on corners
                        self.transform.target_size = (
                            int(max(corners[1][0], corners[2][0])),
                            int(max(corners[2][1], corners[3][1]))
                        )

                        QMessageBox.information(self, "Thành công",
                                                "Đã lưu target corners thành công!")
                        self.close()
                    else:
                        QMessageBox.warning(self, "Lỗi", "Vui lòng nhập đủ 4 điểm!")

                except ValueError:
                    QMessageBox.critical(self, "Lỗi", "Vui lòng nhập số hợp lệ!")

        # Create QApplication if it doesn't exist
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # Create and show dialog
        dialog = TargetCornersDialog(self)
        dialog.show()

        # Run the application event loop
        app.exec_()


def main():
    # Example usage
    import os
    config_dir = r"config"  # replace with your config file path
    config = json.load(open(os.path.join(config_dir, "cameras.json"), "r"))  # replace with your config file path
    config = config.get("cameras")
    for i, camera in enumerate(config):
        print(
            f"Config Camera {i + 1}: {camera['camera_id']}, Source: {camera['source']}, Position: {camera['position']}")

        camera_config = config[i]  # Assuming you want to use the first camera configuration
        transformer = BirdEyeViewTransform()
        transformer.set_src_points_by_monitor(camera_config)  # Replace with your image path
        transformer.set_target_corners_by_monitor()  # Set target corners through GUI
        transformer.save_config_BEV(
            os.path.join(config_dir, f"config_BEV_{camera["camera_id"]}.json"))  # Save configuration to file
    # transformer.load_config_BEV(r'config_BEV_CAM001.json')
    # transformer.demo("config.jpg")  # Replace with your image path


if __name__ == "__main__":
    main()
