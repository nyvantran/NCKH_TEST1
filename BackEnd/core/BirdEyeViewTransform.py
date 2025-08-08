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

    def set_src_points_by_monitor(self, img_path):
        """
        Set the source points by selecting them on the monitor.

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
                draw_points(display_image, self.points)

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


def draw_points(image, points, color=(0, 0, 255), radius=8, thickness=-1):
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


def main():
    # Example usage

    transformer = BirdEyeViewTransform()
    # # transformer.set_src_points_by_monitor("frame_21.jpg")  # Replace with your image path
    # w, h = np.array([5, 4]) * 0.4
    # target_corners = [[0, 0], [w, 0], [w, h], [0, h]]  # Define target corners
    # # transformer.set_hography_matrix(target_corners=target_corners)
    # # transformer.save_config_BEV('config_BEV_CAM0010.json')
    transformer.load_config_BEV(r'D:\Project\NCKH\NCKH_TEST1\config\config_BEV_CAM001.json')
    transformer.demo("frame_21.jpg")  # Replace with your image path

    # point1_px, point2_px = (100, 150), (200, 250)  # ví dụ tọa độ 2 điểm
    # p1_px, p2_px = np.array(point1_px, dtype='float32'), np.array(point2_px, dtype='float32')
    # hography_matrix = np.array([[3.43724676e-02, 1.43550909e-02, -2.16796165e+01],
    #                             [3.43285291e-04, 1.10537864e-01, -2.60488312e+01],
    #                             [1.25993357e-04, 1.21887670e-02, 1.00000000e+00]])  # ví dụ ma trận homography
    # point_src_reshaped = np.array([[p1_px, p2_px]], dtype='float32')  # chuyển đổi thành định dạng phù hợp
    # point_dst_transformed = cv2.perspectiveTransform(point_src_reshaped, hography_matrix)  # áp dụng biến đổi phối cảnh
    # Trích xuất tọa độ từ kết quả
    # pdt1 = point_dst_transformed[0][0]  # tọa độ điểm 1 sau biến đổi
    # pdt2 = point_dst_transformed[0][1]  # tọa độ điểm 2 sau biến đổi

    # pdt1 = np.array([100, 150], dtype='float32')  # ví dụ tọa độ điểm 1 sau biến đổi
    # pdt2 = np.array([200, 250], dtype='float32')  # ví dụ tọa độ điểm 2 sau biến đổi
    # distance = cv2.norm(pdt1, pdt2)  # khoảng cách giữa hai điểm


if __name__ == "__main__":
    main()
