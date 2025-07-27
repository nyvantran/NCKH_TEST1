# gui_app.py

import sys
import os
import json
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
                             QGridLayout, QHeaderView, QMessageBox, QSizePolicy)  # Thêm QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt, QThread

# Import lớp hệ thống từ file backend
from BackEnd.MultiCameraSurveillanceSystem import MultiCameraSurveillanceSystem


class SystemThread(QThread):
    """
    Một luồng riêng để chạy hệ thống backend, giữ cho GUI không bị đóng băng.
    """

    def __init__(self, system: MultiCameraSurveillanceSystem):
        super().__init__()
        self.system = system

    def run(self):
        self.system.start()
        while self.system.running:
            self.msleep(100)

    def stop(self):
        if self.system.running:
            self.system.stop()


class SurveillanceGUI(QMainWindow):
    def __init__(self, config_file="cameras.json"):
        super().__init__()

        # Tạo hệ thống backend
        self.system = MultiCameraSurveillanceSystem(config_file=config_file, batch_size=4)

        # Di chuyển hệ thống sang một luồng riêng
        self.system_thread = SystemThread(self.system)

        self.camera_labels = {}
        self.initUI()
        self.connect_signals()

        # Bắt đầu luồng hệ thống
        self.system_thread.start()

    def initUI(self):
        self.setWindowTitle("Hệ thống Giám sát Đa camera - Giao diện Điều khiển")
        self.setGeometry(100, 100, 1600, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- BÊN TRÁI: KHUNG CAMERA ---
        self.camera_container = QWidget()  # Đổi thành thuộc tính của self
        self.camera_grid_layout = QGridLayout(self.camera_container)
        self.camera_grid_layout.setContentsMargins(0, 0, 0, 0)  # Xóa khoảng trống
        self.camera_grid_layout.setSpacing(5)  # Thêm một chút khoảng cách giữa các camera

        num_cameras = len(self.system.cameras)
        if num_cameras == 0:
            no_cam_label = QLabel("Không có camera nào được cấu hình trong 'cameras.json'.")
            no_cam_label.setAlignment(Qt.AlignCenter)
            no_cam_label.setStyleSheet("font-size: 20px; color: red;")
            self.camera_grid_layout.addWidget(no_cam_label, 0, 0)
        else:
            grid_cols = 2 if num_cameras > 2 else num_cameras if num_cameras > 0 else 1
            sorted_camera_ids = sorted(self.system.cameras.keys())
            for i, camera_id in enumerate(sorted_camera_ids):
                row, col = divmod(i, grid_cols)

                # *** SỬA LỖI: Tạo một container cho mỗi camera ***
                cam_widget = QWidget()
                cam_layout = QVBoxLayout(cam_widget)
                cam_layout.setContentsMargins(0, 0, 0, 0)

                cam_label = QLabel(f"Đang chờ tín hiệu từ {camera_id}...")
                cam_label.setAlignment(Qt.AlignCenter)
                cam_label.setStyleSheet("background-color: black; color: white; font-size: 18px;")

                # *** SỬA LỖI: Thiết lập SizePolicy ***
                # Expanding: cho phép widget giãn ra để lấp đầy không gian.
                # Ignored: bỏ qua kích thước gợi ý (size hint) và lấp đầy không gian được cấp.
                size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                cam_label.setSizePolicy(size_policy)

                cam_layout.addWidget(cam_label)
                self.camera_labels[camera_id] = cam_label
                self.camera_grid_layout.addWidget(cam_widget, row, col)

        main_layout.addWidget(self.camera_container, 3)  # Chiếm 3/4 không gian

        # --- BÊN PHẢI: BẢNG ĐIỀU KHIỂN VÀ LOG ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        right_layout.addWidget(QLabel("<h2>Nhật ký Tiếp xúc</h2>"))
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(6)
        self.log_table.setHorizontalHeaderLabels(["Thời gian", "Camera", "ID 1", "ID 2", "Khoảng cách (m)", "TGTX"])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.log_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.log_table.setEditTriggers(QTableWidget.NoEditTriggers)

        right_layout.addWidget(self.log_table)
        main_layout.addWidget(right_panel, 1)  # Chiếm 1/4 không gian

    def connect_signals(self):
        self.system.new_frame_ready.connect(self.update_camera_feed)
        self.system.violation_detected.connect(self.add_violation_log)
        self.system_thread.finished.connect(self.on_system_thread_finished)
        self.system.system_stopped.connect(self.system_thread.quit)

    def update_camera_feed(self, camera_id, frame):
        if camera_id in self.camera_labels:
            label_to_update = self.camera_labels[camera_id]
            try:
                h, w, ch = frame.shape
                # *** SỬA LỖI: Thay đổi cách tính toán và co giãn pixmap ***
                # 1. Chuyển đổi frame sang QImage
                qt_image = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)

                # 2. Tạo QPixmap từ QImage
                pixmap = QPixmap.fromImage(qt_image)

                # 3. Co giãn pixmap để vừa với kích thước của QLabel, *giữ nguyên tỷ lệ khung hình*
                #    Đây là phần quan trọng nhất. Nó đảm bảo ảnh không bị méo và vừa với không gian đã được cấp.
                scaled_pixmap = pixmap.scaled(label_to_update.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # 4. Đặt pixmap đã co giãn vào label
                label_to_update.setPixmap(scaled_pixmap)

            except Exception as e:
                print(f"Error updating feed for {camera_id}: {e}")

    def add_violation_log(self, camera_id, id1, id2, distance, timestamp, timeclose=1):
        row_position = 0
        self.log_table.insertRow(row_position)
        self.log_table.setItem(row_position, 0, QTableWidgetItem(timestamp))
        self.log_table.setItem(row_position, 1, QTableWidgetItem(camera_id))
        self.log_table.setItem(row_position, 2, QTableWidgetItem(str(id1)))
        self.log_table.setItem(row_position, 3, QTableWidgetItem(str(id2)))
        self.log_table.setItem(row_position, 4, QTableWidgetItem(f"{distance:.2f}"))
        self.log_table.setItem(row_position, 5, QTableWidgetItem(f"{timeclose:.2f}s"))

        for i in range(5):
            self.log_table.item(row_position, i).setBackground(QColor(255, 100, 100, 100))

    def on_system_thread_finished(self):
        QMessageBox.information(self, "Thông báo", "Hệ thống xử lý đã dừng.")

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Thoát Ứng dụng',
                                     "Bạn có chắc muốn thoát không? Hệ thống sẽ dừng lại.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            print("Closing application...")
            self.system_thread.stop()
            self.system_thread.wait(5000)
            event.accept()
        else:
            event.ignore()


def create_default_config():
    if not os.path.exists("../config/cameras.json"):
        print("Creating default 'cameras.json' file.")
        config = {
            "cameras": [
                {
                    "camera_id": "CAM001",
                    "source": "0",
                    "position": "Entrance",
                    "enable_recording": False,
                    "recording_path": "./recordings",
                    "confidence_threshold": 0.4,
                    "social_distance_threshold": 2.0,
                    "warning_duration": 1.0,
                    "loop_video": True
                },
                {
                    "camera_id": "CAM002",
                    "source": "D:\\WorkSpace\\PersonPath22\\tracking-dataset\\dataset\\dataset1\\raw_data\\uid_vid_00000.mp4",
                    "position": "Hall",
                    "enable_recording": False,
                    "recording_path": "./recordings",
                    "confidence_threshold": 0.4,
                    "social_distance_threshold": 2.0,
                    "warning_duration": 1.0,
                    "loop_video": True
                }
            ]
        }
        with open("/config/cameras.json", "w") as f:
            json.dump(config, f, indent=4)


def main():
    # create_default_config()
    app = QApplication(sys.argv)
    main_window = SurveillanceGUI(config_file="config/cameras.json")
    main_window.show()
    sys.exit(app.exec_())
