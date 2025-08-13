import threading
import time
import cv2
import numpy as np
import logging
from typing import List, Dict
from BackEnd.core.BatchProcessor import BatchProcessor
from BackEnd.data.DatabaseManager import DatabaseManager
from BackEnd.core.PersonTracker import PersonTracker
from BackEnd.common.DataClass import CameraConfig, DetectionResult
from datetime import datetime
import os
import BackEnd.config as config


class ImprovedCameraWorker(threading.Thread):
    def __init__(self, config: CameraConfig, batch_processor: BatchProcessor, db_manager: DatabaseManager):
        super().__init__()
        self.config = config
        self.batch_processor = batch_processor
        self.db_manager = db_manager  # Giữ lại để có thể ghi log nếu cần
        self.running = False
        self.tracker = PersonTracker(config.camera_id, config)
        self.logger = logging.getLogger(f"Camera-{config.camera_id}")
        self.cap = None
        self.frame_count = 0
        self.latest_frame = None
        self.latest_frame_lock = threading.Lock()
        self.is_active = True
        self.is_video_file = isinstance(config.source, str) and not config.source.isdigit()

    def run(self):
        self.running = True
        self.logger.info(f"Starting camera {self.config.camera_id}")
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self._open_video_source()
                if not self.cap or not self.cap.isOpened():
                    self.logger.error(f"Cannot open camera source: {self.config.source}. Retrying in 5s.")
                    self.is_active = False
                    time.sleep(5)
                    continue
            self.is_active = True
            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_file and self.config.loop_video:
                    self.logger.info(f"Restarting video file for {self.config.camera_id}.")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.cap.release()
                    continue
                else:
                    self.logger.info(f"End of video or stream error for {self.config.camera_id}.")
                    self.is_active = False
                    break
            self.frame_count += 1
            with self.latest_frame_lock:
                self.latest_frame = frame.copy()
            metadata = {'frame_id': self.frame_count, 'timestamp': time.time(),
                        'confidence_threshold': self.config.confidence_threshold}
            self.batch_processor.add_frame(self.config.camera_id, frame, metadata)
            time.sleep(1 / 30)  # Giới hạn FPS
        self.cleanup()

    def _open_video_source(self):
        try:
            source = self.config.source
            if isinstance(source, str) and source.isdigit(): source = int(source)
            self.cap = cv2.VideoCapture(source)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
                print(
                    f"Opening source {self.config.source} with resolution {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.tracker.current_fps = max(fps, 1) if fps > 0 else 30
                self.logger.info(f"Source {self.config.source} opened. FPS: {self.tracker.current_fps}")
        except Exception as e:
            self.logger.error(f"Error opening source {self.config.source}: {e}")
            self.cap = None

    def get_latest_frame(self):
        with self.latest_frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def process_detections(self, detections: List[Dict], frame: np.ndarray):
        self.tracker.update_tracks(detections)
        newly_warned_pairs = self.tracker.monitor_distances_and_draw(frame)
        result = DetectionResult(
            camera_id=self.config.camera_id,
            frame_id=self.frame_count,
            timestamp=time.time(),
            detections=detections,
            close_pairs=newly_warned_pairs,
            frame=frame
        )
        file_name = datetime.now().strftime("%d-%m-%Y %H-%M-%S" + ".jpg")
        # Lưu khung hình hiện tại (frame) vào biến image
        if newly_warned_pairs:
            print(f"Camera {self.config.camera_id} detected {len(self.tracker.warned_pairs)} violations.")
            for (id1, id2) in self.tracker.warned_pairs:
                self.db_manager.log_event(self.config.camera_id, "violation", id1, id2, file_name)

            save_dir = os.path.join(config.dir_capture)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, file_name)
                try:
                    cv2.imwrite(save_path, frame)
                    self.logger.info(f"Saved violation frame to {save_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save violation frame: {e}")

        if self.frame_count % 300 == 0:
            stats = self.tracker.get_statistics()
            self.db_manager.log_statistics(
                self.config.camera_id,
                stats['total_tracks'],  # Tổng số lượng track theo dõi
                stats['active_tracks'],  # Số lượng track tại khung hình hiện tại phát hiện được
                stats['violations']  # Số lượng cặp đối tượng vi phạm khoảng cách
            )

        return result

    def stop(self):
        self.running = False

    def cleanup(self):
        if self.cap: self.cap.release()
        self.logger.info(f"Camera {self.config.camera_id} stopped.")
