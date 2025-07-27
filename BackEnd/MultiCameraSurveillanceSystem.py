import json
import logging
import threading
import time
import numpy as np
from datetime import datetime
from PyQt5.QtCore import QObject, pyqtSignal
from BackEnd.core.ImprovedCameraWorker import ImprovedCameraWorker
from BackEnd.core.BatchProcessor import BatchProcessor
from BackEnd.data.DatabaseManager import DatabaseManager
from BackEnd.common.DataClass import CameraConfig


class MultiCameraSurveillanceSystem(QObject):
    # Tín hiệu để gửi dữ liệu đến GUI một cách an toàn
    new_frame_ready = pyqtSignal(str, np.ndarray)
    violation_detected = pyqtSignal(str, int, int, float, str)
    system_stopped = pyqtSignal()

    def __init__(self, config_file: str = "cameras.json", batch_size: int = 8):
        super().__init__()
        self.config_file = config_file
        self.batch_size = batch_size
        self.cameras = {}
        self.camera_workers = {}
        self.db_manager = DatabaseManager()
        self.running = False
        self.logger = logging.getLogger("SurveillanceSystem")
        self.batch_processor = BatchProcessor(batch_size=self.batch_size)
        self.load_config()

    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            for cam_config in config['cameras']:
                self.cameras[cam_config['camera_id']] = CameraConfig(**cam_config)
            self.logger.info(f"Loaded {len(self.cameras)} cameras from {self.config_file}")
        except Exception as e:
            self.logger.error(f"Error loading config: {e}. No cameras will be started.", exc_info=True)

    def start(self):
        if not self.cameras:
            self.logger.error("No cameras configured. System will not start.")
            return

        self.logger.info("Starting Multi-Camera Surveillance System")
        self.running = True
        self.batch_processor.start()

        for camera_id, config in self.cameras.items():
            worker = ImprovedCameraWorker(config, self.batch_processor, self.db_manager)
            worker.start()
            self.camera_workers[camera_id] = worker

        self.result_thread = threading.Thread(target=self._process_batch_results, daemon=True)
        self.result_thread.start()

    def _process_batch_results(self):
        while self.running:
            try:
                batch_result = self.batch_processor.get_results()
                if batch_result is None:
                    time.sleep(0.005)
                    continue
                
                for camera_id, detections in batch_result.camera_results.items():
                    worker = self.camera_workers.get(camera_id)
                    if worker and worker.is_active:
                        frame = worker.get_latest_frame()
                        if frame is not None:
                            result = worker.process_detections(detections, frame)
                            self.new_frame_ready.emit(camera_id, result.frame)
                            for id1, id2, distance in result.close_pairs:
                                timestamp_str = datetime.now().strftime("%H:%M:%S")
                                self.violation_detected.emit(camera_id, id1, id2, distance, timestamp_str)
            except Exception as e:
                self.logger.error(f"Error processing batch results: {e}", exc_info=True)
                time.sleep(0.01)

    def stop(self):
        self.logger.info("Stopping surveillance system...")
        self.running = False
        if self.batch_processor:
            self.batch_processor.stop()
        for worker in self.camera_workers.values():
            worker.stop()
        for worker in self.camera_workers.values():
            if worker.is_alive():
                worker.join(timeout=2.0)
        self.logger.info("Surveillance system stopped.")
        self.system_stopped.emit()