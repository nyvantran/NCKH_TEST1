import torch
import cv2
import queue
import threading
import time
import logging
import numpy as np
from collections import deque
from typing import Dict, List, Optional

from BackEnd.common.DataClass import FrameBatch, BatchResult


class BatchProcessor:
    """Xử lý batch frames từ nhiều camera"""

    def __init__(self, batch_size: int = 8, max_wait_time: float = 0.05):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger = logging.getLogger("BatchProcessor")
        self.logger.info(f"Loading YOLOv5 model on {self.device}...")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        self.logger.info("YOLOv5 model loaded.")
        self.input_queue = queue.Queue(maxsize=100)
        self.output_queue = queue.Queue(maxsize=100)
        self.batch_id_counter = 0
        self.running = False
        self.batch_times = deque(maxlen=100)
        self.processor_thread = threading.Thread(target=self._batch_processing_loop)
        self.processor_thread.daemon = True

    def start(self):
        self.running = True
        self.processor_thread.start()
        self.logger.info(f"BatchProcessor started with batch_size={self.batch_size}")

    def stop(self):
        self.running = False
        if self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5.0)
        self.logger.info("BatchProcessor stopped")

    def add_frame(self, camera_id: str, frame: np.ndarray, metadata: Dict):
        try:
            self.input_queue.put((camera_id, frame, metadata), timeout=0.01)
        except queue.Full:
            self.logger.warning("Batch input queue full, dropping frame")

    def get_results(self) -> Optional[BatchResult]:
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def _batch_processing_loop(self):
        pending_frames = {}
        last_batch_time = time.time()
        while self.running:
            try:
                while len(pending_frames) < self.batch_size:
                    try:
                        camera_id, frame, metadata = self.input_queue.get(timeout=0.01)
                        pending_frames[camera_id] = (frame, metadata)
                    except queue.Empty:
                        break
                current_time = time.time()
                should_process = (len(pending_frames) >= self.batch_size or (
                        len(pending_frames) > 0 and (current_time - last_batch_time) >= self.max_wait_time))
                if should_process and pending_frames:
                    batch = self._create_batch(pending_frames)
                    result = self._process_batch(batch)
                    try:
                        self.output_queue.put(result, timeout=0.01)
                    except queue.Full:
                        self.logger.warning("Batch output queue full")
                    pending_frames.clear()
                    last_batch_time = current_time
                else:
                    time.sleep(0.001)
            except Exception as e:
                self.logger.error(f"Error in batch processing loop: {e}", exc_info=True)
                time.sleep(0.01)

    def _create_batch(self, pending_frames: Dict) -> FrameBatch:
        camera_frames = {cam_id: data[0] for cam_id, data in pending_frames.items()}
        camera_metadata = {cam_id: data[1] for cam_id, data in pending_frames.items()}
        batch_id = self.batch_id_counter
        self.batch_id_counter += 1
        return FrameBatch(camera_frames, camera_metadata, batch_id, time.time())

    def _process_batch(self, batch: FrameBatch) -> BatchResult:
        start_time = time.time()
        batch_images = []
        camera_order = []
        for camera_id, frame in batch.camera_frames.items():
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_images.append(rgb_frame)
            camera_order.append(camera_id)
        with torch.no_grad():
            results = self.model(batch_images, size=640)
        camera_results = {}
        for i, camera_id in enumerate(camera_order):
            detections = self._extract_detections(results.pred[i],
                                                  batch.camera_metadata[camera_id].get('confidence_threshold', 0.5))
            camera_results[camera_id] = detections
        processing_time = time.time() - start_time
        self.batch_times.append(processing_time)
        return BatchResult(batch.batch_id, camera_results, processing_time, time.time())

    def _extract_detections(self, predictions, confidence_threshold: float) -> List[Dict]:
        detections = []
        for *xyxy, conf, cls in predictions:
            if int(cls) == 0 and conf > confidence_threshold:
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append(
                    {'bbox': (x1, y1, x2, y2), 'center': ((x1 + x2) // 2, (y1 + y2) // 2), 'confidence': float(conf),
                     'area': (x2 - x1) * (y2 - y1), 'height_pixels': y2 - y1})
        return detections