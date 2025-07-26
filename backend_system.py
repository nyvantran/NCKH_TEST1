# backend_system.py

import torch
import cv2
import time
import numpy as np
import threading
import queue
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
import json
from datetime import datetime
import os
import sqlite3
import warnings
from PyQt5.QtCore import QObject, pyqtSignal
from BirdEyeViewTransform import BirdEyeViewTransform

# --- Các lớp bạn đã cung cấp ---
# (Dán các lớp FrameBatch, BatchResult, DetectionResult, CameraConfig,
# BatchProcessor, DatabaseManager, PersonTracker, Track, ImprovedCameraWorker
# vào đây, không cần thay đổi gì so với phiên bản trước)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclass
class FrameBatch:
    camera_frames: Dict[str, np.ndarray]
    camera_metadata: Dict[str, Dict]
    batch_id: int
    timestamp: float


@dataclass
class BatchResult:
    batch_id: int
    camera_results: Dict[str, List[Dict]]
    processing_time: float
    timestamp: float


@dataclass
class DetectionResult:
    camera_id: str
    frame_id: int
    timestamp: float
    detections: List[Dict]
    close_pairs: List[Tuple[int, int, float]]
    frame: np.ndarray = None


@dataclass
class CameraConfig:
    camera_id: str
    source: str
    position: str
    enable_recording: bool = True
    recording_path: str = None
    confidence_threshold: float = 0.5
    social_distance_threshold: float = 2.0
    warning_duration: float = 1.0
    loop_video: bool = True


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


class DatabaseManager:
    def __init__(self, db_path="surveillance.db"):
        self.db_path = db_path
        # ... (code giữ nguyên)

    pass


class Track:
    def __init__(self, track_id, detection):
        self.id = track_id
        self.bbox = detection['bbox']
        self.center = detection['center']
        self.confidence = detection['confidence']
        self.height_pixels = detection['height_pixels']
        self.disappeared = 0
        self.trail = deque([detection['center']], maxlen=30)

    def update(self, detection):
        self.bbox = detection['bbox']
        self.center = detection['center']
        self.confidence = detection['confidence']
        self.height_pixels = detection['height_pixels']
        self.disappeared = 0
        self.trail.append(detection['center'])


class PersonTracker:

    def __init__(self, camera_id: str, config: CameraConfig):
        self.camera_id = camera_id
        self.config = config
        self.tracks = {}
        self.next_id = 1
        self.max_disappeared = 30
        self.max_distance = 150  # Tăng nhẹ để ổn định hơn
        self.SOCIAL_DISTANCE_THRESHOLD = config.social_distance_threshold
        self.WARNING_DURATION = config.warning_duration
        self.bev_distance = BirdEyeViewTransform()
        self.bev_distance.load_config_BEV(f"config_BEV_{self.camera_id}.json")
        self.frame_count = 0
        self.current_fps = 30
        self.distance_history = defaultdict(lambda: deque(maxlen=int(self.current_fps * self.WARNING_DURATION * 1.5)))
        self.warned_pairs = set()
        self.colors = [tuple(np.random.randint(64, 255, 3).tolist()) for _ in range(100)]
        self.logger = logging.getLogger(f"Tracker-{camera_id}")

    def calculate_real_distance(self, center1, center2, height1, height2):
        xy_leg1 = (center1[0], center1[1] + height1 / 2)
        xy_leg2 = (center2[0], center2[1] + height2 / 2)
        return self.bev_distance.calculate_distance(xy_leg1, xy_leg2)

    def update_tracks(self, detections):
        active_track_ids = list(self.tracks.keys())
        if not detections:
            for track_id in active_track_ids:
                self.tracks[track_id].disappeared += 1
            return
        if not self.tracks:
            for det in detections:
                self.tracks[self.next_id] = Track(self.next_id, det)
                self.next_id += 1
            return
        cost_matrix = np.zeros((len(active_track_ids), len(detections)))
        for i, track_id in enumerate(active_track_ids):
            for j, det in enumerate(detections):
                dist = np.linalg.norm(np.array(self.tracks[track_id].center) - np.array(det['center']))
                cost_matrix[i, j] = dist
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_track_ids = set()
        assigned_det_indices = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.max_distance:
                track_id = active_track_ids[r]
                self.tracks[track_id].update(detections[c])
                assigned_track_ids.add(track_id)
                assigned_det_indices.add(c)
        unassigned_track_ids = set(active_track_ids) - assigned_track_ids
        for track_id in unassigned_track_ids:
            self.tracks[track_id].disappeared += 1
        new_det_indices = set(range(len(detections))) - assigned_det_indices
        for i in new_det_indices:
            self.tracks[self.next_id] = Track(self.next_id, detections[i])
            self.next_id += 1
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.disappeared <= self.max_disappeared}

    def monitor_distances_and_draw(self, frame):
        active_tracks = {tid: t for tid, t in self.tracks.items() if t.disappeared == 0}
        close_pairs_info = []
        newly_warned_pairs_data = []

        # Draw tracks first
        for tid, track in active_tracks.items():
            x1, y1, x2, y2 = track.bbox
            color = self.colors[tid % len(self.colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'ID: {tid}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Monitor and draw violation lines
        active_track_list = list(active_tracks.items())
        for i in range(len(active_track_list)):
            for j in range(i + 1, len(active_track_list)):
                id1, track1 = active_track_list[i]
                id2, track2 = active_track_list[j]
                distance = self.calculate_real_distance(track1.center, track2.center, track1.height_pixels,
                                                        track2.height_pixels)
                pair_key = tuple(sorted((id1, id2)))
                self.distance_history[pair_key].append(distance)

                if distance < self.SOCIAL_DISTANCE_THRESHOLD:
                    close_pairs_info.append(((id1, id2), distance))
                    close_frames = sum(1 for d in self.distance_history[pair_key] if d < self.SOCIAL_DISTANCE_THRESHOLD)
                    if self.current_fps > 0:
                        close_time = close_frames / self.current_fps
                        if close_time >= self.WARNING_DURATION and pair_key not in self.warned_pairs:
                            self.warned_pairs.add(pair_key)
                            newly_warned_pairs_data.append((id1, id2, distance))
                else:
                    self.warned_pairs.discard(pair_key)

        for (id1, id2), distance in close_pairs_info:
            track1, track2 = self.tracks[id1], self.tracks[id2]
            cv2.line(frame, track1.center, track2.center, (0, 0, 255), 2)
            mid_point = ((track1.center[0] + track2.center[0]) // 2, (track1.center[1] + track2.center[1]) // 2)
            cv2.putText(frame, f'{distance:.1f}m', mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return newly_warned_pairs_data


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
            time.sleep(1 / 35)  # Giới hạn FPS
        self.cleanup()

    def _open_video_source(self):
        try:
            source = self.config.source
            if isinstance(source, str) and source.isdigit(): source = int(source)
            self.cap = cv2.VideoCapture(source)
            if self.cap.isOpened():
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
        return result

    def stop(self):
        self.running = False

    def cleanup(self):
        if self.cap: self.cap.release()
        self.logger.info(f"Camera {self.config.camera_id} stopped.")


# Lớp hệ thống chính kế thừa QObject để phát tín hiệu
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
