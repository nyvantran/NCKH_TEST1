import cv2
import numpy as np
import logging
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
from BackEnd.core.BirdEyeViewTransform import BirdEyeViewTransform
from BackEnd.common.DataClass import CameraConfig
from BackEnd.data import DatabaseManager
import time

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
        self.bev_distance.load_config_BEV(f"D:\WorkSpace\\model clone\\NCKH_TEST1\\config\\config_BEV_{self.camera_id}.json")
        
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
    
    def get_statistics(self):
        """Get current statistics"""
        active_tracks = sum(1 for track in self.tracks.values() if track.disappeared == 0)
        return {
            'active_tracks': active_tracks,
            'total_tracks': len(self.tracks),
            'violations': len(self.warned_pairs)
        }
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