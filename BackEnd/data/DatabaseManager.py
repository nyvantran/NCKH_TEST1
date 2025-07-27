import sqlite3
import time


class DatabaseManager:
    """Quản lý database để lưu trữ kết quả"""

    def __init__(self, db_path: str = "surveillance.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Khởi tạo database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Bảng events
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS events
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           camera_id
                           TEXT,
                           event_type
                           TEXT,
                           timestamp
                           REAL,
                           person_id1
                           INTEGER,
                           person_id2
                           INTEGER,
                           description
                           TEXT
                       )
                       ''')

        # Bảng statistics
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS statistics
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           camera_id
                           TEXT,
                           timestamp
                           REAL,
                           total_persons
                           INTEGER,
                           active_persons
                           INTEGER,
                           violations
                           INTEGER
                       )
                       ''')

        # Bảng performance
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS performance
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           timestamp
                           REAL,
                           batch_size
                           INTEGER,
                           processing_time
                           REAL,
                           fps
                           REAL
                       )
                       ''')

        conn.commit()
        conn.close()

    def log_event(self, camera_id: str, event_type: str, person_id1: int,
                  person_id2: int = None, description: str = ""):
        """Ghi lại sự kiện"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       INSERT INTO events (camera_id, event_type, timestamp, person_id1, person_id2,
                                           description)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ''', (camera_id, event_type, time.time(), person_id1, person_id2, description))

        conn.commit()
        conn.close()

    def log_statistics(self, camera_id: str, total_persons: int, active_persons: int, violations: int):
        """Ghi lại thống kê"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       INSERT INTO statistics (camera_id, timestamp, total_persons, active_persons, violations)
                       VALUES (?, ?, ?, ?, ?)
                       ''', (camera_id, time.time(), total_persons, active_persons, violations))

        conn.commit()
        conn.close()

    def log_performance(self, batch_size: int, processing_time: float, fps: float):
        """Ghi lại performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       INSERT INTO performance (timestamp, batch_size, processing_time, fps)
                       VALUES (?, ?, ?, ?)
                       ''', (time.time(), batch_size, processing_time, fps))

        conn.commit()
        conn.close()