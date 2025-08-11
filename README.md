# NCKH_TEST1

## 📋 Mô tả dự án

pass ([123](./config/cameras.json))

## ✨ Tính năng chính

- **Chức năng 1**: Chức năng pass.
- **Chức năng 2**: Chức năng pass.
- **Chức năng 3**: Chức năng pass.
- **Chức năng 4**: Chức năng pass.
- **Chức năng 5**: Chức năng pass.
- **Chức năng 6**: Chức năng pass.

## 🚀 Công nghệ sử dụng

- **Ngôn ngữ lập trình**: Python
- **Framework GUI**: PyQt5
- **Thư viện xử lý ảnh**: OpenCV
- **Thư viện chạy model AI**: PyTorch

[//]: # (- **Face Detection - RetinaFace**)

[//]: # (- **Face Recognition - GhostFaceNets**:)

- **Cơ sở dữ liệu**: SQLite

## 📦 Cài đặt

### Yêu cầu hệ thống

- Python version 3.12 trở lên
- DeskTop có camera
- Windows/macOS/Linux

### Cài đặt dependencies phải sửa

```bash
# Clone repository
git clone https://github.com/nyvantran/NCKH_YOLOv5_social_distancing.git
cd NCKH_YOLOv5_social_distancing

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### requirements.txt

```
opencv-python==4.8.0.74
numpy==1.24.3
pandas==2.0.2
Pillow==9.5.0
```

## 🎯 Cách sử dụng

### 1. Khởi chạy ứng dụng

```bash
python main.py
```

### 2. Chức năng 1

1. pass
2. pass
3. pass
4. pass

### 3. Chức năng 2

1. pass
2. pass
3. pass

### 4. Chức năng 3

1. pass
2. pass
3. pass

## 📁 Cấu trúc project

```
NCKH_YOLOv5_social_distancing
│   .gitignore
│   main.py
│   README.md
│   surveillance.db
│   yolov5m.pt
│
├───BackEnd
│   │   config.py
│   │   MultiCameraSurveillanceSystem.py
│   │
│   ├───common
│   │       DataClass.py
│   │
│   ├───core
│   │       BatchProcessor.py
│   │       BirdEyeViewTransform.py
│   │       ImprovedCameraWorker.py
│   │       PersonTracker.py
│   │   
│   └───data
│           DatabaseManager.py
│   
├───capture      
│       27-07-2025 10-03-16.jpg
│       27-07-2025 10-03-42.jpg
│       27-07-2025 10-03-57.jpg
│       27-07-2025 10-04-11.jpg
│       27-07-2025 10-04-22.jpg
│    
├───config
│       cameras.json
│       config_BEV_CAM001.json
│       config_BEV_CAM002.json
│       config_BEV_CAM003.json
│       config_BEV_CAM004.json
│
└───FontEnd
        gui_app.py
    


```

## 🔧 Cấu hình

### Cấu hình camera

- **cameras**: là cấu hình của các camera trong hệ thống
  - **camera_id**: là id của camera, định dạng là CAM001, CAM002, ...
  - **source**: là đường dẫn đến camera hoặc video, có thể là `0` cho camera mặc định hoặc đường dẫn đến file video
  - **position**: là vị trí của camera trong hệ thống, có thể là `Position_1`, `Position_2`, ...
  - **enable_recording**: có ghi hình hay không, giá trị là `true` hoặc `false`
  - **recording_path**: là đường dẫn lưu video, ví dụ `./recordings`
  - **confidence_threshold**: là ngưỡng tin cậy để nhận diện người, giá trị từ `0.0` đến `1.0`
  - **social_distance_threshold**: là ngưỡng khoảng cách xã hội, giá trị tính bằng mét
  - **warning_duration**: là thời gian cảnh báo khi vi phạm khoảng cách xã hội, tính bằng giây
  - **loop_video**: có lặp lại video hay không, giá trị là `true` hoặc `false`
  - **frame_height**: là chiều cao của khung hình, tính bằng pixel
  - **frame_width**: là chiều rộng của khung hình, tính bằng pixel

```json
{
  "cameras": [
    {
      "camera_id": "CAM001",
      "source": "0",
      "position": "Position_2",
      "enable_recording": true,
      "recording_path": "./recordings",
      "confidence_threshold": 0.4,
      "social_distance_threshold": 2,
      "warning_duration": 1,
      "loop_video": true,
      "frame_height": 720,
      "frame_width": 1280
    }
  ]
}
```

### Cấu hình BEV Transform

khởi chạy file /BackEnd/core/BirdEyeViewTransform.py cách config là chọn 4 điểm trên ảnh và tọa độ 4 điểm trên thực
tế. [video hướng dẫn config BEV](/video_demo_config.mp4)

```bash
python /BackEnd/core/BirdEyeViewTransform.py
```

[//]: # (## 📊 Tính năng 1)

[//]: # ()
[//]: # (- **Nhận diện nhiều khuôn mặt**: Có thể nhận diện đồng thời nhiều sinh viên)

[//]: # (- **Chống gian lận**: Phát hiện ảnh giả, video replay &#40;đang tích hợp&#41;)

## 🐛 Troubleshooting

### Lỗi camera không hoạt động

```bash #sẽ sửa
# Kiểm tra camera
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Lỗi cài đặt dlib

```bash

```

### Lỗi nhận diện kém

- Kiểm tra ánh sáng
- Điều chỉnh confidence_threshold

## 📈 Roadmap

[//]: # (## 🤝 Đóng góp)

[//]: # ()
[//]: # (1. Fork dự án)

[//]: # (2. Tạo branch tính năng &#40;`git checkout -b feature/AmazingFeature`&#41;)

[//]: # (3. Commit thay đổi &#40;`git commit -m 'Add some AmazingFeature'`&#41;)

[//]: # (4. Push lên branch &#40;`git push origin feature/AmazingFeature`&#41;)

[//]: # (5. Tạo Pull Request)

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👥 Tác giả

[//]: # (- **namkuner** - *Developer* - [GitHub]&#40;https://github.com/namkuner&#41;)

## 📞 Liên hệ

[//]: # ()

[//]: # (- Email: namkuner@gmail.com)

[//]: # (- GitHub: [@namkuner]&#40;https://github.com/namkuner&#41;)

[//]: # (- LinkedIn:[Nam Phạm]&#40;https://www.linkedin.com/in/nam-pha%CC%A3m-b94697257/&#41;)

[//]: # (- Youtube: [namkuner]&#40;https://www.youtube.com/@namkuner&#41;)

[//]: # (- FaceBook: [Nam Phạm]&#40;https://www.facebook.com/nam.pham.927201/&#41;)

## 🙏 Acknowledgments

---

⭐ **Nếu project này hữu ích, hãy cho một star nhé!** ⭐
