# # NCKH_YOLOv5_social_distancing

## 📋 Mô tả dự án

Dự án NCKH_YOLOv5_social_distancing là một hệ thống giám sát đa camera sử dụng công nghệ YOLOv5 để phát hiện và theo dõi
người trong các khu vực công cộng,nhằm đảm bảo tuân thủ khoảng cách xã hội.
Hệ thống này sửa dụng Bird eye view transform để đo khoảng cách của mỗi người trong camera với sai số khoảng ± 5%.
Hệ thống bao gồm nhiều camera được cấu hình để phát hiện người, phát âm thanh cảnh báo và ghi lại hình ảnh cảnh báo khi
có vi phạm về khoảng cách xã hội.

## ✨ Tính năng chính

- **Phát hiện con người**: sử dụng camera phát hiện con người.
- **Đo khoảng cách giữa 2 người**: Đo khoảng cách mỗi người trong camera.
- **Cảnh báo vi phạm khoảng cách xã hội**: khi khoảng cách giữa 2 người nhỏ hơn ngưỡng cho phép, hệ thống sẽ cảnh báo
  bằng âm thanh và hiển thị thông báo trên giao diện người dùng. Lưu cảnh báo vào database và ghi hình ảnh cảnh báo.

## 🚀 Công nghệ sử dụng

- **Ngôn ngữ lập trình**: Python
- **Framework GUI**: PyQt5
- **Thư viện xử lý ảnh**: OpenCV
- **Thư viện chạy model AI**: PyTorch
- **Mô hình phát hiện đối tượng**: YOLOv5m
- **Cơ sở dữ liệu**: SQLite
- **Phát âm thanh cảnh báo**: ffplay (một phần của FFmpeg của hệ điều hành)

## 📦 Cài đặt

### Yêu cầu hệ thống

- Python version 3.12 trở lên
- DeskTop có kết nối tới camera, loa
- Windows/macOS/Linux

### Cài đặt FFmpeg

chạy lệnh sau để cài đặt FFmpeg

```bash
# Windows 
winget install "FFmpeg (Essentials Build)"
# macOS
brew install ffmpeg
# Linux
sudo apt update
sudo apt install ffmpeg
```

### Cài đặt dependencies

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

### [requirements.txt](./requirements.txt)

## 🔧 Cấu hình

### Cấu hình camera

- **cameras**: là cấu hình của các camera trong hệ thống
    - **camera_id**: là id của camera, định dạng là CAM001, CAM002, ...
    - **source**: là đường dẫn đến camera hoặc video, có thể là `0`, `1`, `2` ... cho các camera mặc định hoặc đường dẫn
      đến file video
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
tế. [video hướng dẫn config BEV](video/video_demo_config_BEV.mp4)

```bash
python BackEnd/core/BirdEyeViewTransform.py
```

[//]: # (## 📊 Tính năng 1)

[//]: # ()

[//]: # (- **Nhận diện nhiều khuôn mặt**: Có thể nhận diện đồng thời nhiều sinh viên)

[//]: # (- **Chống gian lận**: Phát hiện ảnh giả, video replay &#40;đang tích hợp&#41;)

## 🎯 Cách sử dụng

### 1. Khởi chạy ứng dụng

```bash
python main.py
```

### 2. Xem các đối tượng vi phạm khoảng cách xã hội

- **xem hình ảnh các đối tượng vi phạm**: khi có đối tượng vi phạm khoảng cách xã hội, hệ thống sẽ lưu hình ảnh cảnh báo
  vào thư mục `capture`
- **xem lịch sử vi phạm**: hệ thống sẽ lưu thông tin vi phạm vào cơ sở dữ liệu `surveillance.db`, bạn có thể sử dụng
  các công cụ quản lý SQLite để xem lịch sử vi phạm.

[//]: # ()

[//]: # (### 3. Chức năng 2)

[//]: # ()

[//]: # (1. pass)

[//]: # (2. pass)

[//]: # (3. pass)

[//]: # ()

[//]: # (### 4. Chức năng 3)

[//]: # ()

[//]: # (1. pass)

[//]: # (2. pass)

[//]: # (3. pass)

## 📁 Cấu trúc project

```
NCKH_YOLOv5_social_distancing
│   .gitignore
│   main.py
│   README.md
│   requirements.txt
│   surveillance.db
│   yolov5m.pt
│
├───BackEnd
│   │   config.py
│   │   MultiCameraSurveillanceSystem.py
│   │
│   ├───audio
│   │       CAM001_violation.mp3
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
├───FontEnd
│       gui_app.py
│
└───video
        videotest.mp4
        video_demo_config_BEV.mp4
```

## 🔧 Cấu hình

### Cấu hình camera

- **cameras**: là cấu hình của các camera trong hệ thống
    - **camera_id**: là id của camera, định dạng là CAM001, CAM002, ...
    - **source**: là đường dẫn đến camera hoặc video, có thể là `0`, `1`, `2` ... cho các camera mặc định hoặc đường dẫn
      đến file video
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
tế. [video hướng dẫn config BEV](video/video_demo_config_BEV.mp4)

```bash
python BackEnd/core/BirdEyeViewTransform.py
```

[//]: # (## 📊 Tính năng 1)

[//]: # ()

[//]: # (- **Nhận diện nhiều khuôn mặt**: Có thể nhận diện đồng thời nhiều sinh viên)

[//]: # (- **Chống gian lận**: Phát hiện ảnh giả, video replay &#40;đang tích hợp&#41;)

## 🐛 Troubleshooting

### Lỗi camera không hoạt động hoặc nguồn video không mở được

```bash 
# Kiểm tra camera
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

```bash 
# Kiểm tra video
python -c "import cv2; print(cv2.VideoCapture(\"video//videotest.mp4\").isOpened())" #thay bằng đường dẫn video của bạn"
```

### Lỗi cài đặt dlib

```bash

```

[//]: # (### Lỗi nhận diện kém)

[//]: # ()

[//]: # (- Kiểm tra ánh sáng)

[//]: # (- Điều chỉnh confidence_threshold)

## 📈 Roadmap

[//]: # (## 🤝 Đóng góp)

[//]: # ()

[//]: # (1. Fork dự án)

[//]: # (2. Tạo branch tính năng &#40;`git checkout -b feature/AmazingFeature`&#41;)

[//]: # (3. Commit thay đổi &#40;`git commit -m 'Add some AmazingFeature'`&#41;)

[//]: # (4. Push lên branch &#40;`git push origin feature/AmazingFeature`&#41;)

[//]: # (5. Tạo Pull Request)

[//]: # (## 📄 License)

[//]: # ()

[//]: # (Distributed under the MIT License. See `LICENSE` for more information.)

## 👥 Tác giả

- **TranDoManhDuy** - *Developer* - [GitHub](https://github.com/TranDoManhDuy)
- **nyvantran** - *Developer* - [GitHub](https://github.com/nyvantran)
- **HieuITMHG** - *Developer* - [GitHub](https://github.com/HieuITMHG);

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
