# YOLO 커스텀 객체 감지 프로젝트

YOLOv11 기반으로 웹캠에서 커스텀 객체를 감지하고 실시간 위치를 추정하는 프로젝트입니다.

## 프로젝트 흐름

1. 웹캠으로 이미지 캡처 → Roboflow 업로드
2. Roboflow에서 바운딩박스 라벨링
3. YOLOv8 커스텀 학습
4. 학습된 모델로 실시간 감지 및 위치 추정

## 감지 클래스

- `case` (5cm x 5cm)
- `test1`

## 파일 설명

| 파일 | 설명 |
|------|------|
| `main.py` | 기본 YOLOv8 웹캠 실행 (yolov8n.pt 기본 모델, 레거시) |
| `test1.py` | 웹캠 캡처 + Roboflow 자동 업로드 |
| `test1_train.py` | 커스텀 데이터셋으로 YOLOv8 학습 |
| `test1_play.py` | 학습된 모델로 웹캠 실시간 감지 |
| `test1_webcam.py` | 웹캠 기반 실시간 위치 추정 (캘리브레이션 필요) |
| `test1_realsense.py` | Intel RealSense D405 기반 정밀 3D 좌표 추정 |

## 사용 방법

### 1. 이미지 수집 및 업로드
```bash
python test1.py
# Space: 캡처 & Roboflow 업로드 | Q: 종료
```

### 2. 데이터셋 준비
- Roboflow에서 바운딩박스 라벨링 후 YOLOv8 형식으로 Export
- `test1/` 폴더에 저장
- `test1/data.yaml`에서 train, val 경로를 절대경로로 수정

### 3. 모델 학습
```bash
python test1_train.py
# 학습 결과: runs/detect/test1_model/weights/best.pt
```

### 4. 실시간 감지 테스트
```bash
python test1_play.py
# Q: 종료
```

### 5. 위치 추정 (웹캠)
```bash
python test1_webcam.py
# case를 30cm 거리에 들고 Space로 캘리브레이션 후 시작
# 출력: X(좌우), Y(상하), Z(거리) 단위: cm
```

### 6. 위치 추정 (RealSense D405)
```bash
python test1_realsense.py
# Intel RealSense D405 연결 필요
# 출력: X(좌우), Y(상하), Z(거리) 단위: m
```

## 설치

```bash
pip install -r requirements.txt
```

## 환경 변수

`.env` 파일을 프로젝트 루트에 생성:

```
ROBOFLOW_API_KEY=your_api_key
ROBOFLOW_WORKSPACE=your_workspace
ROBOFLOW_PROJECT=your_project
```

## 데이터셋 구조

```
test1/
├── train/
│   ├── images/
│   └── labels/
└── data.yaml
```

## 향후 계획

- ROS2 연동으로 로봇에 실시간 좌표 전송
