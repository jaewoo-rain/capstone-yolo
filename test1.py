import cv2
import os
import time
from roboflow import Roboflow
from dotenv import load_dotenv

"""
실행하면 웹캠이 열림
Space 키 → 프레임 캡처 + Roboflow 자동 업로드
Q 키 → 종료
"""

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
PROJECT = os.getenv("ROBOFLOW_PROJECT")
SAVE_DIR = "captured_images"

os.makedirs(SAVE_DIR, exist_ok=True)

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)

cap = cv2.VideoCapture(0)
count = 0

print("Space: 캡처 & 업로드 | Q: 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow("Webcam - Space: Capture, Q: Quit", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        filename = f"{SAVE_DIR}/capture_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"저장: {filename}")

        try:
            project.upload(filename)
            print(f"Roboflow 업로드 완료 ({count + 1}장)")
            count += 1
        except Exception as e:
            print(f"업로드 실패: {e}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"총 {count}장 업로드 완료")
