import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

model = YOLO("runs/detect/test1_model/weights/best.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

align = rs.align(rs.stream.color)
depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        results = model(color_image, verbose=False)
        annotated = results[0].plot()

        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            depth_m = depth_image[cy, cx] * depth_scale

            if depth_m > 0:
                point = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_m)
                rx, ry, rz = point

                label = f"X:{rx:.3f} Y:{ry:.3f} Z:{rz:.3f}m"
                cv2.putText(annotated, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

        cv2.imshow("RealSense D405 + YOLO", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
