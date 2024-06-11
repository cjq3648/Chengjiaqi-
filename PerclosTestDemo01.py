import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
import dlib
from scipy.spatial import distance as dist
from collections import deque

# 初始化MTCNN和dlib的预测器
mtcnn = MTCNN(keep_all=True, select_largest=False, device='cuda' if torch.cuda.is_available() else 'cpu')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = min(A,B) / C
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    mar = (A + B + C) / (3.0 * D)
    return mar

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

EAR_THRESHOLD = 0.23
MOUTH_AR_THRESH = 0.6
YAWN_FREQUENCY_THRESHOLD = 3  # 假设在短时间内（例如几分钟内）超过此次数即认为频繁
PERCLOS_THRESHOLD = 15.0  # PERCLOS的阈值

closed_eye_frames = deque(maxlen=300)
yawn_detected = False
yawn_count = 0

# 在初始化部分
yawn_state = 'closed'  # 初始假设嘴巴是闭合的
cooldown_frames = 90  # 设定冷却时间为90帧
current_cooldown = 0  # 当前冷却剩余帧数

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes.astype(int):
            x, y, max_x, max_y = box
            rect = dlib.rectangle(left=x, top=y, right=max_x, bottom=max_y)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shape = predictor(gray, rect)
            shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

            leftEye = shape[42:48]
            rightEye = shape[36:42]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

            mouth = shape[48:68]
            mar = mouth_aspect_ratio(mouth)

            closed_eye_frames.append(1 if ear < EAR_THRESHOLD else 0)

            if current_cooldown > 0:
                current_cooldown -= 1  # 减少冷却帧数
                continue  # 在冷却期间不做任何状态更新
            else:
                if mar > MOUTH_AR_THRESH:
                    if yawn_state == 'closed':  # 如果之前嘴巴是闭合的
                        yawn_state = 'open'  # 更新状态为张开
                else:
                    if yawn_state == 'open':  # 如果之前嘴巴是张开的
                        yawn_count += 1  # 计数一次打哈欠
                        yawn_state = 'closed'  # 更新状态为闭合
                        current_cooldown = cooldown_frames  # 设置冷却时间

    # 如果没有检测到人脸，平滑PERCLOS值
    if boxes is None or len(boxes) == 0:
        closed_eye_frames.append(closed_eye_frames[-1] if len(closed_eye_frames) > 0 else 0)

    perclos = (sum(closed_eye_frames) / len(closed_eye_frames)) * 100 if closed_eye_frames else 0

    cv2.putText(frame, f"PERCLOS: {perclos:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Yawns: {yawn_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 显示疲劳警告
    if yawn_count > YAWN_FREQUENCY_THRESHOLD:
        cv2.putText(frame, "WARNING: Yawning too frequently!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if perclos > PERCLOS_THRESHOLD:
        cv2.putText(frame, "WARNING: High PERCLOS, you may be fatigued!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
