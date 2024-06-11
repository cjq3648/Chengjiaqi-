import torch
import torch.nn as nn
import os
import mysql.connector
from mysql.connector import Error
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
from datetime import datetime
import threading
import cv2
import dlib
import numpy as np
from collections import deque
from torchvision import models, transforms
from facenet_pytorch import MTCNN

# 定义卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(3*3*512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 6)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = CNN().to(device)
model_path = os.path.join(os.getcwd(), 'cnn2_model.pth')  # 使用绝对路径
cnn_model.load_state_dict(torch.load(model_path, map_location=device))
cnn_model.eval()

# 定义预处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 表情标签
labels = ['disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 初始化MTCNN
mtcnn = MTCNN(keep_all=True, device=device)

class LoginApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Login and Register")
        self.master.geometry("1280x800")
        self.master.resizable(False, False)
        self.master.configure(bg="#1e1e1e")  # Dark background

        window_width = 1280
        window_height = 800
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.master.geometry("+{}+{}".format(x, y))

        self.login_frame = tk.Frame(self.master, bg="#1e1e1e")
        self.login_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.username_label = tk.Label(self.login_frame, text="Username:", bg="#1e1e1e", fg="#dcdcdc", font=("Helvetica", 20))
        self.username_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.username_entry = tk.Entry(self.login_frame, bg="#2d2d2d", fg="#dcdcdc", bd=0, font=("Helvetica", 20))
        self.username_entry.grid(row=0, column=1, padx=10, pady=5)

        self.password_label = tk.Label(self.login_frame, text="Password:", bg="#1e1e1e", fg="#dcdcdc", font=("Helvetica", 20))
        self.password_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.password_entry = tk.Entry(self.login_frame, show="*", bg="#2d2d2d", fg="#dcdcdc", bd=0, font=("Helvetica", 20))
        self.password_entry.grid(row=1, column=1, padx=10, pady=5)

        self.login_button = tk.Button(self.login_frame, text="Login", command=self.login, bg="DarkSeaGreen", fg="LightGoldenrodYellow", font=("Helvetica", 20, "bold"), padx=111, pady=5)
        self.login_button.grid(row=2, columnspan=2, pady=2)

        self.register_button = tk.Button(self.login_frame, text="Register", command=self.open_register_window, bg="IndianRed", fg="LightGoldenrodYellow", font=("Helvetica", 20, "bold"), padx=120, pady=5)
        self.register_button.grid(row=3, columnspan=2, pady=2)

        self.exit_button = tk.Button(self.login_frame, text="Exit", command=self.exit_application, bg="FireBrick", fg="LightGoldenrodYellow", font=("Helvetica", 20, "bold"), padx=120, pady=5)
        self.exit_button.grid(row=4, columnspan=2, pady=2)

        try:
            self.connection = mysql.connector.connect(
                host='localhost',
                user='root',
                password='password',
                database='forward_test'
            )
            print("Connected to MySQL database")
        except Error as e:
            print("Error connecting to MySQL database:", e)
            messagebox.showerror("Error", "Failed to connect to database")

        # Initialize dlib face and landmark detector
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Initialize PERCLOS detection parameter
        self.frames_per_second = 30
        self.PERCLOS_THRESHOLD = 25.0
        self.closed_eye_frames = deque(maxlen=self.frames_per_second)

        # Initialize MobileNetV2 model
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)
        self.model.load_state_dict(torch.load('fatigue_detection_model(1).pth', map_location=torch.device('cpu')))
        self.model.eval()

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        if self.validate_user(username, password):
            self.current_user = username  # 设置当前用户
            if username == "root":
                self.login_frame.place_forget()
                self.open_log_window()  # 打开展示日志的窗口
            else:
                self.exit_flag = False  # 添加退出标志
                threading.Thread(target=self.open_camera).start()
                self.login_frame.place_forget()
        else:
            messagebox.showerror("Error", "Invalid username or password")

    def get_all_users(self):
        try:
            cursor = self.connection.cursor()
            query = "SELECT username FROM users"
            cursor.execute(query)
            users = cursor.fetchall()  # 获取所有结果
            cursor.close()
            return [user[0] for user in users]  # 返回用户名列表
        except Error as e:
            print("获取用户列表错误:", e)
            return []

    def open_log_window(self, user=None):
        self.log_window = tk.Toplevel(self.master)
        self.log_window.title("Log Information")
        self.log_window.geometry("1280x800")
        self.log_window.resizable(True, True)
        self.log_window.configure(bg="#1e1e1e")  # Dark background

        # 为root用户添加选择用户的下拉菜单
        if self.current_user == "root":
            self.user_list = self.get_all_users()  # 假设存在一个方法来获取所有用户列表
            self.selected_user = tk.StringVar()
            self.selected_user.set(self.user_list[0])  # 设置默认值
            self.user_dropdown = tk.OptionMenu(self.log_window, self.selected_user, *self.user_list)

            # 修改OptionMenu的配置来设置宽度
            self.user_dropdown.config(bg="#2d2d2d", fg="#dcdcdc", font=("Helvetica", 30), width=30)  # 设置宽度为20
            self.user_dropdown.pack(pady=10)

            view_logs_button = tk.Button(self.log_window, text="View Logs", command=lambda: self.view_user_logs(self.selected_user.get()), bg="DimGray", fg="LightGoldenrodYellow", font=("Helvetica", 30, "bold"),width=40)
            view_logs_button.pack(pady=10)

        scrollbar = ttk.Scrollbar(self.log_window, style="TScrollbar")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        style = ttk.Style()
        style.theme_use('default')
        style.configure("TScrollbar", gripcount=0,
                        background="DimGray", darkcolor="DimGray", lightcolor="DimGray",
                        troughcolor="#2d2d2d", bordercolor="#2d2d2d", arrowcolor="white")

        self.log_text = tk.Text(self.log_window, height=30, width=100, bg="#2d2d2d", fg="#dcdcdc", bd=2, font=("Helvetica", 20), yscrollcommand=scrollbar.set)
        self.log_text.pack(pady=80, padx=80, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)

        if user:
            self.view_user_logs(user)

    def view_user_logs(self, user):
        self.log_text.delete(1.0, tk.END)  # 清除旧日志
        logs = self.fetch_logs(user, self.current_user == "root")
        for log in logs:
            log_entry = f"ID: {log[0]}, User: {log[1]}, Timestamp: {log[2]}, Status: {log[3]}\n" if self.current_user == "root" else f"ID: {log[0]}, Timestamp: {log[1]}, Status: {log[2]}\n"
            self.log_text.insert(tk.END, log_entry)

    def fetch_logs(self, user, is_root=False):
        try:
            cursor = self.connection.cursor()
            if is_root:
                # root用户可以查看所有日志，或指定用户的日志
                query = "SELECT id, user, timestamp, status FROM logs WHERE user = %s ORDER BY timestamp DESC"
                cursor.execute(query, (user,))
            else:
                # 非root用户只能查看自己的日志
                query = "SELECT id, timestamp, status FROM logs WHERE user = %s ORDER BY timestamp DESC"
                cursor.execute(query, (user,))
            logs = cursor.fetchall()
            cursor.close()
            return logs
        except Error as e:
            print("Error fetching logs:", e)
            return []

    def validate_user(self, username, password):
        try:
            cursor = self.connection.cursor()
            query = "SELECT * FROM users WHERE username = %s AND password = %s"
            cursor.execute(query, (username, password))
            user = cursor.fetchone()
            cursor.close()
            if user:
                return True
            else:
                return False
        except Error as e:
            print("Error validating user:", e)
            return False

    def open_register_window(self):
        self.master.destroy()  # 关闭原窗口

        self.register_window = tk.Tk()  # 创建新的Tk窗口
        self.register_window.title("Register")
        self.register_window.geometry("1280x800")  # 设置窗口大小与初始窗口一致
        self.register_window.resizable(False, False)
        self.register_window.configure(bg="#1e1e1e")  # Dark background

        container_frame = tk.Frame(self.register_window, bg="#1e1e1e")  # 创建容器框架
        container_frame.place(relx=0.5, rely=0.5, anchor="center")  # 容器框架居中

        username_label = tk.Label(container_frame, text="Username:", bg="#1e1e1e", fg="#dcdcdc", font=("Helvetica", 20))
        username_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")
        username_entry = tk.Entry(container_frame, bg="#2d2d2d", fg="#dcdcdc", bd=0, font=("Helvetica", 20))
        username_entry.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        password_label = tk.Label(container_frame, text="Password:", bg="#1e1e1e", fg="#dcdcdc", font=("Helvetica", 20))
        password_label.grid(row=1, column=0, padx=10, pady=5, sticky="e")
        password_entry = tk.Entry(container_frame, show="*", bg="#2d2d2d", fg="#dcdcdc", bd=0, font=("Helvetica", 20))
        password_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        register_button = tk.Button(container_frame, text="Register",
                                    command=lambda: self.register(username_entry.get(), password_entry.get()),
                                    bg="PaleVioletRed", fg="LightGoldenrodYellow", font=("Helvetica", 20, "bold"), padx=120, pady=5)
        register_button.grid(row=2, columnspan=2, pady=20)

        # 设置网格布局的列权重，使内容居中对齐
        container_frame.grid_columnconfigure(0, weight=1)
        container_frame.grid_columnconfigure(1, weight=1)

    def register(self, username, password):
        if self.check_existing_user(username):
            messagebox.showerror("Error", "Username already exists")
            return
        if username and password:
            try:
                cursor = self.connection.cursor()
                query = "INSERT INTO users (username, password) VALUES (%s, %s)"
                cursor.execute(query, (username, password))
                self.connection.commit()
                cursor.close()
                messagebox.showinfo("Success", "Registration successful")
                self.register_window.destroy()
            except Error as e:
                print("Error registering user:", e)
                messagebox.showerror("Error", "Failed to register user")
        else:
            messagebox.showerror("Error", "Username and password cannot be empty")

    def check_existing_user(self, username):
        try:
            cursor = self.connection.cursor()
            query = "SELECT * FROM users WHERE username = %s"
            cursor.execute(query, (username,))
            user = cursor.fetchone()
            cursor.close()
            if user:
                return True
            else:
                return False
        except Error as e:
            print("Error checking existing user:", e)
            return True

    def open_camera(self):
        self.camera_frame = tk.Frame(self.master, bg="#1e1e1e")
        self.camera_frame.place(relx=0, rely=0, relwidth=0.75, relheight=1)

        self.log_frame = tk.Frame(self.master, bg="#1e1e1e")
        self.log_frame.place(relx=0.75, rely=0, relwidth=0.25, relheight=1)

        self.log_label = tk.Label(self.log_frame, text="Log Information", bg="#1e1e1e", fg="#dcdcdc", font=("Helvetica", 17))
        self.log_label.pack(pady=5)

        self.log_text = tk.Text(self.log_frame, height=20, width=40, bg="#2d2d2d", fg="#dcdcdc", bd=0, font=("Helvetica", 17))
        self.log_text.pack(pady=5, fill=tk.BOTH, expand=True)

        self.log_text.insert(tk.END, "Log started at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

        self.exit_button = tk.Button(self.log_frame, text="Exit", command=self.exit_camera, bg="FireBrick", fg="LightGoldenrodYellow", font=("Helvetica", 20, "bold"))
        self.exit_button.pack(pady=10)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera")
            print("Error: Failed to open camera")
            return

        print("Camera opened successfully")

        fatigue_count = 0
        yawn_count = 0
        cooldown_frames = 90
        current_cooldown = 0

        EAR_THRESHOLD = 0.23
        MOUTH_AR_THRESH = 0.6
        YAWN_FREQUENCY_THRESHOLD = 3  # 假设在短时间内（例如几分钟内）超过此次数即认为频繁
        PERCLOS_THRESHOLD = 25.0  # PERCLOS的阈值

        positive_count = 0
        negative_count = 0

        while not self.exit_flag:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to read frame from camera")
                print("Error: Failed to read frame from camera")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)

            for face in faces:
                landmarks = self.landmark_predictor(gray, face)

                left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
                right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])

                ear_left = self.eye_aspect_ratio(left_eye)
                ear_right = self.eye_aspect_ratio(right_eye)

                ear = (ear_left + ear_right) / 2.0

                self.closed_eye_frames.append(1 if ear < EAR_THRESHOLD else 0)

                # 绘制68个面部特征点
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # 绿色点

            perclos = (sum(self.closed_eye_frames) / len(self.closed_eye_frames)) * 100 if self.closed_eye_frames else 0

            prediction = self.predict_fatigue(frame, self.model)

            status = "积极"
            if prediction == 1:
                status = "疲劳"

            now = datetime.now()
            self.log_text.insert(tk.END, f"（{now.strftime('%Y-%m-%d %H:%M:%S')}）{status}\n")
            self.log_text.see(tk.END)
            self.log_text.update_idletasks()

            self.save_log_to_database(self.username_entry.get(), now, status)

            # 使用MTCNN检测人脸
            boxes, _ = mtcnn.detect(frame)

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    face = frame[y1:y2, x1:x2]
                    face_pil = Image.fromarray(face)  # 转换为PIL图像
                    face_tensor = transform(face_pil).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = cnn_model(face_tensor)
                        _, predicted = torch.max(output, 1)
                        label = labels[predicted.item()]

                    if label in ['happy', 'neutral', 'surprise']:
                        positive_count += 1
                    else:
                        negative_count += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # 在视频流上添加 PERCLOS 值
            cv2.putText(frame, f'PERCLOS: {perclos:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 判断 PERCLOS 值是否超过阈值，如果超过，则添加警告信息
            if perclos > PERCLOS_THRESHOLD:
                cv2.putText(frame, 'PERCLOS值过高!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            self.update_camera_frame(frame)

        cap.release()
        print("Camera released")

        # 计算并打印积极表情百分比
        total_count = positive_count + negative_count
        if total_count > 0:
            positive_percentage = (positive_count / total_count) * 100
            print(f'认知负荷分数: {positive_percentage:.2f}%')
            self.save_cognitive_load_score(self.username_entry.get(), positive_percentage)
        else:
            print('未检测到任何表情')

    def save_cognitive_load_score(self, username, score):
        try:
            cursor = self.connection.cursor()
            query = "INSERT INTO userscores (username, score) VALUES (%s, %s) ON DUPLICATE KEY UPDATE score=%s"
            cursor.execute(query, (username, score, score))
            self.connection.commit()
            cursor.close()
        except Error as e:
            print("Error saving cognitive load score to database:", e)

    def exit_camera(self):
        self.exit_flag = True

    def exit_application(self):
        self.master.quit()

    def clear_log_table(self):
        try:
            cursor = self.connection.cursor()
            clear_query = "DELETE FROM logs"
            cursor.execute(clear_query)
            self.connection.commit()
            cursor.close()
        except Error as e:
            print("Error clearing log table:", e)

    def save_log_to_database(self, user, timestamp, status):
        try:
            cursor = self.connection.cursor()
            query = "INSERT INTO logs (user, timestamp, status) VALUES (%s, %s, %s)"
            cursor.execute(query, (user, timestamp, status))
            self.connection.commit()
            cursor.close()
        except Error as e:
            print("Error saving log to database:", e)

    def update_camera_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        self.master.after(0, self.update_camera_frame_ui, frame)

    def update_camera_frame_ui(self, frame):
        frame = ImageTk.PhotoImage(frame)

        if hasattr(self, "camera_label"):
            self.camera_label.configure(image=frame)
            self.camera_label.image = frame
        else:
            self.camera_label = tk.Label(self.camera_frame, image=frame, bg="dimgrey")
            self.camera_label.pack(fill=tk.BOTH, expand=True)

    def predict_fatigue(self, frame, model):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_image = transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            model.eval()
            outputs = model(input_image)
            _, predicted = torch.max(outputs, 1)
            return predicted.item()

    def eye_aspect_ratio(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear


def main():
    root = tk.Tk()
    app = LoginApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
