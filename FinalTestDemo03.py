"""优化root登录查询日志功能"""
import mysql.connector
from mysql.connector import Error
import tkinter as tk
from tkinter import messagebox
import cv2
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image, ImageTk
from torchvision import transforms
from datetime import datetime, timedelta
import threading
import time
import dlib
import numpy as np
from collections import deque


class LoginApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Login and Register")
        self.master.geometry("800x400")
        self.master.resizable(False, False)
        self.master.configure(bg="#f0f0f0")

        window_width = 800
        window_height = 400
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.master.geometry("+{}+{}".format(x, y))

        self.login_frame = tk.Frame(self.master, bg="#f0f0f0")
        self.login_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.username_label = tk.Label(self.login_frame, text="Username:", bg="#f0f0f0", fg="#333",
                                       font=("Helvetica", 12))
        self.username_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.username_entry = tk.Entry(self.login_frame, bg="#f0f0f0", bd=0, font=("Helvetica", 12))
        self.username_entry.grid(row=0, column=1, padx=10, pady=5)

        self.password_label = tk.Label(self.login_frame, text="Password:", bg="#f0f0f0", fg="#333",
                                       font=("Helvetica", 12))
        self.password_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.password_entry = tk.Entry(self.login_frame, show="*", bg="#f0f0f0", bd=0, font=("Helvetica", 12))
        self.password_entry.grid(row=1, column=1, padx=10, pady=5)

        self.login_button = tk.Button(self.login_frame, text="Login", command=self.login, bg="#4CAF50", fg="white",
                                      font=("Helvetica", 12, "bold"), padx=10, pady=5)
        self.login_button.grid(row=2, columnspan=2, pady=10)

        self.register_button = tk.Button(self.login_frame, text="Register", command=self.open_register_window,
                                         bg="#2196F3", fg="white", font=("Helvetica", 12, "bold"), padx=10, pady=5)
        self.register_button.grid(row=3, columnspan=2, pady=10)

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

        # Initialize PERCLOS detection parameters
        self.frames_per_second = 30
        self.PERCLOS_THRESHOLD = 30.0
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
        self.log_window.geometry("800x600")
        self.log_window.resizable(True, True)
        self.log_window.configure(bg="#f0f0f0")

        # 为root用户添加选择用户的下拉菜单
        if self.current_user == "root":
            self.user_list = self.get_all_users()  # 假设存在一个方法来获取所有用户列表
            self.selected_user = tk.StringVar()
            self.user_dropdown = tk.OptionMenu(self.log_window, self.selected_user, *self.user_list)
            self.user_dropdown.pack()

            view_logs_button = tk.Button(self.log_window, text="View Logs",
                                         command=lambda: self.view_user_logs(self.selected_user.get()))
            view_logs_button.pack()

        scrollbar = tk.Scrollbar(self.log_window)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(self.log_window, height=30, width=100, bg="white", bd=0, font=("Helvetica", 10),
                                yscrollcommand=scrollbar.set)
        self.log_text.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
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
        self.register_window = tk.Toplevel(self.master)
        self.register_window.title("Register")
        self.register_window.geometry("300x200")
        self.register_window.resizable(False, False)
        self.register_window.configure(bg="#f0f0f0")

        username_label = tk.Label(self.register_window, text="Username:", bg="#f0f0f0", fg="#333",
                                  font=("Helvetica", 12))
        username_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        username_entry = tk.Entry(self.register_window, bg="#f0f0f0", bd=0, font=("Helvetica", 12))
        username_entry.grid(row=0, column=1, padx=10, pady=5)

        password_label = tk.Label(self.register_window, text="Password:", bg="#f0f0f0", fg="#333",
                                  font=("Helvetica", 12))
        password_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        password_entry = tk.Entry(self.register_window, show="*", bg="#f0f0f0", bd=0, font=("Helvetica", 12))
        password_entry.grid(row=1, column=1, padx=10, pady=5)

        register_button = tk.Button(self.register_window, text="Register",
                                    command=lambda: self.register(username_entry.get(), password_entry.get()),
                                    bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), padx=10, pady=5)
        register_button.grid(row=2, columnspan=2, pady=10)

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
        self.clear_log_table()
        self.camera_frame = tk.Frame(self.master, bg="white")
        self.camera_frame.place(relx=0, rely=0, relwidth=0.75, relheight=1)

        self.log_frame = tk.Frame(self.master, bg="#f0f0f0")
        self.log_frame.place(relx=0.75, rely=0, relwidth=0.25, relheight=1)

        self.log_label = tk.Label(self.log_frame, text="Log Information", bg="#f0f0f0", fg="#333",
                                  font=("Helvetica", 12))
        self.log_label.pack(pady=5)

        self.log_text = tk.Text(self.log_frame, height=20, width=40, bg="white", bd=0, font=("Helvetica", 10))
        self.log_text.pack(pady=5, fill=tk.BOTH, expand=True)

        self.log_text.insert(tk.END, "Log started at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

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
        PERCLOS_THRESHOLD = 15.0  # PERCLOS的阈值

        while True:
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

            perclos = (sum(self.closed_eye_frames) / len(self.closed_eye_frames)) * 100 if self.closed_eye_frames else 0

            prediction = self.predict_fatigue(frame, self.model)

            if perclos >= self.PERCLOS_THRESHOLD:
                status = "眼部疲劳"
            elif prediction == 1:
                status = "身体疲劳"
            else:
                status = "积极"

            now = datetime.now()
            self.log_text.insert(tk.END, f"（{now.strftime('%Y-%m-%d %H:%M:%S')}）：{status}\n")
            self.log_text.see(tk.END)
            self.log_text.update_idletasks()

            self.save_log_to_database(self.username_entry.get(),now, status)

            cv2.putText(frame, f"PERCLOS: {perclos:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            #cv2.putText(frame, f"Yawns: {yawn_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            self.update_camera_frame(frame)

            fatigue_count %= self.frames_per_second
            time.sleep(1)

        cap.release()
        print("Camera released")

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
            query = "INSERT INTO logs (user,timestamp, status) VALUES (%s, %s, %s)"
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
            self.camera_label = tk.Label(self.camera_frame, image=frame, bg="white")
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
