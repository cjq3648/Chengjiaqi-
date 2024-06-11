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

class LoginApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Login and Register")
        self.master.geometry("800x400")
        self.master.resizable(False, False)
        self.master.configure(bg="#f0f0f0")

        # Center the window on the screen
        window_width = 800
        window_height = 400
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.master.geometry("+{}+{}".format(x, y))

        self.login_frame = tk.Frame(self.master, bg="#f0f0f0")
        self.login_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.username_label = tk.Label(self.login_frame, text="Username:", bg="#f0f0f0", fg="#333", font=("Helvetica", 12))
        self.username_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.username_entry = tk.Entry(self.login_frame, bg="#f0f0f0", bd=0, font=("Helvetica", 12))
        self.username_entry.grid(row=0, column=1, padx=10, pady=5)

        self.password_label = tk.Label(self.login_frame, text="Password:", bg="#f0f0f0", fg="#333", font=("Helvetica", 12))
        self.password_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.password_entry = tk.Entry(self.login_frame, show="*", bg="#f0f0f0", bd=0, font=("Helvetica", 12))
        self.password_entry.grid(row=1, column=1, padx=10, pady=5)

        self.login_button = tk.Button(self.login_frame, text="Login", command=self.login, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), padx=10, pady=5)
        self.login_button.grid(row=2, columnspan=2, pady=10)

        self.register_button = tk.Button(self.login_frame, text="Register", command=self.open_register_window, bg="#2196F3", fg="white", font=("Helvetica", 12, "bold"), padx=10, pady=5)
        self.register_button.grid(row=3, columnspan=2, pady=10)

        # Connect to MySQL database
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

        # 创建一个新的MobileNetV2模型并加载参数
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)  # 假设你的数据集有两个类别
        self.model.load_state_dict(torch.load('fatigue_detection_model(1).pth', map_location=torch.device('cpu')))
        self.model.eval()

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        if self.validate_user(username, password):
            threading.Thread(target=self.open_camera).start()
            self.login_frame.place_forget()
        else:
            messagebox.showerror("Error", "Invalid username or password")

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
        # Create a new window for registration
        self.register_window = tk.Toplevel(self.master)
        self.register_window.title("Register")
        self.register_window.geometry("300x200")
        self.register_window.resizable(False, False)
        self.register_window.configure(bg="#f0f0f0")

        username_label = tk.Label(self.register_window, text="Username:", bg="#f0f0f0", fg="#333", font=("Helvetica", 12))
        username_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        username_entry = tk.Entry(self.register_window, bg="#f0f0f0", bd=0, font=("Helvetica", 12))
        username_entry.grid(row=0, column=1, padx=10, pady=5)

        password_label = tk.Label(self.register_window, text="Password:", bg="#f0f0f0", fg="#333", font=("Helvetica", 12))
        password_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        password_entry = tk.Entry(self.register_window, show="*", bg="#f0f0f0", bd=0, font=("Helvetica", 12))
        password_entry.grid(row=1, column=1, padx=10, pady=5)

        register_button = tk.Button(self.register_window, text="Register", command=lambda: self.register(username_entry.get(), password_entry.get()), bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), padx=10, pady=5)
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

        self.log_label = tk.Label(self.log_frame, text="Log Information", bg="#f0f0f0", fg="#333", font=("Helvetica", 12))
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

        frames_per_second = 30
        fatigue_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to read frame from camera")
                print("Error: Failed to read frame from camera")
                break

            prediction = self.predict_fatigue(frame, self.model)
            fatigue_count += prediction
            if fatigue_count >= frames_per_second // 2:  # 超过半数判断为疲劳
                status = "疲劳"
            else:
                status = "积极"
            now = datetime.now()
            self.log_text.insert(tk.END, f"（{now.strftime('%Y-%m-%d %H:%M:%S')}）：{status}\n")
            self.log_text.see(tk.END)
            self.log_text.update_idletasks()

            # 保存日志到数据库
            self.save_log_to_database(now, status)

            # 更新摄像头画面
            self.update_camera_frame(frame)

            # 重置每秒的帧数和疲劳判断数量
            fatigue_count %= frames_per_second
            time.sleep(1)  # 等待1秒

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

    def save_log_to_database(self, timestamp, status):
        try:
            cursor = self.connection.cursor()
            query = "INSERT INTO logs (timestamp, status) VALUES (%s, %s)"
            cursor.execute(query, (timestamp, status))
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

def main():
    root = tk.Tk()
    app = LoginApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
