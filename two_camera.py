import cv2
import customtkinter as ctk
from customtkinter import CTkImage

from PIL import Image, ImageTk, ImageDraw
import time
import datetime
import os
import glob

# กำหนดธีม - ใช้ธีม minimal และสีส้ม
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")  # ใช้เป็นพื้นฐานแล้วจะกำหนดสีส้มเอง

# สร้างโฟลเดอร์สำหรับบันทึกวิดีโอ
recordings_folder = "recordings"
if not os.path.exists(recordings_folder):
    os.makedirs(recordings_folder)

def get_available_cameras(max_cams=5):
    available = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            available.append(f"Camera {i}")
        cap.release()
    return available

class DualCameraApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sitting Posture Recorder")
        self.geometry("1300x800")
        self.fps = 10
        self.frame_interval = int(1000 / self.fps)
        self.last_time = time.perf_counter()
        self.current_fps = 0

        # สีหลักของแอพ
        self.accent_color = "#FF7D3B"  # สีส้ม
        self.text_color = "#333333"    # สีเทาเข้ม
        self.bg_color = "#F5F5F5"      # สีเทาอ่อน
        self.secondary_color = "#FFE0CC"  # สีส้มอ่อน

        # กำหนดสีพื้นหลัก
        self.configure(fg_color=self.bg_color)

        # ตั้งค่ากล้อง
        self.cap1 = cv2.VideoCapture(0)
        self.cap2 = cv2.VideoCapture(1)
        
        # ตัวแปรสำหรับการบันทึก
        self.recording = False
        self.out1 = None
        self.out2 = None
        self.recording_start_time = None
        self.recording_duration = "00:00:00"
        self.current_filename = ""
        self.current_posture = ""
        
        # ตัวแปรสำหรับตาราง log
        self.recording_logs = []

        # สร้าง layout
        self.create_layout()
        
        # โหลดประวัติการบันทึก
        self.load_recording_history()
        
        # เริ่มต้นอัพเดทเฟรม
        self.update_frames()

    def create_layout(self):
        # สร้าง main container
        self.main_frame = ctk.CTkFrame(self, fg_color=self.bg_color, corner_radius=0)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # ✅ ลดความสูง top_frame ให้พอดี
        self.top_frame = ctk.CTkFrame(self.main_frame, fg_color=self.bg_color, corner_radius=0, height=520)
        self.top_frame.pack(fill="x")

        # ✅ ไม่ใส่ expand=True แล้ว เพราะมันจะกินเต็มจน bottom_frame ตก
        # สร้าง bottom_frame รองรับตาราง
        self.bottom_frame = ctk.CTkFrame(self.main_frame, fg_color=self.bg_color, corner_radius=0, width=700, height=250)
        self.bottom_frame.pack(fill="x", padx=(0, 0), pady=(20, 0))

        # แบ่ง top_frame เป็นกล้อง 1 กล้อง 2 และ control panel
        self.camera1_frame = ctk.CTkFrame(self.top_frame, fg_color=self.bg_color, corner_radius=0)
        self.camera1_frame.pack(side="left", fill="both", expand=True)

        self.camera2_frame = ctk.CTkFrame(self.top_frame, fg_color=self.bg_color, corner_radius=0)
        self.camera2_frame.pack(side="left", fill="both", expand=True, padx=(20, 0))

        self.control_frame = ctk.CTkScrollableFrame(self.top_frame, fg_color=self.bg_color, corner_radius=0, width=300, height=500)
        self.control_frame.pack(side="right", fill="y", padx=(20, 0))
        self.control_frame.pack_propagate(True)
        
        # ============ CAMERA FRAMES ============
        # กรอบสำหรับกล้อง 1
        self.video_frame1 = ctk.CTkFrame(self.camera1_frame, fg_color="#FFFFFF", corner_radius=15)
        self.video_frame1.pack(fill="both", expand=True)
        
        self.video_label1 = ctk.CTkLabel(self.video_frame1, text="", corner_radius=10)
        self.video_label1.pack(fill="both", expand=True, padx=10, pady=10)
        
        # กรอบสำหรับกล้อง 2
        self.video_frame2 = ctk.CTkFrame(self.camera2_frame, fg_color="#FFFFFF", corner_radius=15)
        self.video_frame2.pack(fill="both", expand=True)
        
        self.video_label2 = ctk.CTkLabel(self.video_frame2, text="", corner_radius=10)
        self.video_label2.pack(fill="both", expand=True, padx=10, pady=10)
        
        # ============ CONTROL FRAME ============
        # ชื่อแอพ
        self.title_label = ctk.CTkLabel(
            self.control_frame, 
            text="Sitting Posture Recorder", 
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=self.accent_color
        )
        self.title_label.pack(pady=(20, 30))
        
        # Camera Selection
        self.create_camera_selection_section()
        # Posture Selection
        self.create_posture_section()
        
        # FPS Setting
        self.create_fps_section()
        
        # การบันทึกเวลา
        self.create_timer_section()
        
        # Recording Buttons
        self.create_recording_buttons()

        self.countdown_label = ctk.CTkLabel(
            self.control_frame,
            text="",
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color=self.accent_color
        )
        self.countdown_label.pack(pady=(0, 10))

        
        # สถานะ - แก้ไขส่วนนี้
        self.status_label = ctk.CTkLabel(
            self.control_frame, 
            text="พร้อมบันทึก", 
            font=ctk.CTkFont(size=14),
            text_color=self.text_color
        )
        self.status_label.pack(pady=(20, 0))
        
        # ============ LOG TABLE FRAME ============
        self.create_log_table()

    def create_posture_section(self):
        posture_container = ctk.CTkFrame(self.control_frame, fg_color=self.bg_color, corner_radius=0)
        posture_container.pack(fill="x", pady=(0, 20))
        
        posture_label = ctk.CTkLabel(
            posture_container, 
            text="ท่านั่ง:", 
            font=ctk.CTkFont(size=16),
            text_color=self.text_color
        )
        posture_label.pack(anchor="w", pady=(0, 5))
        
        self.posture_var = ctk.StringVar(value="Forward")
        self.posture_dropdown = ctk.CTkComboBox(
            posture_container, 
            variable=self.posture_var,
            values=[
                "Forward", "Backward", "Lean", "Cross-legged",
                "Feet Supported", "Chin on Hand"
            ],
            width=250,
            fg_color="#FFFFFF",
            border_color=self.accent_color,
            button_color=self.accent_color,
            button_hover_color="#E66C2C",  # สีส้มเข้มขึ้นเมื่อ hover
            dropdown_fg_color="#FFFFFF",
            dropdown_hover_color=self.secondary_color,
            dropdown_text_color=self.text_color,
            text_color=self.text_color,
            font=ctk.CTkFont(size=14),
            corner_radius=8,
            command=self.update_posture_preview  # Add this line
        )
        self.posture_dropdown.pack(fill="x")
        
        # Create image label for preview
        self.posture_image_label = ctk.CTkLabel(posture_container, text="")
        self.posture_image_label.pack(pady=(10, 0))
        
        # Initial preview update
        self.update_posture_preview()

    def update_posture_preview(self, choice=None):
        posture = self.posture_var.get().replace(" ", "_").lower()
        image_path = os.path.join("posture_images", f"{posture}.jpg")

        if os.path.exists(image_path):
            self.posture_image_pil = Image.open(image_path)  # เก็บ PIL image
            self.posture_image_obj = CTkImage(self.posture_image_pil, size=(200, 200))  # เก็บ CTkImage

            self.posture_image_label.configure(image=self.posture_image_obj, text="")
        else:
            self.posture_image_label.configure(image=None, text="No image available")

    def create_fps_section(self):
        fps_container = ctk.CTkFrame(self.control_frame, fg_color=self.bg_color, corner_radius=0)
        fps_container.pack(fill="x", pady=(0, 20))
        
        fps_label = ctk.CTkLabel(
            fps_container, 
            text="FPS:", 
            font=ctk.CTkFont(size=16),
            text_color=self.text_color
        )
        fps_label.pack(anchor="w", pady=(0, 5))
        
        fps_input_frame = ctk.CTkFrame(fps_container, fg_color=self.bg_color, corner_radius=0)
        fps_input_frame.pack(fill="x")
        
        self.fps_entry = ctk.CTkEntry(
            fps_input_frame, 
            width=80,
            fg_color="#FFFFFF",
            border_color=self.accent_color,
            text_color=self.text_color,
            font=ctk.CTkFont(size=14),
            corner_radius=8
        )
        self.fps_entry.insert(0, "10")
        self.fps_entry.pack(side="left", padx=(0, 10))
        
        set_fps_button = ctk.CTkButton(
            fps_input_frame, 
            text="ตั้งค่า", 
            command=self.set_fps,
            fg_color=self.accent_color,
            hover_color="#E66C2C",  # สีส้มเข้มขึ้นเมื่อ hover
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=14),
            corner_radius=8,
            width=80
        )
        set_fps_button.pack(side="left")

    def create_camera_selection_section(self):
        camera_options = get_available_cameras()

        # === Camera 1 dropdown ===
        self.cam1_dropdown = ctk.CTkComboBox(
            self.camera1_frame,
            values=camera_options,
            command=self.update_camera_selection_1,
            width=180,
            fg_color="#FFFFFF",
            border_color=self.accent_color,
            button_color=self.accent_color,
            dropdown_fg_color="#FFFFFF", 
            dropdown_text_color=self.text_color,
            font=ctk.CTkFont(size=14)
        )
        self.cam1_dropdown.set("Camera 0")
        self.cam1_dropdown.pack(padx=10, pady=(10, 0), anchor="w")

        # === Camera 2 dropdown ===
        self.cam2_dropdown = ctk.CTkComboBox(
            self.camera2_frame,
            values=camera_options,
            command=self.update_camera_selection_2,
            width=180,
            fg_color="#FFFFFF",
            border_color=self.accent_color,
            button_color=self.accent_color,
            dropdown_fg_color="#FFFFFF",
            dropdown_text_color=self.text_color,
            font=ctk.CTkFont(size=14)
        )
        self.cam2_dropdown.set("Camera 1")
        self.cam2_dropdown.pack(padx=10, pady=(10, 0), anchor="w")

    def create_timer_section(self):
        timer_container = ctk.CTkFrame(self.control_frame, fg_color=self.bg_color, corner_radius=0)
        timer_container.pack(fill="x", pady=(0, 20))
        
        timer_label = ctk.CTkLabel(
            timer_container, 
            text="ระยะเวลาบันทึก:", 
            font=ctk.CTkFont(size=16),
            text_color=self.text_color
        )
        timer_label.pack(anchor="w", pady=(0, 5))
        
        self.timer_display = ctk.CTkLabel(
            timer_container, 
            text=self.recording_duration,
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=self.accent_color
        )
        self.timer_display.pack(anchor="w")

        self.duration_var = ctk.StringVar(value="10")
        self.duration_dropdown = ctk.CTkComboBox(
            timer_container,
            values=["10", "15", "20"],
            variable=self.duration_var,
            width=80,
            fg_color="#FFFFFF",
            border_color=self.accent_color,
            button_color=self.accent_color,
            dropdown_fg_color="#FFFFFF",
            dropdown_text_color=self.text_color,
            font=ctk.CTkFont(size=14)
        )
        self.duration_dropdown.pack(anchor="w", pady=(5, 0))

    def create_recording_buttons(self):
        buttons_container = ctk.CTkFrame(self.control_frame, fg_color=self.bg_color, corner_radius=0)
        buttons_container.pack(fill="x", pady=(10, 20))

        # ▶ ปุ่มเริ่มบันทึก - ชิดซ้าย
        self.start_button = ctk.CTkButton(
            buttons_container,
            text="▶ เริ่มบันทึก",
            command=self.start_recording,
            fg_color=self.accent_color,
            hover_color="#E66C2C",
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=16, weight="bold"),
            corner_radius=10,
            height=50,
            width=140
        )
        self.start_button.pack(side="left", expand=True, padx=10)

        # ■ ปุ่มหยุดบันทึก - ชิดขวา
        self.stop_button = ctk.CTkButton(
            buttons_container,
            text="■ หยุดบันทึก",
            command=self.stop_recording,
            fg_color="#CCCCCC",
            hover_color="#E66C2C",
            text_color="#666666",
            font=ctk.CTkFont(size=16, weight="bold"),
            corner_radius=10,
            height=50,
            width=140,
            state="disabled"  # ปิดการใช้งานปุ่มหยุดในตอนแรก
        )
        self.stop_button.pack(side="right", expand=True, padx=10)

    def create_log_table(self):
        # สร้างกรอบสำหรับตาราง Log
        log_container = ctk.CTkFrame(self.bottom_frame, fg_color="#FFFFFF", corner_radius=15)
        log_container.pack(fill="both", expand=True)
        
        # หัวข้อตาราง
        log_title = ctk.CTkLabel(
            log_container,
            text="ประวัติการบันทึกวิดีโอ",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.accent_color
        )
        log_title.pack(anchor="w", padx=20, pady=(15, 10))
        
        # สร้างตารางหัวข้อ
        header_frame = ctk.CTkFrame(log_container, fg_color="#FFFFFF", corner_radius=0)
        header_frame.pack(fill="x", padx=20, pady=(0, 5))
        
        # ส่วนหัวของตาราง - เพิ่ม "ลบ" ในรายการ headers
        headers = ["วันที่-เวลา", "ท่านั่ง", "ระยะเวลา", "ดูวิดีโอ", "ลบ"]
        widths = [0.3, 0.3, 0.2, 0.1, 0.1]  # ปรับสัดส่วนความกว้าง
        
        for i, header in enumerate(headers):
            header_label = ctk.CTkLabel(
                header_frame,
                text=header,
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=self.text_color
            )
            header_label.pack(side="left", fill="x", expand=True, padx=5)
        
        # เส้นคั่น
        separator = ctk.CTkFrame(log_container, height=1, fg_color="#DDDDDD")
        separator.pack(fill="x", padx=20, pady=5)
        
        # สร้าง scrollable frame สำหรับข้อมูล
        self.log_scroll_frame = ctk.CTkScrollableFrame(
            log_container, 
            fg_color="#FFFFFF", 
            corner_radius=0,
            height=130
        )
        self.log_scroll_frame.pack(fill="both", expand=True, padx=20, pady=(0, 15))

    def load_recording_history(self):
        # ลบข้อมูลเก่าใน log_scroll_frame ทั้งหมด
        for widget in self.log_scroll_frame.winfo_children():
            widget.destroy()
        
        # อ่านไฟล์วิดีโอทั้งหมดในโฟลเดอร์
        video_files = glob.glob(os.path.join(recordings_folder, "*_camera1.mp4"))
        self.recording_logs = []
        
        for video_file in video_files:
            # แยกชื่อไฟล์และนามสกุล
            filename = os.path.basename(video_file)
            # ตัดออก "_camera1.mp4"
            base_name = filename.replace("_camera1.mp4", "")
            
            # แยกท่านั่งและเวลาบันทึก
            parts = base_name.split("_")
            if len(parts) >= 3:  # ตัวอย่าง: normal_20250423_120000
                posture = parts[0]
                date_str = parts[1]
                time_str = parts[2]
                
                # สร้างวันที่และเวลาที่อ่านได้
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                formatted_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                datetime_str = f"{formatted_date} {formatted_time}"
                
                # ดึงข้อมูลเวลาของวิดีโอ (ความยาวของวิดีโอ)
                try:
                    cap = cv2.VideoCapture(video_file)
                    if cap.isOpened():
                        # ดึงจำนวนเฟรมและ FPS
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps > 0:
                            # คำนวณเวลาทั้งหมดเป็นวินาที
                            duration_sec = frame_count / fps
                            mins = int(duration_sec // 60)
                            secs = int(duration_sec % 60)
                            duration_str = f"{mins:02d}:{secs:02d}"
                        else:
                            duration_str = "N/A"
                        cap.release()
                    else:
                        duration_str = "N/A"
                except:
                    duration_str = "N/A"
                
                # เพิ่มข้อมูลลงในรายการ log
                log_entry = {
                    "datetime": datetime_str,
                    "posture": posture.replace("_", " ").title(),
                    "duration": duration_str,
                    "filename": base_name
                }
                self.recording_logs.append(log_entry)
        
        # เรียงลำดับตามเวลาล่าสุดก่อน
        self.recording_logs.sort(key=lambda x: x["datetime"], reverse=True)
        
        # แสดงรายการใน log table
        self.display_log_entries()

    def display_log_entries(self):
        # แสดงรายการใน log table
        for log in self.recording_logs:
            # สร้างแถวใหม่สำหรับแต่ละรายการ
            row_frame = ctk.CTkFrame(self.log_scroll_frame, fg_color="#FFFFFF", corner_radius=0, height=40)
            row_frame.pack(fill="x", pady=5)
            row_frame.pack_propagate(False)
            
            # คอลัมน์ 1: วันที่และเวลา
            date_label = ctk.CTkLabel(
                row_frame,
                text=log["datetime"],
                font=ctk.CTkFont(size=12),
                text_color=self.text_color
            )
            date_label.pack(side="left", fill="x", expand=True, padx=5)
            
            # คอลัมน์ 2: ท่านั่ง
            posture_label = ctk.CTkLabel(
                row_frame,
                text=log["posture"],
                font=ctk.CTkFont(size=12),
                text_color=self.text_color
            )
            posture_label.pack(side="left", fill="x", expand=True, padx=5)
            
            # คอลัมน์ 3: ระยะเวลา
            duration_label = ctk.CTkLabel(
                row_frame,
                text=log["duration"],
                font=ctk.CTkFont(size=12),
                text_color=self.text_color
            )
            duration_label.pack(side="left", fill="x", expand=True, padx=5)
            
            # คอลัมน์ 4: ปุ่มเปิดวิดีโอ
            view_button = ctk.CTkButton(
                row_frame,
                text="เปิดดู",
                command=lambda filename=log["filename"]: self.open_video(filename),
                fg_color=self.accent_color,
                hover_color="#E66C2C",
                text_color="#FFFFFF",
                font=ctk.CTkFont(size=12),
                corner_radius=6,
                width=80,
                height=25
            )
            view_button.pack(side="left", padx=5)
            
            # คอลัมน์ 5: ปุ่มลบวิดีโอ
            delete_button = ctk.CTkButton(
                row_frame,
                text="ลบ",
                command=lambda filename=log["filename"]: self.delete_video(filename),
                fg_color="#D9534F",  # สีแดง
                hover_color="#C9302C",
                text_color="#FFFFFF",
                font=ctk.CTkFont(size=12),
                corner_radius=6,
                width=80,
                height=25
            )
            delete_button.pack(side="left", padx=5)
    
    def delete_video(self, filename):
        try:
            file1 = os.path.join(recordings_folder, f"{filename}_camera1.mp4")
            file2 = os.path.join(recordings_folder, f"{filename}_camera2.mp4")
            if os.path.exists(file1):
                os.remove(file1)
            if os.path.exists(file2):
                os.remove(file2)
            self.status_label.configure(text=f"ลบวิดีโอ: {filename} แล้ว")
            self.load_recording_history()  # อัปเดตตารางใหม่หลังลบ
        except Exception as e:
            self.status_label.configure(text=f"ไม่สามารถลบวิดีโอได้: {str(e)}")

    
    def open_video(self, filename):
        # เปิดไฟล์วิดีโอด้วยโปรแกรมเริ่มต้นของระบบ
        video_path = os.path.join(recordings_folder, f"{filename}_camera1.mp4")
        if os.path.exists(video_path):
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(video_path)
                elif os.name == 'posix':  # macOS และ Linux
                    import subprocess
                    opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
                    subprocess.call([opener, video_path])
                self.status_label.configure(text=f"กำลังเปิดวิดีโอ: {filename}")
            except:
                self.status_label.configure(text="ไม่สามารถเปิดวิดีโอได้")
        else:
            self.status_label.configure(text="ไม่พบไฟล์วิดีโอ")

    def set_fps(self):
        try:
            new_fps = float(self.fps_entry.get())
            if new_fps > 0:
                self.fps = new_fps
                self.frame_interval = int(1000 / self.fps)
                self.status_label.configure(text=f"FPS ถูกตั้งค่าเป็น {new_fps}")
        except ValueError:
            self.status_label.configure(text="ค่า FPS ไม่ถูกต้อง")

    def update_timer(self):
        if self.recording and self.recording_start_time:
            elapsed = time.time() - self.recording_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.recording_duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            self.timer_display.configure(text=self.recording_duration)

            if not self.halfway_notified and elapsed >= (self.recording_target_duration / 2):
                self.status_label.configure(
                        text="ทำการเปลี่ยนท่านั่ง!",
                        font=ctk.CTkFont(size=20, weight="bold"),
                        text_color="#FFA500"  # สีส้ม
                    )
                self.halfway_notified = True

            if elapsed >= self.recording_target_duration - 3 and elapsed < self.recording_target_duration:
                remaining = int(self.recording_target_duration - elapsed)
                self.countdown_label.configure(text=f"หยุดใน {remaining}")
            elif elapsed >= self.recording_target_duration:
                self.countdown_label.configure(text="")
                self.stop_recording()

    def start_recording(self):
        self.countdown_label.configure(text="เตรียมตัว...")
        self.after(1000, lambda: self.countdown_before_start(3))

    def countdown_before_start(self, seconds):
        if seconds > 0:
            self.countdown_label.configure(text=str(seconds))
            self.after(1000, lambda: self.countdown_before_start(seconds - 1))
        else:
            self.countdown_label.configure(text="เริ่มบันทึก!")
            self.after(1000, lambda: self.start_actual_recording())  # Start actual recording

    def start_actual_recording(self):
        posture = self.posture_var.get().replace(" ", "_").lower()
        self.current_posture = self.posture_var.get()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        base_filename = f'{posture}_{timestamp}'
        self.current_filename = base_filename

        filename1 = os.path.join(recordings_folder, f'{base_filename}_camera1.mp4')
        filename2 = os.path.join(recordings_folder, f'{base_filename}_camera2.mp4')

        self.out1 = cv2.VideoWriter(filename1, fourcc, self.fps, (640, 480))
        self.out2 = cv2.VideoWriter(filename2, fourcc, self.fps, (640, 480))

        self.recording = True
        self.recording_start_time = time.time()
        self.recording_target_duration = int(self.duration_var.get())
        self.halfway_notified = False  # Add a flag for halfway notification

        self.start_button.configure(state="disabled", fg_color="#CCCCCC", text_color="#666666")
        self.status_label.configure(text=f"กำลังบันทึก... ({posture})")
        self.countdown_label.configure(text="")  # Clear countdown

    def stop_recording(self):
        self.recording = False
        if self.out1: self.out1.release()
        if self.out2: self.out2.release()
        
        # อัพเดทสถานะและปุ่ม
        self.start_button.configure(state="normal", fg_color=self.accent_color, text_color="#FFFFFF")
        self.stop_button.configure(state="disabled", fg_color="#CCCCCC", text_color="#666666")
        self.status_label.configure(text=f"บันทึกเสร็จสิ้น ระยะเวลา {self.recording_duration}")
        
        # เพิ่มรายการใหม่ลงในประวัติ
        # โหลดประวัติการบันทึกใหม่เพื่อให้มีข้อมูลล่าสุด
        self.load_recording_history()
        
        # รีเซ็ตเวลา
        self.recording_duration = "00:00:00"
        self.timer_display.configure(text=self.recording_duration)

    def update_frames(self):
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()

        now = time.perf_counter()
        self.current_fps = 1 / (now - self.last_time)
        self.last_time = now

        if ret1 and ret2:
            # ปรับขนาดเฟรมให้เป็น 640x480
            frame1 = cv2.resize(frame1, (640, 480))
            frame2 = cv2.resize(frame2, (640, 480))
            
            # แปลงเฟรมเป็น RGB และเตรียมสำหรับแสดงผล
            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            
            # แสดงข้อมูลบนเฟรม
            def add_info_to_frame(frame, camera_num):
                pil_img = Image.fromarray(frame)
                draw = ImageDraw.Draw(pil_img)
                
                # แสดง FPS และหมายเลขกล้อง
                fps_text = f"FPS: {self.current_fps:.2f}"
                camera_text = f"Camera {camera_num}"
                draw.text((20, 20), camera_text, fill=(255, 125, 59), font=None)
                draw.text((20, 45), fps_text, fill=(255, 125, 59), font=None)
                
                if self.recording:
                    # แสดงสถานะการบันทึกและระยะเวลา
                    record_text = f"● REC: {self.recording_duration}"
                    draw.text((20, 70), record_text, fill=(255, 0, 0), font=None)
                
                return pil_img
            
            # เพิ่มข้อความลงในแต่ละเฟรม
            pil_img1 = add_info_to_frame(frame1_rgb, 1)
            pil_img2 = add_info_to_frame(frame2_rgb, 2)
            
            # แปลงเป็น ImageTk สำหรับแสดงผล
            img1 = ImageTk.PhotoImage(pil_img1)
            img2 = ImageTk.PhotoImage(pil_img2)
            
            # แสดงผลบนหน้าจอ
            self.video_label1.configure(image=img1)
            self.video_label1.imgtk = img1
            self.video_label2.configure(image=img2)
            self.video_label2.imgtk = img2

            if self.recording:
                self.out1.write(frame1)
                self.out2.write(frame2)
                self.update_timer()

        self.after(self.frame_interval, self.update_frames)

    def update_camera_selection_1(self, choice):
        index = int(choice.split()[-1])
        if self.cap1.isOpened():
            self.cap1.release()
        if cv2.VideoCapture(index).read()[0]:  # ตรวจสอบก่อนเปิด
            self.cap1 = cv2.VideoCapture(index)
            self.status_label.configure(text=f"Camera 1 → Camera {index}")
        else:
            self.status_label.configure(text=f"Camera {index} ไม่พร้อมใช้งาน")

    def update_camera_selection_2(self, choice):
        index = int(choice.split()[-1])
        if self.cap2.isOpened():
            self.cap2.release()
        if cv2.VideoCapture(index).read()[0]:
            self.cap2 = cv2.VideoCapture(index)
            self.status_label.configure(text=f"Camera 2 → Camera {index}")
        else:
            self.status_label.configure(text=f"Camera {index} ไม่พร้อมใช้งาน")

    def on_closing(self):
        if self.recording:
            self.stop_recording()
        if self.cap1.isOpened():
            self.cap1.release()
        if self.cap2.isOpened():
            self.cap2.release()
        self.destroy()

if __name__ == "__main__":
    app = DualCameraApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    import sys  # เพิ่มการ import sys สำหรับใช้ในฟังก์ชัน open_video
    app.mainloop()