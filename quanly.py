import requests
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patheffects as path_effects

SERVER_URL = "http://127.0.0.1:5000"

class AdminMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Quản lý Server Tài Xỉu")
        self.root.geometry("600x750")
        self.root.configure(bg="#ffffff")

        # Tạo Notebook (các tab)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Tab 1: Chính (chứa các nút điều khiển)
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Chính")

        # Tab 2: Thống Kê (chứa thông tin thống kê và biểu đồ)
        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="Thống Kê")

        # --- Tab Chính ---
        # Tiêu đề
        title_label = tk.Label(self.main_tab, text="Quản lý Server Tài Xỉu", 
                               font=("Helvetica", 24, "bold"), fg="#2c3e50", bg="#e6f3fa")
        title_label.pack(pady=20)

        # Frame chứa các nút
        button_frame = tk.Frame(self.main_tab, bg="#e6f3fa")
        button_frame.pack(pady=20)

        buttons = [
            ("Khởi động huấn luyện ▶", self.start_training),
            ("Dừng huấn luyện ⏹", self.stop_training),
            ("Xóa dữ liệu 🗑", self.clear_data),
            ("Xem thống kê 📊", self.switch_to_stats_tab),
            ("Thoát 🚪", self.root.quit)
        ]
        for text, command in buttons:
            btn = tk.Button(button_frame, text=text, command=command, width=25, 
                            font=("Helvetica", 12, "bold"), bg="#3498db", fg="#ffffff", 
                            activebackground="#2980b9", bd=0)
            btn.pack(pady=15)

        # --- Tab Thống Kê ---
        # Frame chứa thống kê
        stats_frame = tk.LabelFrame(self.stats_tab, text="Thống kê Server", 
                                    font=("Helvetica", 14, "bold"), bg="#f7f9fc", fg="#2c3e50", 
                                    padx=15, pady=15)
        stats_frame.pack(pady=20, padx=20, fill="x")

        self.stats_label = tk.Label(stats_frame, text="Chưa có dữ liệu", 
                                    font=("Helvetica", 12), fg="#2c3e50", bg="#f7f9fc")
        self.stats_label.pack(pady=10)

        # Biểu đồ
        self.fig, self.ax = plt.subplots(figsize=(5, 3), facecolor="#ffffff")
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.stats_tab)
        self.canvas_widget.get_tk_widget().pack(pady=20)
        self.update_stats_chart()

    def start_training(self):
        try:
            response = requests.post(f"{SERVER_URL}/control", json={"action": "start"}, timeout=5)
            if response.status_code == 200:
                messagebox.showinfo("Thành công", response.json()["message"])
            else:
                messagebox.showerror("Lỗi", f"Server trả về lỗi: {response.status_code}")
        except requests.exceptions.RequestException:
            messagebox.showerror("Lỗi", "Không thể kết nối đến server!")

    def stop_training(self):
        try:
            response = requests.post(f"{SERVER_URL}/control", json={"action": "stop"}, timeout=5)
            if response.status_code == 200:
                messagebox.showinfo("Thành công", response.json()["message"])
            else:
                messagebox.showerror("Lỗi", f"Server trả về lỗi: {response.status_code}")
        except requests.exceptions.RequestException:
            messagebox.showerror("Lỗi", "Không thể kết nối đến server!")

    def clear_data(self):
        if messagebox.askyesno("Xác nhận", "Bạn có chắc muốn xóa toàn bộ dữ liệu?"):
            try:
                response = requests.post(f"{SERVER_URL}/control", json={"action": "clear"}, timeout=5)
                if response.status_code == 200:
                    messagebox.showinfo("Thành công", response.json()["message"])
                    self.stats_label.config(text="Chưa có dữ liệu")
                    self.update_stats_chart()
                else:
                    messagebox.showerror("Lỗi", f"Server trả về lỗi: {response.status_code}")
            except requests.exceptions.RequestException:
                messagebox.showerror("Lỗi", "Không thể kết nối đến server!")

    def switch_to_stats_tab(self):
        # Chuyển sang tab "Thống Kê" và cập nhật dữ liệu
        self.notebook.select(self.stats_tab)
        self.show_stats()

    def show_stats(self):
        try:
            response = requests.post(f"{SERVER_URL}/control", json={"action": "stats"}, timeout=5)
            if response.status_code == 200:
                stats = response.json()
                text = (f"Tổng số kết quả: {stats['total_records']}\n"
                        f"Trung bình tổng: {stats['average_total']:.1f}\n"
                        f"Tỷ lệ Tài (50 lần): {stats['tai_ratio']:.2%}\n"
                        f"Chuỗi dài nhất (20 lần): {stats['streak']}")
                self.stats_label.config(text=text)
                self.update_stats_chart(stats)
            else:
                messagebox.showerror("Lỗi", f"Server trả về lỗi: {response.status_code}")
        except requests.exceptions.RequestException:
            messagebox.showerror("Lỗi", "Không thể kết nối đến server!")

    def update_stats_chart(self, stats=None):
        if stats is None:
            stats = {"tai_ratio": 0.5}
        self.ax.clear()
        trend = stats.get("tai_ratio", 0.5)
        bars = self.ax.bar(["Tài", "Xỉu"], [trend, 1 - trend], color=["#e74c3c", "#3498db"], 
                           edgecolor="#2c3e50", linewidth=1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Tỷ lệ Tài/Xỉu (50 lần gần nhất)", color="#2c3e50", fontweight="bold", fontsize=14)
        self.ax.set_facecolor("#f7f9fc")
        self.fig.patch.set_facecolor("#ffffff")
        self.ax.tick_params(axis="x", colors="#2c3e50", labelsize=12)
        self.ax.tick_params(axis="y", colors="#2c3e50", labelsize=12)
        for bar in bars:
            bar.set_alpha(0.9)
            height = bar.get_height()
            text = self.ax.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2%}", 
                                ha="center", va="bottom", color="#2c3e50", fontweight="bold", fontsize=12)
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground="#ffffff"), path_effects.Normal()])
        self.canvas_widget.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AdminMenu(root)
    root.mainloop()