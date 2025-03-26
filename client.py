import tkinter as tk
from tkinter import messagebox
import requests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

SERVER_URL = "http://127.0.0.1:5000"

class TaixiuClient:
    def __init__(self, root):
        self.root = root
        self.root.title("Tài Xỉu Dự Đoán - Pro Edition")
        self.root.geometry("1000x800")
        self.root.configure(bg="#1e1e2f")

        self.window_size = 10  #
        self.entries = []
        self.session = [0] * self.window_size

        self.title_label = tk.Label(root, text="AI TÀI XỈU",
                                    font=("Helvetica", 20, "bold"), bg="#1e1e2f", fg="#ecf0f1")
        self.title_label.pack(pady=20)

        self.input_frame = tk.Frame(root, bg="#1e1e2f")
        self.input_frame.pack()
        for i in range(self.window_size):
            entry = tk.Entry(self.input_frame, width=5, font=("Helvetica", 14), justify="center",
                             bg="#34495e", fg="#ecf0f1", insertbackground="#ecf0f1")
            entry.grid(row=0, column=i, padx=5, pady=5)
            entry.insert(0, "0")
            self.entries.append(entry)

        self.predict_btn = tk.Button(root, text="DỰ ĐOÁN", command=self.predict,
                                     font=("Helvetica", 14, "bold"), bg="#3498db", fg="#ffffff", width=15)
        self.predict_btn.pack(pady=15)

        self.result_label = tk.Label(root, text="Dự đoán: Chưa có", font=("Helvetica", 16, "bold"), 
                                     bg="#1e1e2f", fg="#f1c40f")
        self.result_label.pack(pady=10)

        # Frame phản hồi
        self.feedback_frame = tk.Frame(root, bg="#1e1e2f")
        self.feedback_frame.pack(pady=10)
        tk.Button(self.feedback_frame, text="Đúng ✅", command=self.feedback_correct,
                  font=("Helvetica", 12, "bold"), bg="#2ecc71", fg="#ffffff").pack(side=tk.LEFT, padx=15)
        tk.Button(self.feedback_frame, text="Sai ❌", command=self.feedback_incorrect,
                  font=("Helvetica", 12, "bold"), bg="#e74c3c", fg="#ffffff").pack(side=tk.LEFT, padx=15)

        self.stats_label = tk.Label(root, text="Thống kê: Đang tải...", font=("Helvetica", 14),
                                    bg="#1e1e2f", fg="#bdc3c7")
        self.stats_label.pack(pady=5)

        self.fig, self.ax = plt.subplots(figsize=(7, 3), facecolor="#1e1e2f")
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget.get_tk_widget().pack()
        self.update_trend()

    def predict(self):
        try:
            self.session = [int(entry.get()) for entry in self.entries]
            response = requests.post(f"{SERVER_URL}/predict", json={"session": self.session}, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get("prediction", "Lỗi")
                confidence = data.get("confidence", 0)
                stats = data.get("stats", {})
                
                self.result_label.config(text=f"Dự đoán: {prediction} (Xác suất: {confidence:.2%})")
                self.stats_label.config(text=f"Số lần chơi: {stats.get('total_records', 0)} | "
                                            f"Trung bình: {stats.get('average_total', 0):.1f} | "
                                            f"Tài Ratio: {stats.get('tai_ratio', 0) * 100:.1f}% | "
                                            f"Streak: {stats.get('streak', 0)}")
                self.update_trend()
            else:
                messagebox.showerror("Lỗi", f"Server trả về lỗi: {response.status_code}")
        except requests.exceptions.RequestException:
            messagebox.showerror("Lỗi", "Không thể kết nối đến server!")
        except ValueError:
            messagebox.showerror("Lỗi", "Dữ liệu nhập không hợp lệ! Vui lòng nhập số.")

    def feedback_correct(self):
        if not self.session:
            messagebox.showerror("Lỗi", "Vui lòng dự đoán trước!")
            return
        actual = self.session[-1]
        requests.post(f"{SERVER_URL}/feedback", json={"actual_result": actual})
        self.shift_entries(actual)
        messagebox.showinfo("Phản hồi", "Đã ghi nhận kết quả đúng!")

    def feedback_incorrect(self):
        try:
            actual = int(self.entries[-1].get())
            requests.post(f"{SERVER_URL}/feedback", json={"actual_result": actual})
            self.shift_entries(actual)
            messagebox.showinfo("Phản hồi", "Đã ghi nhận kết quả sai!")
        except ValueError:
            messagebox.showerror("Lỗi", "Vui lòng nhập kết quả thực tế hợp lệ!")

    def shift_entries(self, actual_result=None):
        if actual_result is not None:
            self.session = self.session[1:] + [actual_result]
        for i, entry in enumerate(self.entries):
            entry.delete(0, tk.END)
            entry.insert(0, str(self.session[i]))

    def update_trend(self):
        try:
            response = requests.post(f"{SERVER_URL}/predict", json={"session": self.session}, timeout=5)
            if response.status_code == 200:
                stats = response.json().get("stats", {})
                trend = stats.get("tai_ratio", 0.5)

                self.ax.clear()
                bars = self.ax.bar(["Tài", "Xỉu"], [trend, 1 - trend], color=["#e74c3c", "#3498db"])
                self.ax.set_ylim(0, 1)
                self.ax.set_title("Xu hướng gần đây (50 lần)", color="#ecf0f1", fontsize=14)
                self.ax.set_facecolor("#2a2a4a")
                self.fig.patch.set_facecolor("#1e1e2f")
                for bar in bars:
                    height = bar.get_height()
                    self.ax.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2%}", 
                                 ha="center", va="bottom", color="#ecf0f1", fontsize=12)
                self.canvas_widget.draw()
        except requests.exceptions.RequestException:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = TaixiuClient(root)
    root.mainloop()