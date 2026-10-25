import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

ALGORITHMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "algorithms")
sys.path.insert(0, ALGORITHMS_DIR)

from loader import to_gray  # noqa: E402
from edge import detect_roi  # noqa: E402
from duty_profile import (track_finger_points, angle_from_points, find_deskew_angle,  # noqa: E402
                           find_deskew_angle_fft, deskew)

MAX_DISPLAY = 420  # per-panel max display size (two panels side by side)
MANUAL_ANGLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manual_angles.json")


def load_manual_angles():
    if os.path.exists(MANUAL_ANGLES_PATH):
        with open(MANUAL_ANGLES_PATH) as f:
            return json.load(f)
    return {}


def save_manual_angle(image_path, angle):
    angles = load_manual_angles()
    angles[os.path.abspath(image_path)] = angle
    with open(MANUAL_ANGLES_PATH, "w") as f:
        json.dump(angles, f, indent=2)


class DeskewCheckApp:
    def __init__(self, root):
        self.root = root
        root.title("Deskew Angle Check")

        top = tk.Frame(root)
        top.pack(side="top", fill="x", padx=8, pady=8)
        tk.Button(top, text="Select Image(s)...", command=self.select_images).pack(side="left")
        tk.Button(top, text="< Prev", command=self.prev_image).pack(side="left", padx=(8, 0))
        tk.Button(top, text="Next >", command=self.next_image).pack(side="left")
        self.nav_var = tk.StringVar(value="No images loaded")
        tk.Label(top, textvariable=self.nav_var, anchor="w").pack(side="left", padx=8)

        info = tk.Frame(root)
        info.pack(side="top", fill="x", padx=8)
        self.info_var = tk.StringVar(value="")
        tk.Label(info, textvariable=self.info_var, anchor="w", justify="left", font=("Segoe UI", 10, "bold")).pack(
            fill="x")
        self.manual_var = tk.StringVar(value="Click 2 points along one metal finger in the left panel to set the angle manually.")
        tk.Label(info, textvariable=self.manual_var, anchor="w", justify="left", fg="#0a5").pack(fill="x")

        manual_row = tk.Frame(root)
        manual_row.pack(side="top", fill="x", padx=8)
        tk.Button(manual_row, text="Clear Points", command=self.clear_points).pack(side="left")
        self.save_btn = tk.Button(manual_row, text="Save Manual Angle for This Image", command=self.save_current,
                                   state="disabled")
        self.save_btn.pack(side="left", padx=(8, 0))
        self.save_fft_btn = tk.Button(manual_row, text="Save FFT Angle for This Image", command=self.save_fft,
                                       state="disabled")
        self.save_fft_btn.pack(side="left", padx=(8, 0))

        preview_row = tk.Frame(root)
        preview_row.pack(side="top", fill="x", padx=8)
        tk.Label(preview_row, text="Preview:").pack(side="left")
        self.preview_mode = tk.StringVar(value="fft")
        for value, label in [("auto", "Automatic (tracked)"), ("fft", "FFT (periodic)"), ("manual", "Manual (click)")]:
            tk.Radiobutton(preview_row, text=label, variable=self.preview_mode, value=value,
                           command=self.refresh_deskewed_panel).pack(side="left", padx=(4, 0))

        canvas_frame = tk.Frame(root)
        canvas_frame.pack(side="top", fill="both", expand=True, padx=8, pady=8)

        left_col = tk.Frame(canvas_frame)
        left_col.pack(side="left", padx=(0, 4))
        tk.Label(left_col, text="Original (click 2 points along a finger; auto-tracked line in red)").pack()
        self.canvas_orig = tk.Canvas(left_col, bg="black", cursor="crosshair")
        self.canvas_orig.pack()
        self.canvas_orig.bind("<Button-1>", self.on_click)

        right_col = tk.Frame(canvas_frame)
        right_col.pack(side="left", padx=(4, 0))
        tk.Label(right_col, text="Deskewed (lines should be horizontal)").pack()
        self.canvas_deskewed = tk.Canvas(right_col, bg="black")
        self.canvas_deskewed.pack()

        self.image_paths = []
        self.index = -1
        self.tk_img_orig = None
        self.tk_img_deskewed = None
        self.zoom = 1.0
        self.gray = None
        self.auto_angle = None
        self.manual_points = []  # up to 2 (x, y) points in ORIGINAL image pixel space

    def select_images(self):
        paths = filedialog.askopenfilenames(
            title="Select image(s)",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if not paths:
            return
        self.image_paths = list(paths)
        self.index = 0
        self.show_current()

    def prev_image(self):
        if not self.image_paths:
            return
        self.index = (self.index - 1) % len(self.image_paths)
        self.show_current()

    def next_image(self):
        if not self.image_paths:
            return
        self.index = (self.index + 1) % len(self.image_paths)
        self.show_current()

    def clear_points(self):
        self.manual_points = []
        self.manual_var.set("Click 2 points along one metal finger in the left panel to set the angle manually.")
        self.save_btn.config(state="disabled")
        self.redraw_original()
        self.refresh_deskewed_panel()

    def show_current(self):
        path = self.image_paths[self.index]
        self.nav_var.set(f"{self.index + 1}/{len(self.image_paths)}: {os.path.basename(path)}")
        self.manual_points = []
        self.save_btn.config(state="disabled")
        self.save_fft_btn.config(state="disabled")
        self.root.update()

        try:
            self.gray = to_gray(path)
            roi = detect_roi(path, roi_height=400, show=False)
            self.x_left, self.x_right = roi["x_left"], roi["x_right"]
            points = track_finger_points(self.gray, self.x_left, self.x_right)
            # find_deskew_angle (not angle_from_points directly) -- the automatic
            # tracker needs its own separately-validated sign, see duty_profile.py
            self.auto_angle = find_deskew_angle(self.gray, self.x_left, self.x_right)
            self.tracked_points = points
            try:
                self.fft_angle = find_deskew_angle_fft(self.gray, self.x_left, self.x_right)
                self.save_fft_btn.config(state="normal")
            except Exception as e:
                self.fft_angle = None
                self.save_fft_btn.config(state="disabled")
                print(f"FFT angle failed for {path}: {e}")
        except Exception as e:
            self.info_var.set(f"FAILED: {e}")
            self.canvas_orig.delete("all")
            self.canvas_deskewed.delete("all")
            self.gray = None
            return

        saved = load_manual_angles().get(os.path.abspath(path))
        saved_note = f"   [saved manual angle: {saved:.2f} deg]" if saved is not None else ""
        fft_note = f"{self.fft_angle:.2f} deg" if self.fft_angle is not None else "FAILED"
        self.info_var.set(
            f"x_left={self.x_left} x_right={self.x_right}   automatic (tracked)={self.auto_angle:.2f} deg"
            f"   fft={fft_note}{saved_note}")
        self.manual_var.set("Click 2 points along one metal finger in the left panel to set the angle manually.")

        h, w = self.gray.shape[:2]
        self.zoom = min(MAX_DISPLAY / w, MAX_DISPLAY / h, 4.0)
        self.redraw_original()
        self.refresh_deskewed_panel()

    def on_click(self, event):
        if self.gray is None:
            return
        x_orig = event.x / self.zoom
        y_orig = event.y / self.zoom
        if len(self.manual_points) >= 2:
            self.manual_points = []
        self.manual_points.append((x_orig, y_orig))
        self.redraw_original()

        if len(self.manual_points) == 2:
            manual_angle = angle_from_points(self.manual_points)
            self.manual_var.set(f"Manual angle from your 2 points: {manual_angle:.2f} deg")
            self.save_btn.config(state="normal")
            self.refresh_deskewed_panel()
        else:
            self.manual_var.set("Click a second point along the same finger.")
            self.save_btn.config(state="disabled")

    def redraw_original(self):
        orig_color = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        for (x, y) in self.tracked_points:
            cv2.circle(orig_color, (int(x), int(y)), 1, (0, 0, 255), -1)
        for (x, y) in self.manual_points:
            cv2.drawMarker(orig_color, (int(x), int(y)), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
        if len(self.manual_points) == 2:
            p0 = tuple(int(v) for v in self.manual_points[0])
            p1 = tuple(int(v) for v in self.manual_points[1])
            cv2.line(orig_color, p0, p1, (0, 255, 0), 1)

        self.tk_img_orig = self._to_photoimage(orig_color, is_bgr=True)
        self.canvas_orig.config(width=self.tk_img_orig.width(), height=self.tk_img_orig.height())
        self.canvas_orig.delete("all")
        self.canvas_orig.create_image(0, 0, anchor="nw", image=self.tk_img_orig)

    def refresh_deskewed_panel(self):
        if self.gray is None:
            return
        mode = self.preview_mode.get()
        if mode == "manual" and len(self.manual_points) == 2:
            angle = angle_from_points(self.manual_points)
        elif mode == "fft" and self.fft_angle is not None:
            angle = self.fft_angle
        else:
            angle = self.auto_angle
        deskewed = deskew(self.gray, angle)
        self.tk_img_deskewed = self._to_photoimage(deskewed, is_bgr=False)
        self.canvas_deskewed.config(width=self.tk_img_deskewed.width(), height=self.tk_img_deskewed.height())
        self.canvas_deskewed.delete("all")
        self.canvas_deskewed.create_image(0, 0, anchor="nw", image=self.tk_img_deskewed)

    def save_current(self):
        if len(self.manual_points) != 2:
            return
        angle = angle_from_points(self.manual_points)
        path = self.image_paths[self.index]
        save_manual_angle(path, angle)
        messagebox.showinfo("Saved", f"Saved manual angle {angle:.2f} deg for {os.path.basename(path)}")
        self.info_var.set(
            f"x_left={self.x_left} x_right={self.x_right}   automatic (tracked)={self.auto_angle:.2f} deg"
            f"   [saved manual angle: {angle:.2f} deg]")

    def save_fft(self):
        if self.fft_angle is None:
            return
        path = self.image_paths[self.index]
        save_manual_angle(path, self.fft_angle)
        messagebox.showinfo("Saved", f"Saved FFT angle {self.fft_angle:.2f} deg for {os.path.basename(path)}")
        self.info_var.set(
            f"x_left={self.x_left} x_right={self.x_right}   automatic (tracked)={self.auto_angle:.2f} deg"
            f"   [saved manual angle: {self.fft_angle:.2f} deg (from FFT)]")

    def _to_photoimage(self, img, is_bgr):
        h, w = img.shape[:2]
        disp_w, disp_h = max(1, int(w * self.zoom)), max(1, int(h * self.zoom))
        interp = cv2.INTER_NEAREST if self.zoom >= 1 else cv2.INTER_AREA
        resized = cv2.resize(img, (disp_w, disp_h), interpolation=interp)
        if is_bgr:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return ImageTk.PhotoImage(Image.fromarray(resized))


if __name__ == "__main__":
    root = tk.Tk()
    DeskewCheckApp(root)
    root.mainloop()
